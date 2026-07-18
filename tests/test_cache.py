import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sparseflow.benchmark import generate_trace, load_trace, parse_byte_budgets, run_expert_benchmark
from sparseflow.buffers import BufferPool
from sparseflow.cache import ExpertCache
from sparseflow.cache_policy import make_cache_policy
from sparseflow.loader import ShardReader
from sparseflow.locator import ExpertLocator
from sparseflow.native_moe import prepare_native_expert_batch
from sparseflow.trace import load_route_trace
from sparseflow.policy_benchmark import run_policy_replay
from sparseflow.trace import RouteTrace, TraceGroup


def write_shard(path: Path, tensors):
    offset = 0
    header = {}
    payload = bytearray()
    for name, size, shape in tensors:
        header[name] = {
            "dtype": "U8",
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        payload.extend(bytes((offset + i) % 251 for i in range(size)))
        offset += size
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class ExpertCacheTest(unittest.TestCase):
    def test_buffer_pool_reuses_bounded_size_classes(self):
        pool = BufferPool(max_cached_bytes=8, max_per_size=1)
        first = pool.acquire(4)
        pool.release(first)
        second = pool.acquire(4)
        self.assertIs(second, first)
        pool.release(second)
        pool.release(bytearray(4))

        self.assertEqual(pool.stats.allocations, 1)
        self.assertEqual(pool.stats.reuses, 1)
        self.assertEqual(pool.stats.dropped, 1)
        self.assertLessEqual(pool.cached_bytes, 8)

    def test_lightweight_counters_skip_policy_diagnostics(self):
        policy = make_cache_policy("heat", max_hot_entries=1)
        cache = ExpertCache(max_bytes=16, policy=policy)
        cache.put_sized(0, 0, 4)

        with patch.object(
            policy,
            "snapshot",
            side_effect=AssertionError("policy diagnostics entered hot path"),
        ):
            self.assertEqual(cache.counters()["cache_entries"], 1)

    def test_detailed_cache_timings_are_opt_in(self):
        cache = ExpertCache(max_bytes=2, collect_timings=True)
        cache.put_sized(0, 0, 2)
        cache.lookup(0, 0)
        cache.put_sized(0, 1, 2)
        counters = cache.counters()

        self.assertGreaterEqual(counters["timing_cache_lookup_ms_total"], 0.0)
        self.assertGreaterEqual(counters["timing_victim_selection_ms_total"], 0.0)
        self.assertGreaterEqual(counters["timing_policy_maintenance_ms_total"], 0.0)

    def test_eviction_listener_receives_exact_entry(self):
        evicted = []
        cache = ExpertCache(max_bytes=2)
        cache.add_eviction_listener(evicted.append)
        first = cache.put_sized(0, 0, 2)
        cache.put_sized(0, 1, 2)

        self.assertEqual(evicted, [first])
        self.assertEqual(cache.entries, 1)

    def test_lease_keeps_backing_entry_stable_until_release(self):
        evicted = []
        cache = ExpertCache(max_bytes=2)
        cache.add_eviction_listener(evicted.append)
        first = cache.put_loaded(0, 0, {"weight": b"aa"})
        lease = cache.lease(first, references=(object(),))

        transient = cache.put_loaded(0, 1, {"weight": b"bb"})
        self.assertIs(cache.peek(0, 0), first)
        self.assertIsNone(cache.peek(0, 1))
        self.assertEqual(cache.pinned_entries, 1)
        self.assertEqual(cache.pinned_bytes, 2)
        self.assertEqual(evicted, [])
        self.assertEqual(transient.expert_id, 1)

        lease.release()
        cache.put_loaded(0, 1, {"weight": b"bb"})
        self.assertIsNone(cache.peek(0, 0))
        self.assertIsNotNone(cache.peek(0, 1))
        self.assertEqual(evicted, [first])

    def test_clear_rejects_active_lease(self):
        cache = ExpertCache(max_bytes=2)
        entry = cache.put_loaded(0, 0, {"weight": b"aa"})
        lease = cache.lease(entry)
        with self.assertRaisesRegex(RuntimeError, "leases are active"):
            cache.clear()
        lease.release()
        cache.clear()

    def test_native_batch_releases_provider_leases_on_success_and_failure(self):
        cache = ExpertCache(max_bytes=4)
        entries = {
            expert_id: cache.put_loaded(0, expert_id, {"weight": bytes([expert_id])})
            for expert_id in (0, 1)
        }
        results = {
            expert_id: {
                "gate_up_proj": {"native_int8": True, "expert_id": expert_id},
                "down_proj": None,
            }
            for expert_id in (0, 1)
        }

        class Provider:
            def __init__(self):
                self.cache = cache
                self._decoded = {
                    (0, expert_id): (entries[expert_id], results[expert_id])
                    for expert_id in (0, 1)
                }
                self.fail_on = None

            def prepare(self, _layer, _expert_ids):
                pass

            def get(self, _layer, expert_id):
                if expert_id == self.fail_on:
                    raise RuntimeError("injected provider failure")
                return results[expert_id]

        provider = Provider()
        with prepare_native_expert_batch(provider, 0, (0, 1)) as batch:
            self.assertEqual(cache.pinned_entries, 2)
            self.assertEqual([item["expert_id"] for item in batch.weights], [0, 1])
        self.assertEqual(cache.pinned_entries, 0)

        provider.fail_on = 1
        with self.assertRaisesRegex(RuntimeError, "injected provider failure"):
            prepare_native_expert_batch(provider, 0, (0, 1))
        self.assertEqual(cache.pinned_entries, 0)

    def test_no_cache_policy_rejects_admission_under_nonzero_budget(self):
        cache = ExpertCache(max_bytes=16, policy=make_cache_policy("none"))
        cache.begin_forward(0, "decode")
        cache.put_sized(0, 0, 4)

        self.assertEqual(cache.entries, 0)
        self.assertEqual(cache.cached_bytes, 0)
        self.assertEqual(cache.stats.admission_rejections, 1)

    def test_heat_policy_protects_hot_entry_and_uses_second_touch_prefill(self):
        policy = make_cache_policy("heat", max_hot_entries=1)
        cache = ExpertCache(max_bytes=2, policy=policy)
        cache.begin_forward(0, "prefill")
        cache.observe_routes(0, {0: 3, 1: 1})
        cache.put_sized(0, 0, 1)
        cache.put_sized(0, 1, 1)

        self.assertIsNotNone(cache.peek(0, 0))
        self.assertIsNone(cache.peek(0, 1))
        self.assertEqual(cache.stats.admission_rejections, 1)

        cache.observe_routes(0, {1: 1})
        cache.put_sized(0, 1, 1)
        cache.observe_routes(0, {2: 2})
        cache.put_sized(0, 2, 1)
        self.assertIsNotNone(cache.peek(0, 0))
        self.assertIsNone(cache.peek(0, 1))
        self.assertIsNotNone(cache.peek(0, 2))

    def test_heat_decay_eventually_demotes_stale_hot_entry(self):
        policy = make_cache_policy("heat", max_hot_entries=1)
        policy.begin_forward(0, "decode")
        policy.observe_routes(0, {3: 3}, "decode")
        self.assertEqual(policy.hot_keys(), ((0, 3),))
        for forward in range(1, 14):
            policy.begin_forward(forward, "decode")
        self.assertEqual(policy.hot_keys(), ())

    def test_per_layer_lru_and_stats(self):
        cache = ExpertCache(capacity_per_layer=1)

        def load(expert_id):
            return lambda: {"weight": bytes([expert_id])}

        cache.get_or_load(0, 0, load(0))
        cache.get_or_load(0, 0, load(0))
        cache.get_or_load(0, 1, load(1))
        cache.get_or_load(0, 0, load(0))
        cache.get_or_load(1, 0, load(0))

        self.assertEqual(cache.stats.requests, 5)
        self.assertEqual(cache.stats.hits, 1)
        self.assertEqual(cache.stats.misses, 4)
        self.assertEqual(cache.stats.evictions, 2)
        self.assertEqual(cache.stats.loaded_bytes, 4)
        self.assertEqual(cache.entries, 2)
        self.assertEqual(cache.cached_bytes, 2)

    def test_shard_reader_reuses_file_descriptors(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shard.bin"
            path.write_bytes(b"abcdefgh")
            with ShardReader() as reader:
                self.assertEqual(reader.read(path, 1, 3), b"bcd")
                self.assertEqual(reader.read(path, 2, 2), b"cd")
                self.assertEqual(reader.read_calls, 2)
                self.assertEqual(reader.read_bytes, 5)
                self.assertEqual(len(reader._fds), 1)

    def test_shard_reader_reads_into_owned_buffer(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shard.bin"
            path.write_bytes(b"abcdefgh")
            target = bytearray(3)
            with ShardReader() as reader:
                self.assertEqual(reader.readinto(path, 2, target), 3)
                self.assertEqual(target, b"cde")
                self.assertEqual(reader.read_calls, 1)
                self.assertEqual(reader.read_bytes, 3)

    def test_batch_read_coalesces_multiple_expert_slices(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5_moe",
                        "text_config": {
                            "model_type": "qwen3_5_moe_text",
                            "num_hidden_layers": 1,
                            "num_experts": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            write_shard(
                model / "model.safetensors",
                [
                    (
                        "model.language_model.layers.0.mlp.experts.gate_up_proj",
                        2 * 3,
                        [2, 3],
                    ),
                    (
                        "model.language_model.layers.0.mlp.experts.down_proj",
                        2 * 1,
                        [2, 1],
                    ),
                ],
            )
            locator = ExpertLocator(model)
            locations = [locator.locate(0, 0), locator.locate(0, 1)]
            with ShardReader() as reader:
                payloads = reader.read_locations(locations)
                stats = reader.last_batch_stats
            self.assertEqual(set(payloads), {(0, 0), (0, 1)})
            self.assertEqual(stats.logical_ranges, 4)
            self.assertEqual(stats.ranges, 1)
            self.assertEqual(stats.read_calls, 1)
            self.assertEqual(stats.useful_bytes, 8)
            self.assertEqual(stats.physical_bytes, 8)
            self.assertEqual(stats.wasted_bytes, 0)

    def test_global_byte_budget_evicts_across_layers(self):
        cache = ExpertCache(max_bytes=3)

        cache.get_or_load(0, 0, lambda: {"weight": b"aa"})
        cache.get_or_load(1, 0, lambda: {"weight": b"bb"})
        self.assertEqual(cache.cached_bytes, 2)
        cache.get_or_load(2, 0, lambda: {"weight": b"cc"})

        self.assertEqual(cache.cached_bytes, 2)
        self.assertIsNone(cache.lookup(0, 0))
        self.assertIsNotNone(cache.lookup(2, 0))

    def test_byte_budget_parser(self):
        self.assertEqual(parse_byte_budgets("512MiB,1GiB"), [512 * 1024**2, 1024**3])


class ExpertBenchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name)
        (self.model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5_moe",
                    "text_config": {
                        "model_type": "qwen3_5_moe_text",
                        "num_hidden_layers": 1,
                        "num_experts": 2,
                    },
                }
            ),
            encoding="utf-8",
        )
        write_shard(
            self.model / "model.safetensors",
            [
                (
                    "model.language_model.layers.0.mlp.experts.gate_up_proj",
                    2 * 3,
                    [2, 3],
                ),
                (
                    "model.language_model.layers.0.mlp.experts.down_proj",
                    2 * 1,
                    [2, 1],
                ),
            ],
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_generated_trace_and_capacity_sweep(self):
        trace = generate_trace([0], num_experts=2, tokens=3, top_k=1, mode="locality")
        result = run_expert_benchmark(self.model, [0, 1], trace)

        self.assertEqual(result["trace"]["requests"], 3)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["logical_bytes"], 12)
        self.assertGreaterEqual(result["results"][1]["hit_rate"], result["results"][0]["hit_rate"])

    def test_route_trace_json_replay_ignores_extra_metadata(self):
        trace_path = self.model / "route.json"
        trace_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "qwen3_5_moe_route_trace",
                    "requests": [
                        {"forward": 0, "row": 0, "layer": 0, "expert": 1},
                        {"forward": 0, "row": 0, "layer": 0, "expert": 0},
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(load_trace(trace_path), [(0, 1), (0, 0)])

    def test_grouped_trace_preserves_forward_and_batch_union(self):
        trace_path = self.model / "route-v2.json"
        trace_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "forwards": [
                        {
                            "forward": 0,
                            "phase": "prefill",
                            "rows": [
                                {
                                    "row": 0,
                                    "token_position": 0,
                                    "token_id": 10,
                                    "layers": [{"layer": 0, "expert_ids": [0, 1]}],
                                },
                                {
                                    "row": 1,
                                    "token_position": 1,
                                    "token_id": 11,
                                    "layers": [{"layer": 0, "expert_ids": [1, 0]}],
                                },
                            ],
                        },
                        {
                            "forward": 1,
                            "phase": "decode",
                            "rows": [
                                {
                                    "row": 0,
                                    "token_position": 2,
                                    "token_id": 12,
                                    "layers": [{"layer": 0, "expert_ids": [1]}],
                                }
                            ],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        trace = load_route_trace(trace_path)
        self.assertEqual(trace.raw_requests, 5)
        self.assertEqual(trace.phases, {"prefill": 4, "decode": 1})
        self.assertEqual(trace.batch_union_requests(), [(0, 0), (0, 1), (0, 1)])
        self.assertEqual([group.raw_requests for group in trace.replay_groups(True)], [4, 1])

        result = run_expert_benchmark(self.model, [0], trace, batch_union=True)
        self.assertEqual(result["trace"]["raw_requests"], 5)
        self.assertEqual(result["trace"]["effective_requests"], 3)
        self.assertEqual(result["trace"]["batch_union_deduped_requests"], 2)
        self.assertEqual(result["results"][0]["phase_metrics"]["prefill"]["requests"], 2)
        self.assertEqual(result["results"][0]["phase_metrics"]["decode"]["requests"], 1)

    def test_stage73_policy_replay_uses_real_cache_policy_and_prefetch_accounting(self):
        trace = RouteTrace(
            groups=(
                TraceGroup(0, "prefill", 0, 0, 10, ((0, 0), (0, 1))),
                TraceGroup(1, "decode", 0, 1, 11, ((0, 0), (0, 1))),
                TraceGroup(2, "decode", 0, 2, 12, ((0, 0), (0, 1))),
                TraceGroup(3, "decode", 0, 3, 13, ((0, 0), (0, 1))),
                TraceGroup(4, "decode", 0, 4, 14, ((0, 0),)),
            )
        )
        no_cache = run_policy_replay(self.model, trace, "S0", max_bytes=8)
        predictive = run_policy_replay(
            self.model,
            trace,
            "S4",
            max_bytes=4,
            prefetch_budget_ratio=1.0,
        )

        self.assertEqual(no_cache["cache"]["hits"], 0)
        self.assertGreater(predictive["prefetch"]["submitted"], 0)
        self.assertGreater(predictive["prefetch"]["hits"], 0)
        self.assertTrue(all(predictive["invariants"].values()))


# [Main Dev]
