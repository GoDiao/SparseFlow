from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Iterable

from .cache import ExpertCache
from .loader import ShardReader
from .moe_probe import (
    CachedExpertProvider,
    _cache_stats_delta,
    _cache_stats_snapshot,
    _compare_tensor,
    _qwen36_moe_spans,
    _read_tensor,
    _route_hidden_states,
    _validate_moe_spans,
    run_moe_kernel,
)
from .locator import ExpertLocator


class Qwen36MoeOnlyRuntime:
    """A multi-layer runtime for the Qwen3.6 MoE blocks only.

    This deliberately excludes attention, Gated DeltaNet, residuals, KV
    cache, embeddings, and the language-model head.  It exists to validate
    that layer-wise expert storage, cache, coalescing, and prefetch compose
    correctly before the full Transformer runtime is attempted.
    """

    def __init__(
        self,
        model_dir: str | Path,
        layers: Iterable[int],
        reader: ShardReader,
        torch,
        mode: str = "streaming",
        cache: ExpertCache | None = None,
        prefetch_workers: int = 0,
        coalesce_gap: int = 0,
    ):
        if mode not in {"resident", "streaming"}:
            raise ValueError("mode must be resident or streaming")
        layers = tuple(dict.fromkeys(int(layer) for layer in layers))
        if not layers:
            raise ValueError("at least one layer is required")
        if any(layer < 0 for layer in layers):
            raise ValueError("layers must be non-negative")
        if mode == "streaming" and cache is None:
            raise ValueError("streaming runtime requires an ExpertCache")
        if mode == "resident" and prefetch_workers:
            raise ValueError("prefetch is only valid for streaming runtime")

        self.model_dir = Path(model_dir).expanduser().resolve()
        self.locator = ExpertLocator(self.model_dir)
        self.layers = layers
        self.reader = reader
        self.torch = torch
        self.mode = mode
        self.cache = cache
        self.prefetch_workers = prefetch_workers
        self.coalesce_gap = coalesce_gap
        self._common: dict[int, dict[str, Any]] = {}
        self._resident_routed: dict[int, dict[str, Any]] = {}
        self.provider = (
            CachedExpertProvider(
                self.model_dir,
                cache,
                reader,
                torch,
                prefetch_workers=prefetch_workers,
                coalesce_gap=coalesce_gap,
            )
            if mode == "streaming"
            else None
        )

    def run(self, hidden_states) -> dict[str, Any]:
        current = hidden_states
        details: list[dict[str, Any]] = []
        with self.torch.inference_mode():
            for layer in self.layers:
                spans = _qwen36_moe_spans(self.locator, layer)
                _validate_moe_spans(spans)
                common_before = (self.reader.read_calls, self.reader.read_bytes)
                common = self._common.get(layer)
                if common is None:
                    common = {
                        name: _read_tensor(span, self.reader, self.torch)
                        for name, span in spans.items()
                        if name not in {"gate_up_proj", "down_proj"}
                    }
                    self._common[layer] = common
                common_after = (self.reader.read_calls, self.reader.read_bytes)

                top_k = int(self.locator.config.text_config.get("num_experts_per_tok", 0) or 0)
                if top_k <= 0 or top_k > self.locator.num_experts:
                    raise ValueError(f"invalid num_experts_per_tok: {top_k}")
                route_weights, selected, router_logits = _route_hidden_states(
                    current,
                    common["router"],
                    top_k,
                )

                layer_cache_before = (
                    _cache_stats_snapshot(self.cache) if self.cache is not None else None
                )
                routed_before = (self.reader.read_calls, self.reader.read_bytes)
                if self.mode == "resident":
                    routed = self._resident_routed.get(layer)
                    if routed is None:
                        routed = {
                            "gate_up_proj": _read_tensor(
                                spans["gate_up_proj"], self.reader, self.torch
                            ),
                            "down_proj": _read_tensor(
                                spans["down_proj"], self.reader, self.torch
                            ),
                        }
                        self._resident_routed[layer] = routed
                    result = run_moe_kernel(
                        current,
                        common,
                        selected,
                        route_weights,
                        lambda expert_id: {
                            "gate_up_proj": routed["gate_up_proj"][expert_id],
                            "down_proj": routed["down_proj"][expert_id],
                        },
                    )
                else:
                    assert self.provider is not None
                    result = run_moe_kernel(
                        current,
                        common,
                        selected,
                        route_weights,
                        lambda expert_id: self.provider.get(layer, expert_id),
                        (lambda expert_ids: self.provider.prefetch(layer, expert_ids))
                        if self.prefetch_workers > 0
                        else None,
                    )
                routed_after = (self.reader.read_calls, self.reader.read_bytes)
                layer_cache_after = (
                    _cache_stats_snapshot(self.cache) if self.cache is not None else None
                )
                current = result["final_output"]
                details.append(
                    {
                        "layer": layer,
                        "selected_experts": selected,
                        "routing_weights": route_weights,
                        "router_logits": router_logits,
                        "result": result,
                        "storage": {
                            "common_read_calls": common_after[0] - common_before[0],
                            "common_read_bytes": common_after[1] - common_before[1],
                            "routed_read_calls": routed_after[0] - routed_before[0],
                            "routed_read_bytes": routed_after[1] - routed_before[1],
                        },
                        "cache": (
                            _cache_stats_delta(layer_cache_before, layer_cache_after)
                            if layer_cache_before is not None and layer_cache_after is not None
                            else None
                        ),
                    }
                )
        return {"hidden_states": current, "layers": details}

    def close(self) -> None:
        if self.provider is not None:
            self.provider.close()


def compare_multilayer_moe_paths(
    model_dir: str | Path,
    layers: Iterable[int] = (0, 1),
    rows: int = 2,
    seed: int = 1234,
    cache_slots: int | None = 16,
    cache_bytes: int | None = None,
    prefetch_workers: int = 0,
    coalesce_gap: int = 0,
) -> dict[str, Any]:
    """Compare resident and streaming execution across several MoE layers."""

    if rows < 1:
        raise ValueError("rows must be positive")
    layers = tuple(dict.fromkeys(int(layer) for layer in layers))
    if not layers:
        raise ValueError("at least one layer is required")

    import torch

    root = Path(model_dir).expanduser().resolve()
    locator = ExpertLocator(root)
    invalid = [layer for layer in layers if layer >= int(locator.config.text_config.get("num_hidden_layers", 0) or 0)]
    if invalid:
        raise ValueError(f"layers outside model range: {invalid}")
    if cache_slots is None and cache_bytes is None:
        cache_slots = locator.num_experts

    first_spans = _qwen36_moe_spans(locator, layers[0])
    hidden_size = int(first_spans["gate_up_proj"].shape[-1])
    dtype = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[first_spans["gate_up_proj"].dtype]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randn(rows, hidden_size, generator=generator, dtype=dtype)

    with ShardReader() as resident_reader, ShardReader() as streaming_reader:
        resident_runtime = Qwen36MoeOnlyRuntime(
            root,
            layers,
            resident_reader,
            torch,
            mode="resident",
        )
        streaming_cache = ExpertCache(
            capacity_per_layer=cache_slots,
            max_bytes=cache_bytes,
        )
        streaming_runtime = Qwen36MoeOnlyRuntime(
            root,
            layers,
            streaming_reader,
            torch,
            mode="streaming",
            cache=streaming_cache,
            prefetch_workers=prefetch_workers,
            coalesce_gap=coalesce_gap,
        )
        resident_started = time.perf_counter()
        resident_result = resident_runtime.run(initial)
        resident_wall_ms = (time.perf_counter() - resident_started) * 1000.0
        streaming_started = time.perf_counter()
        streaming_result = streaming_runtime.run(initial.clone())
        streaming_wall_ms = (time.perf_counter() - streaming_started) * 1000.0
        streaming_runtime.close()
        resident_runtime.close()

        per_layer = []
        names = (
            "selected_experts",
            "routing_weights",
            "router_logits",
            "routed_output",
            "shared_output",
            "final_output",
        )
        for resident_layer, streaming_layer in zip(
            resident_result["layers"], streaming_result["layers"]
        ):
            comparisons = {
                "selected_experts": _compare_tensor(
                    resident_layer["selected_experts"],
                    streaming_layer["selected_experts"],
                    torch,
                ),
                "routing_weights": _compare_tensor(
                    resident_layer["routing_weights"],
                    streaming_layer["routing_weights"],
                    torch,
                ),
                "router_logits": _compare_tensor(
                    resident_layer["router_logits"],
                    streaming_layer["router_logits"],
                    torch,
                ),
                "routed_output": _compare_tensor(
                    resident_layer["result"]["routed_output"],
                    streaming_layer["result"]["routed_output"],
                    torch,
                ),
                "shared_output": _compare_tensor(
                    resident_layer["result"]["shared_output"],
                    streaming_layer["result"]["shared_output"],
                    torch,
                ),
                "final_output": _compare_tensor(
                    resident_layer["result"]["final_output"],
                    streaming_layer["result"]["final_output"],
                    torch,
                ),
            }
            per_layer.append(
                {
                    "layer": resident_layer["layer"],
                    "unique_experts": len(streaming_layer["selected_experts"].unique().tolist()),
                    "resident_storage": resident_layer["storage"],
                    "streaming_storage": streaming_layer["storage"],
                    "cache": streaming_layer["cache"],
                    "comparison": comparisons,
                }
            )

        all_exact = all(
            layer["comparison"][name]["exact_equal"]
            for layer in per_layer
            for name in names
        )
        stream_read_calls = streaming_reader.read_calls
        stream_read_bytes = streaming_reader.read_bytes
        resident_read_calls = resident_reader.read_calls
        resident_read_bytes = resident_reader.read_bytes
        stream_common_calls = sum(
            item["streaming_storage"]["common_read_calls"] for item in per_layer
        )
        stream_common_bytes = sum(
            item["streaming_storage"]["common_read_bytes"] for item in per_layer
        )
        stream_routed_calls = sum(
            item["streaming_storage"]["routed_read_calls"] for item in per_layer
        )
        stream_routed_bytes = sum(
            item["streaming_storage"]["routed_read_bytes"] for item in per_layer
        )
        cache_result = streaming_cache.stats_dict()
        prefetch = streaming_runtime.provider.prefetch_stats() if streaming_runtime.provider else {}
        useful_stream_bytes = (
            prefetch.get("useful_bytes", stream_routed_bytes)
            if prefetch_workers > 0
            else stream_routed_bytes
        )

    return {
        "schema_version": 1,
        "kind": "qwen3_5_moe_multilayer_correctness",
        "agent": "Main Dev",
        "model": str(root),
        "layers": list(layers),
        "rows": rows,
        "seed": seed,
        "hidden_size": hidden_size,
        "dtype": str(dtype).replace("torch.", ""),
        "mode": "MoE-only; no attention/residual/KV/generation",
        "cache_policy": {
            "capacity_per_layer": cache_slots,
            "max_bytes": cache_bytes,
            "prefetch_workers": prefetch_workers,
            "coalesce_gap": coalesce_gap,
        },
        "resident_storage": {
            "read_calls": resident_read_calls,
            "read_bytes": resident_read_bytes,
        },
        "streaming_storage": {
            "read_calls": stream_read_calls,
            "read_bytes": stream_read_bytes,
            "common_read_calls": stream_common_calls,
            "common_read_bytes": stream_common_bytes,
            "routed_read_calls": stream_routed_calls,
            "routed_read_bytes": stream_routed_bytes,
            "loaded_bytes": cache_result["loaded_bytes"],
        },
        "timing": {
            "resident_wall_ms": resident_wall_ms,
            "streaming_wall_ms": streaming_wall_ms,
            "page_cache_state": "not controlled; interpret as a functional timing sample",
        },
        "cache": cache_result,
        "prefetch": prefetch,
        "correctness": {
            "all_exact_equal": all_exact,
            "max_abs_error": max(
                (item["comparison"][name].get("max_abs_error", 0.0)
                 for item in per_layer for name in names),
                default=0.0,
            ),
            "max_rel_error": max(
                (item["comparison"][name].get("max_rel_error", 0.0)
                 for item in per_layer for name in names),
                default=0.0,
            ),
        },
        "invariants": {
            "cache_request_partition": cache_result["requests"]
            == cache_result["hits"] + cache_result["misses"],
            "loaded_bytes_match_reader": cache_result["loaded_bytes"] == useful_stream_bytes,
            "physical_read_at_least_useful": stream_routed_bytes >= useful_stream_bytes,
            "cached_bytes_within_budget": (
                cache_bytes is None or cache_result["cached_bytes"] <= cache_bytes
            ),
        },
        "per_layer": per_layer,
    }
