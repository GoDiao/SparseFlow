from __future__ import annotations

import gc
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import ExpertCache
from .cache_policy import POLICY_VARIANTS, make_cache_policy
from .expert_provider import (
    ExpertProvider,
    ResidentExpertProvider,
    StreamingExpertProvider,
)
from .loader import ShardReader
from .locator import ExpertLocator
from .memory_loader import (
    build_qwen36_meta_text_model,
    current_rss_bytes,
    materialize_qwen36_text_model,
    peak_rss_bytes,
)
from .moe_probe import run_routed_experts
from .telemetry import RuntimeTelemetry


RUNTIME_ID = "qwen36-text-memory-native-v1"
EXPERT_MODULE_ID = "SparseFlowQwenExperts-v1"
DISPATCH_ID = "qwen36-topk-index-add-v1"
KERNEL_ID = "bf16-linear-silu-linear-eager-v1"


try:
    from torch import nn as _torch_nn
except ImportError:  # pragma: no cover - text runtime requires torch at use time
    class _Module:
        def __init__(self, *args, **kwargs):
            super().__init__()

    _torch_nn = None
else:
    _Module = _torch_nn.Module


@dataclass
class RouteAudit:
    """Compact exact route trace used by the C3-R/C3-S correctness gate."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def record(self, layer: int, selected_experts) -> None:
        values = selected_experts.detach().to(device="cpu").contiguous()
        raw = values.numpy().tobytes()
        expert_ids = tuple(int(value) for value in values.unique(sorted=True).tolist())
        self.records.append(
            {
                "ordinal": len(self.records),
                "layer": layer,
                "shape": list(values.shape),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "unique_experts": len(expert_ids),
                "expert_ids": list(expert_ids),
            }
        )

    def routes_since(self, start: int) -> dict[int, tuple[int, ...]]:
        result: dict[int, tuple[int, ...]] = {}
        for record in self.records[start:]:
            result[int(record["layer"])] = tuple(int(value) for value in record["expert_ids"])
        return result


class SparseFlowQwenExperts(_Module):
    """Transformers-compatible routed-expert module backed by SparseFlow."""

    def __init__(
        self,
        layer: int,
        provider: ExpertProvider,
        route_audit: RouteAudit,
        telemetry: RuntimeTelemetry | None = None,
    ):
        super().__init__()
        self.layer = layer
        self.provider = provider
        self.route_audit = route_audit
        self.telemetry = telemetry or RuntimeTelemetry("none")

    def forward(self, hidden_states, top_k_index, top_k_weights):
        self.route_audit.record(self.layer, top_k_index)
        self.provider.observe_routes(self.layer, top_k_index)
        before = self.provider.snapshot()
        started = time.perf_counter()
        result = run_routed_experts(
            hidden_states,
            top_k_index,
            top_k_weights,
            lambda expert_id: self.provider.get(self.layer, expert_id),
            prepare_routed=lambda expert_ids: self.provider.prepare(self.layer, expert_ids),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.telemetry.record_layer(
            self.layer,
            top_k_index,
            before,
            self.provider.snapshot(),
            elapsed_ms,
        )
        return result


class Qwen36TextRuntime:
    """Python reference runtime for text-only Qwen3.6 inference.

    The surrounding Transformer implementation comes from the installed
    Transformers reference.  In ``streaming`` mode, every Qwen3.6 routed
    expert module is replaced with :class:`SparseFlowQwenExperts`, so the
    complete model forward uses SparseFlow's SSD/cache path for MoE experts.
    ``transformers`` load mode keeps the Stage 6 reference behavior.  The
    ``memory-native`` mode builds a text-only meta model, installs streaming
    experts first, and materializes only resident checkpoint tensors.
    """

    def __init__(
        self,
        model,
        tokenizer,
        torch,
        model_dir: str | Path,
        mode: str = "resident",
        reader: ShardReader | None = None,
        cache: ExpertCache | None = None,
        provider: ExpertProvider | None = None,
        route_audit: RouteAudit | None = None,
        telemetry: RuntimeTelemetry | None = None,
        prefetch_workers: int = 0,
        experts_implementation: str = "eager",
        load_mode: str = "transformers",
        loader_report: dict[str, Any] | None = None,
    ):
        if mode not in {"resident", "streaming"}:
            raise ValueError("mode must be resident or streaming")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.torch = torch
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.mode = mode
        self.reader = reader
        self.cache = cache
        self.provider = provider
        self.route_audit = route_audit
        self.telemetry = telemetry or RuntimeTelemetry("none")
        self.prefetch_workers = prefetch_workers
        self.experts_implementation = experts_implementation
        self.load_mode = load_mode
        self.loader_report = loader_report

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        mode: str = "resident",
        dtype: str = "bf16",
        device_map: dict[str, str] | str | None = None,
        cache_slots: int | None = 16,
        cache_bytes: int | None = None,
        prefetch_workers: int = 0,
        coalesce_gap: int = 0,
        cache_policy: str = "lru",
        prefetch_policy: str = "current-route",
        prefetch_budget_ratio: float = 0.25,
        hot_ratio: float = 0.25,
        telemetry_level: str = "summary",
        experts_implementation: str | None = None,
        load_mode: str = "transformers",
    ) -> "Qwen36TextRuntime":
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        root = Path(model_dir).expanduser().resolve()
        dtype_value = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }.get(dtype.lower())
        if dtype_value is None:
            raise ValueError("dtype must be bf16, fp16, or fp32")
        if load_mode not in {"transformers", "memory-native"}:
            raise ValueError("load_mode must be transformers or memory-native")
        if load_mode == "memory-native" and experts_implementation not in {None, "eager"}:
            raise ValueError("SparseFlow memory-native runtime requires eager expert arithmetic")
        if mode == "streaming" and experts_implementation not in {None, "eager"}:
            raise ValueError("SparseFlow streaming currently implements eager expert arithmetic")
        if load_mode == "memory-native" and mode == "resident" and prefetch_workers:
            raise ValueError("resident expert backend does not support prefetch workers")
        if not 0.0 <= hot_ratio <= 1.0:
            raise ValueError("hot_ratio must be in [0, 1]")
        if not 0.0 <= prefetch_budget_ratio <= 1.0:
            raise ValueError("prefetch_budget_ratio must be in [0, 1]")
        if device_map is None:
            device_map = {"": "cpu"}
        tokenizer = AutoTokenizer.from_pretrained(
            root,
            local_files_only=True,
            use_fast=True,
        )
        if load_mode == "memory-native":
            if device_map != {"": "cpu"} and device_map != "cpu":
                raise ValueError("memory-native loading currently requires CPU model placement")
            reader = ShardReader()
            cache = None
            provider: ExpertProvider | None = None
            try:
                if mode == "streaming":
                    if cache_slots is None and cache_bytes is None:
                        cache_slots = 16
                    policy = make_cache_policy(
                        cache_policy,
                        _max_hot_entries(root, cache_slots, cache_bytes, hot_ratio),
                    )
                    cache = ExpertCache(
                        capacity_per_layer=cache_slots,
                        max_bytes=cache_bytes,
                        policy=policy,
                    )
                    provider = StreamingExpertProvider(
                        root,
                        cache,
                        reader,
                        torch,
                        prefetch_workers=prefetch_workers,
                        coalesce_gap=coalesce_gap,
                        prefetch_policy=prefetch_policy,
                        prefetch_budget_ratio=prefetch_budget_ratio,
                    )
                else:
                    provider = ResidentExpertProvider(root, reader, torch)
                route_audit = RouteAudit()
                telemetry = RuntimeTelemetry(telemetry_level)
                build = build_qwen36_meta_text_model(
                    root,
                    expert_factory=lambda layer: SparseFlowQwenExperts(
                        layer,
                        provider,
                        route_audit,
                        telemetry,
                    ),
                )
                materialized = materialize_qwen36_text_model(build, dtype=dtype)
            except Exception:
                if provider is not None:
                    provider.close()
                reader.close()
                raise
            assert provider is not None
            loader_report = materialized.as_dict()
            loader_report["expert_reader_calls_after_init"] = reader.read_calls
            loader_report["expert_reader_bytes_after_init"] = reader.read_bytes
            loader_report["expert_provider"] = provider.storage_report()
            return cls(
                materialized.model,
                tokenizer,
                torch,
                root,
                mode=mode,
                reader=reader,
                cache=cache,
                provider=provider,
                route_audit=route_audit,
                telemetry=telemetry,
                prefetch_workers=prefetch_workers,
                experts_implementation="eager",
                load_mode="memory-native",
                loader_report=loader_report,
            )

        model = AutoModelForImageTextToText.from_pretrained(
            root,
            local_files_only=True,
            dtype=dtype_value,
            device_map=device_map,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.eval()
        requested_experts = "eager" if mode == "streaming" else experts_implementation
        if requested_experts is not None:
            setter = getattr(model, "set_experts_implementation", None)
            if setter is None:
                raise ValueError("loaded Transformers model cannot select an experts implementation")
            setter(requested_experts)
        active_experts = requested_experts or cls._current_experts_implementation(model)
        if mode == "resident":
            return cls(
                model,
                tokenizer,
                torch,
                root,
                mode="resident",
                experts_implementation=active_experts,
                load_mode="transformers",
                loader_report={"mode": "transformers-full"},
            )

        if device_map != {"": "cpu"} and device_map != "cpu":
            raise ValueError("SparseFlow streaming reference currently requires CPU model placement")
        if next(model.parameters()).device.type != "cpu":
            raise ValueError("SparseFlow streaming reference requires CPU model parameters")
        if cache_slots is None and cache_bytes is None:
            cache_slots = 16
        reader = ShardReader()
        policy = make_cache_policy(
            cache_policy,
            _max_hot_entries(root, cache_slots, cache_bytes, hot_ratio),
        )
        cache = ExpertCache(
            capacity_per_layer=cache_slots,
            max_bytes=cache_bytes,
            policy=policy,
        )
        provider = StreamingExpertProvider(
            root,
            cache,
            reader,
            torch,
            prefetch_workers=prefetch_workers,
            coalesce_gap=coalesce_gap,
            prefetch_policy=prefetch_policy,
            prefetch_budget_ratio=prefetch_budget_ratio,
        )
        route_audit = RouteAudit()
        telemetry = RuntimeTelemetry(telemetry_level)
        cls._attach_sparseflow_experts(model, provider, route_audit, telemetry)
        return cls(
            model,
            tokenizer,
            torch,
            root,
            mode="streaming",
            reader=reader,
            cache=cache,
            provider=provider,
            route_audit=route_audit,
            telemetry=telemetry,
            prefetch_workers=prefetch_workers,
            experts_implementation="eager",
            load_mode="transformers",
            loader_report={"mode": "transformers-full-then-replace"},
        )

    @staticmethod
    def _attach_sparseflow_experts(
        model,
        provider: ExpertProvider,
        route_audit: RouteAudit,
        telemetry: RuntimeTelemetry,
    ) -> None:
        language_model = model.model
        layers = (
            language_model.language_model.layers
            if hasattr(language_model, "language_model")
            else language_model.layers
        )
        for layer_index, decoder_layer in enumerate(layers):
            decoder_layer.mlp.experts = SparseFlowQwenExperts(
                layer_index,
                provider,
                route_audit,
                telemetry,
            )

    @staticmethod
    def _current_experts_implementation(model) -> str:
        try:
            experts = model.model.language_model.layers[0].mlp.experts
            return str(experts.config._experts_implementation)
        except (AttributeError, IndexError):
            return "unknown"

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def encode_chat(self, prompt: str) -> dict[str, Any]:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        encoded = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }

    def prefill(self, inputs: dict[str, Any]):
        """Run prompt prefill and return logits plus Transformers Cache."""

        with self.torch.inference_mode():
            return self.model(
                **inputs,
                use_cache=True,
            )

    def decode(self, token_ids, attention_mask, past_key_values):
        """Run one incremental token forward using the returned Cache."""

        with self.torch.inference_mode():
            return self.model(
                input_ids=token_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

    def greedy_generate(
        self,
        prompt: str,
        max_new_tokens: int = 8,
        stop_on_eos: bool = True,
        record_logit_fingerprints: bool = False,
    ) -> dict[str, Any]:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        inputs = self.encode_chat(prompt)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", self.torch.ones_like(input_ids))
        rss_before_prefill = current_rss_bytes()
        storage_before_prefill = self.provider.snapshot() if self.provider is not None else None
        route_start = len(self.route_audit.records) if self.route_audit is not None else 0
        self._begin_forward(0, "prefill", int(input_ids.shape[-1]) - 1)

        prefill_started = time.perf_counter()
        first = self.prefill(inputs)
        next_token = first.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        prefill_seconds = time.perf_counter() - prefill_started
        storage_after_prefill = self.provider.snapshot() if self.provider is not None else None
        generated = [next_token]
        logit_fingerprints = (
            [_logit_fingerprint(first.logits[:, -1, :], self.torch)]
            if record_logit_fingerprints
            else None
        )
        past = getattr(first, "past_key_values", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        decode_durations: list[float] = []
        decode_storage: list[dict[str, Any]] = []
        previous_decode_routes: dict[int, tuple[int, ...]] = {}

        for forward_index in range(1, max_new_tokens):
            attention_mask = self.torch.cat(
                [attention_mask, self.torch.ones_like(next_token)],
                dim=-1,
            )
            self._begin_forward(
                forward_index,
                "decode",
                int(input_ids.shape[-1]) + forward_index - 1,
            )
            if self.provider is not None:
                self.provider.predict(previous_decode_routes)
            decode_route_start = (
                len(self.route_audit.records) if self.route_audit is not None else 0
            )
            started = time.perf_counter()
            output = self.decode(next_token, attention_mask, past)
            decode_durations.append(time.perf_counter() - started)
            if self.provider is not None:
                decode_storage.append(self.provider.snapshot())
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            if logit_fingerprints is not None:
                logit_fingerprints.append(
                    _logit_fingerprint(output.logits[:, -1, :], self.torch)
                )
            past = getattr(output, "past_key_values", None)
            if self.route_audit is not None:
                previous_decode_routes = self.route_audit.routes_since(decode_route_start)
            if stop_on_eos and eos_id is not None and bool((next_token == eos_id).all()):
                break

        generated_ids = self.torch.cat(generated, dim=-1)
        text = self.tokenizer.decode(
            generated_ids[0].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        finalize_started = time.perf_counter()
        if self.provider is not None:
            self.provider.finish_generation()
        prefetch_finalize_seconds = time.perf_counter() - finalize_started
        storage_after_generation = self.provider.snapshot() if self.provider is not None else None
        route_records = (
            self.route_audit.records[route_start:] if self.route_audit is not None else None
        )
        return {
            "agent": "Main Dev",
            "mode": self.mode,
            "expert_backend": (
                self.provider.backend_id if self.provider is not None else "transformers-resident"
            ),
            "load_mode": self.load_mode,
            "experts_implementation": self.experts_implementation,
            "runtime_identity": {
                "runtime_id": RUNTIME_ID if self.load_mode == "memory-native" else "transformers-reference",
                "expert_module_id": EXPERT_MODULE_ID if self.provider is not None else "transformers-experts",
                "dispatch_id": DISPATCH_ID if self.provider is not None else "transformers-dispatch",
                "kernel_id": KERNEL_ID if self.provider is not None else self.experts_implementation,
            },
            "prompt": prompt,
            "input_ids": input_ids[0].tolist(),
            "generated_ids": generated_ids[0].tolist(),
            "generated_tokens": int(generated_ids.shape[-1]),
            "text": text,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": sum(decode_durations),
            "decode_token_seconds": decode_durations,
            "prefetch_finalize_seconds": prefetch_finalize_seconds,
            "logit_fingerprints": logit_fingerprints,
            "cache": self.cache.stats_dict() if self.cache is not None else None,
            "prefetch": self.provider.prefetch_stats() if self.provider is not None else None,
            "provider_storage": self.provider.storage_report() if self.provider is not None else None,
            "storage_phases": {
                "before_prefill": storage_before_prefill,
                "after_prefill": storage_after_prefill,
                "after_each_decode": decode_storage,
                "after_generation": storage_after_generation,
            },
            "route_audit": route_records,
            "telemetry": self.telemetry.as_dict(),
            "loader": self.loader_report,
            "memory": {
                "rss_before_prefill": rss_before_prefill,
                "rss_after_generation": current_rss_bytes(),
                "process_peak_rss": peak_rss_bytes(),
            },
        }

    def _begin_forward(
        self,
        forward: int,
        phase: str,
        token_position: int | None,
    ) -> None:
        if self.provider is not None:
            self.provider.begin_forward(forward, phase)
        self.telemetry.begin_forward(forward, phase, token_position)

    def close(self) -> None:
        if self.provider is not None:
            self.provider.close()
        if self.reader is not None:
            self.reader.close()

    def __enter__(self) -> "Qwen36TextRuntime":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def _max_hot_entries(
    model_dir: Path,
    cache_slots: int | None,
    cache_bytes: int | None,
    hot_ratio: float,
) -> int:
    if hot_ratio <= 0.0:
        return 0
    locator = ExpertLocator(model_dir)
    if cache_bytes is not None:
        first_layer = locator.layers[0]
        expert_bytes = locator.locate(first_layer, 0).nbytes
        total_entries = cache_bytes // expert_bytes
    elif cache_slots is not None:
        total_entries = cache_slots * len(locator.layers)
    else:
        total_entries = 0
    return max(0, int(total_entries * hot_ratio))


def compare_text_paths(
    model_dir: str | Path,
    prompt: str,
    max_new_tokens: int = 8,
    dtype: str = "bf16",
    cache_slots: int | None = 16,
    cache_bytes: int | None = None,
    prefetch_workers: int = 0,
    coalesce_gap: int = 0,
    streaming_load_mode: str = "transformers",
    cache_policy: str = "lru",
    prefetch_policy: str = "current-route",
    prefetch_budget_ratio: float = 0.25,
    hot_ratio: float = 0.25,
    telemetry_level: str = "summary",
) -> dict[str, Any]:
    """Run resident and streaming text generation as a reproducible gate.

    The two models are loaded sequentially.  The resident runtime is closed
    and released before the streaming checkpoint is loaded, which avoids
    requiring enough RAM for two complete reference models at once.
    """

    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    root = Path(model_dir).expanduser().resolve()
    resident, resident_load_seconds = _run_text_path(
        root,
        prompt,
        max_new_tokens,
        mode="resident",
        dtype=dtype,
        experts_implementation="eager",
        load_mode="transformers",
    )
    gc.collect()

    streaming, streaming_load_seconds = _run_text_path(
        root,
        prompt,
        max_new_tokens,
        mode="streaming",
        dtype=dtype,
        cache_slots=cache_slots,
        cache_bytes=cache_bytes,
        prefetch_workers=prefetch_workers,
        coalesce_gap=coalesce_gap,
        experts_implementation="eager",
        load_mode=streaming_load_mode,
        cache_policy=cache_policy,
        prefetch_policy=prefetch_policy,
        prefetch_budget_ratio=prefetch_budget_ratio,
        hot_ratio=hot_ratio,
        telemetry_level=telemetry_level,
    )
    comparison = compare_generation_results(resident, streaming)
    return {
        "schema_version": 1,
        "kind": "qwen3_5_text_runtime_correctness",
        "agent": "Main Dev",
        "model": str(root),
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "dtype": dtype,
        "experts_implementation": "eager",
        "streaming_load_mode": streaming_load_mode,
        "cache_policy": {
            "capacity_per_layer": cache_slots,
            "max_bytes": cache_bytes,
            "prefetch_workers": prefetch_workers,
            "coalesce_gap": coalesce_gap,
        },
        "resident": {
            **resident,
            "load_seconds": resident_load_seconds,
        },
        "streaming": {
            **streaming,
            "load_seconds": streaming_load_seconds,
        },
        "correctness": comparison,
        "boundary": (
            "text-only memory-native streaming; routed experts were never fully materialized"
            if streaming_load_mode == "memory-native"
            else (
                "text-only Python reference; checkpoint is loaded by Transformers "
                "before routed experts are replaced"
            )
        ),
    }


def compare_sparseflow_runtime_paths(
    model_dir: str | Path,
    prompt: str,
    max_new_tokens: int = 4,
    dtype: str = "bf16",
    cache_slots: int | None = 16,
    cache_bytes: int | None = None,
    prefetch_workers: int = 0,
    coalesce_gap: int = 0,
    cache_policy: str = "lru",
    prefetch_policy: str = "current-route",
    prefetch_budget_ratio: float = 0.25,
    hot_ratio: float = 0.25,
    telemetry_level: str = "summary",
) -> dict[str, Any]:
    """Compare C3-R and C3-S with one memory-native runtime and expert kernel."""

    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if dtype != "bf16":
        raise ValueError("Stage 7.2 C3-R/C3-S runtime-check currently requires bf16")

    root = Path(model_dir).expanduser().resolve()
    resident, resident_load_seconds = _run_text_path(
        root,
        prompt,
        max_new_tokens,
        mode="resident",
        dtype=dtype,
        experts_implementation="eager",
        load_mode="memory-native",
    )
    gc.collect()
    streaming, streaming_load_seconds = _run_text_path(
        root,
        prompt,
        max_new_tokens,
        mode="streaming",
        dtype=dtype,
        cache_slots=cache_slots,
        cache_bytes=cache_bytes,
        prefetch_workers=prefetch_workers,
        coalesce_gap=coalesce_gap,
        cache_policy=cache_policy,
        prefetch_policy=prefetch_policy,
        prefetch_budget_ratio=prefetch_budget_ratio,
        hot_ratio=hot_ratio,
        telemetry_level=telemetry_level,
        experts_implementation="eager",
        load_mode="memory-native",
    )

    correctness = compare_generation_results(resident, streaming)
    route_audit_equal = resident.get("route_audit") == streaming.get("route_audit")
    identity_equal = resident.get("runtime_identity") == streaming.get("runtime_identity")
    resident_storage = resident["provider_storage"]
    resident_phases = resident["storage_phases"]
    streaming_phases = streaming["storage_phases"]
    resident_generation_reads = _reader_delta(
        resident_phases["before_prefill"],
        resident_phases["after_generation"],
    )
    streaming_generation_reads = _reader_delta(
        streaming_phases["before_prefill"],
        streaming_phases["after_generation"],
    )

    # Header-only expected footprint; no provider payload is materialized here.
    expert_locator = ExpertLocator(root)
    expected_layers = len(expert_locator.layers)
    expected_experts = expected_layers * expert_locator.num_experts
    expected_bytes = sum(
        span.nbytes
        for layer in expert_locator.layers
        for span in expert_locator.fused_parts(layer).values()
    )
    invariants = {
        "same_runtime_identity": identity_equal,
        "same_route_audit": route_audit_equal,
        "full_generation_exact": correctness["all_equal"],
        "resident_all_experts_loaded": (
            resident_storage["resident_layers"] == expected_layers
            and resident_storage["resident_experts"] == expected_experts
        ),
        "resident_payload_exact": resident_storage["resident_bytes"] == expected_bytes,
        "resident_preload_payload_exact": resident_storage["preload_read_bytes"] == expected_bytes,
        "resident_generation_zero_expert_io": (
            resident_generation_reads["read_calls"] == 0
            and resident_generation_reads["read_bytes"] == 0
        ),
        "streaming_init_zero_expert_io": (
            streaming["loader"]["expert_reader_calls_after_init"] == 0
            and streaming["loader"]["expert_reader_bytes_after_init"] == 0
        ),
        "streaming_generation_reads_experts": streaming_generation_reads["read_bytes"] > 0,
    }
    return {
        "schema_version": 2,
        "kind": "qwen3_5_stage7_2_c3_runtime_correctness",
        "stage": "7.2",
        "agent": "Main Dev",
        "model": str(root),
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "dtype": dtype,
        "runtime_identity": resident["runtime_identity"],
        "cache_policy": {
            "capacity_per_layer": cache_slots,
            "max_bytes": cache_bytes,
            "prefetch_workers": prefetch_workers,
            "coalesce_gap": coalesce_gap,
            "cache_policy": cache_policy,
            "prefetch_policy": prefetch_policy,
            "prefetch_budget_ratio": prefetch_budget_ratio,
            "hot_ratio": hot_ratio,
        },
        "expected": {
            "layers": expected_layers,
            "experts_per_layer": expert_locator.num_experts,
            "logical_experts": expected_experts,
            "routed_expert_bytes": expected_bytes,
        },
        "resident": {
            **resident,
            "load_seconds": resident_load_seconds,
            "generation_expert_io": resident_generation_reads,
        },
        "streaming": {
            **streaming,
            "load_seconds": streaming_load_seconds,
            "generation_expert_io": streaming_generation_reads,
        },
        "correctness": {
            **correctness,
            "route_audit_equal": route_audit_equal,
            "runtime_identity_equal": identity_equal,
        },
        "invariants": invariants,
        "all_invariants_pass": all(invariants.values()),
        "boundary": (
            "C3-R and C3-S use the same text-only memory-native model, "
            "SparseFlow expert module, dispatch order, eager BF16 kernel, KV cache, "
            "and greedy generation loop; only the expert provider differs."
        ),
    }


def compare_sparseflow_policy_paths(
    model_dir: str | Path,
    prompt: str,
    max_new_tokens: int = 8,
    cache_bytes: int = 4 * 1024**3,
    variants: tuple[str, ...] = tuple(POLICY_VARIANTS),
    prefetch_workers: int = 2,
    prefetch_budget_ratio: float = 0.10,
    hot_ratio: float = 0.25,
    telemetry_level: str = "summary",
) -> dict[str, Any]:
    """Validate all Stage 7.3 policies against one memory-native C3-R run."""

    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if max_new_tokens < 2:
        raise ValueError("Stage 7.3 policy-check requires at least two generated tokens")
    if cache_bytes <= 0:
        raise ValueError("Stage 7.3 policy-check requires a positive cache byte budget")
    unknown = sorted(set(variants) - set(POLICY_VARIANTS))
    if unknown:
        raise ValueError(f"unknown Stage 7.3 policy variants: {unknown}")

    root = Path(model_dir).expanduser().resolve()
    resident, resident_load_seconds = _run_text_path(
        root,
        prompt,
        max_new_tokens,
        mode="resident",
        dtype="bf16",
        experts_implementation="eager",
        load_mode="memory-native",
        telemetry_level=telemetry_level,
    )
    gc.collect()
    results = []
    for variant in variants:
        config = POLICY_VARIANTS[variant]
        workers = prefetch_workers if config["prefetch_policy"] != "none" else 0
        streaming, load_seconds = _run_text_path(
            root,
            prompt,
            max_new_tokens,
            mode="streaming",
            dtype="bf16",
            cache_slots=None,
            cache_bytes=cache_bytes,
            prefetch_workers=workers,
            cache_policy=config["cache_policy"],
            prefetch_policy=config["prefetch_policy"],
            prefetch_budget_ratio=prefetch_budget_ratio,
            hot_ratio=hot_ratio,
            telemetry_level=telemetry_level,
            experts_implementation="eager",
            load_mode="memory-native",
        )
        correctness = compare_generation_results(resident, streaming)
        correctness["route_audit_equal"] = resident["route_audit"] == streaming["route_audit"]
        correctness["runtime_identity_equal"] = (
            resident["runtime_identity"] == streaming["runtime_identity"]
        )
        provider = streaming["provider_storage"]
        demand_accounted = provider["demand_requests"] == (
            provider["demand_reuse_hits"]
            + provider["demand_prefetch_served"]
            + provider["demand_misses"]
        )
        prefetch = streaming["prefetch"]
        invariants = {
            "full_generation_exact": correctness["all_equal"],
            "route_audit_exact": correctness["route_audit_equal"],
            "runtime_identity_exact": correctness["runtime_identity_equal"],
            "streaming_init_zero_expert_io": (
                streaming["loader"]["expert_reader_calls_after_init"] == 0
                and streaming["loader"]["expert_reader_bytes_after_init"] == 0
            ),
            "cache_budget_respected": provider["cached_bytes"] <= cache_bytes,
            "demand_accounting_exact": demand_accounted,
            "prefetch_failures_zero": prefetch["failed"] == 0,
        }
        results.append(
            {
                "variant": variant,
                **config,
                "prefetch_workers": workers,
                "load_seconds": load_seconds,
                "streaming": streaming,
                "correctness": correctness,
                "invariants": invariants,
                "all_invariants_pass": all(invariants.values()),
            }
        )
        gc.collect()

    return {
        "schema_version": 1,
        "kind": "qwen3_5_stage7_3_policy_correctness",
        "stage": "7.3",
        "agent": "Main Dev",
        "model": str(root),
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "cache_bytes": cache_bytes,
        "hot_ratio": hot_ratio,
        "prefetch_budget_ratio": prefetch_budget_ratio,
        "runtime_identity": resident["runtime_identity"],
        "resident": {**resident, "load_seconds": resident_load_seconds},
        "variants": results,
        "all_invariants_pass": all(item["all_invariants_pass"] for item in results),
        "boundary": (
            "All variants use fresh streaming runtimes and compare against one C3-R run; "
            "only cache/admission/prefetch policy changes."
        ),
    }


def compare_generation_results(
    resident: dict[str, Any],
    streaming: dict[str, Any],
) -> dict[str, bool]:
    """Compare the externally observable generation state of two runtimes."""

    input_ids_equal = resident.get("input_ids") == streaming.get("input_ids")
    generated_ids_equal = resident.get("generated_ids") == streaming.get("generated_ids")
    text_equal = resident.get("text") == streaming.get("text")
    generated_tokens_equal = (
        resident.get("generated_tokens") == streaming.get("generated_tokens")
    )
    resident_fingerprints = resident.get("logit_fingerprints")
    streaming_fingerprints = streaming.get("logit_fingerprints")
    logit_fingerprints_equal = (
        resident_fingerprints is not None
        and streaming_fingerprints is not None
        and resident_fingerprints == streaming_fingerprints
    )
    return {
        "input_ids_equal": input_ids_equal,
        "generated_ids_equal": generated_ids_equal,
        "generated_tokens_equal": generated_tokens_equal,
        "text_equal": text_equal,
        "logit_fingerprints_equal": logit_fingerprints_equal,
        "all_equal": (
            input_ids_equal
            and generated_ids_equal
            and generated_tokens_equal
            and text_equal
            and logit_fingerprints_equal
        ),
    }


def _run_text_path(
    model_dir: Path,
    prompt: str,
    max_new_tokens: int,
    mode: str,
    dtype: str,
    cache_slots: int | None = None,
    cache_bytes: int | None = None,
    prefetch_workers: int = 0,
    coalesce_gap: int = 0,
    experts_implementation: str | None = None,
    load_mode: str = "transformers",
    cache_policy: str = "lru",
    prefetch_policy: str = "current-route",
    prefetch_budget_ratio: float = 0.25,
    hot_ratio: float = 0.25,
    telemetry_level: str = "summary",
) -> tuple[dict[str, Any], float]:
    load_started = time.perf_counter()
    runtime = Qwen36TextRuntime.from_pretrained(
        model_dir,
        mode=mode,
        dtype=dtype,
        cache_slots=cache_slots,
        cache_bytes=cache_bytes,
        prefetch_workers=prefetch_workers,
        coalesce_gap=coalesce_gap,
        experts_implementation=experts_implementation,
        load_mode=load_mode,
        cache_policy=cache_policy,
        prefetch_policy=prefetch_policy,
        prefetch_budget_ratio=prefetch_budget_ratio,
        hot_ratio=hot_ratio,
        telemetry_level=telemetry_level,
    )
    load_seconds = time.perf_counter() - load_started
    try:
        result = runtime.greedy_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            record_logit_fingerprints=True,
        )
    finally:
        runtime.close()
    return result, load_seconds


def _reader_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    return {
        "read_calls": int(after["reader_calls"]) - int(before["reader_calls"]),
        "read_bytes": int(after["reader_bytes"]) - int(before["reader_bytes"]),
    }


def _logit_fingerprint(logits, torch) -> dict[str, Any]:
    """Return an exact digest for one batch of next-token logits."""

    values = logits.detach().to(device="cpu").contiguous()
    raw = values.view(torch.uint8).numpy().tobytes()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dtype": str(values.dtype).replace("torch.", ""),
        "shape": list(values.shape),
        "argmax": values.argmax(dim=-1).tolist(),
    }
