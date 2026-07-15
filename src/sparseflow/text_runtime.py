from __future__ import annotations

import gc
import hashlib
import time
from pathlib import Path
from typing import Any

from .cache import ExpertCache
from .loader import ShardReader
from .memory_loader import (
    build_qwen36_meta_text_model,
    materialize_qwen36_text_model,
)
from .moe_probe import CachedExpertProvider, run_routed_experts


try:
    from torch import nn as _torch_nn
except ImportError:  # pragma: no cover - text runtime requires torch at use time
    class _Module:
        def __init__(self, *args, **kwargs):
            super().__init__()

    _torch_nn = None
else:
    _Module = _torch_nn.Module


class SparseFlowQwenExperts(_Module):
    """Transformers-compatible routed-expert module backed by SparseFlow."""

    def __init__(self, layer: int, provider: CachedExpertProvider, prefetch: bool):
        super().__init__()
        self.layer = layer
        self.provider = provider
        self.prefetch_enabled = prefetch

    def forward(self, hidden_states, top_k_index, top_k_weights):
        prepare = (
            lambda expert_ids: self.provider.prefetch(self.layer, expert_ids)
            if self.prefetch_enabled
            else None
        )
        return run_routed_experts(
            hidden_states,
            top_k_index,
            top_k_weights,
            lambda expert_id: self.provider.get(self.layer, expert_id),
            prepare_routed=prepare,
        )


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
        provider: CachedExpertProvider | None = None,
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
        if load_mode == "memory-native" and mode != "streaming":
            raise ValueError("memory-native loading currently requires streaming mode")
        if mode == "streaming" and experts_implementation not in {None, "eager"}:
            raise ValueError("SparseFlow streaming currently implements eager expert arithmetic")
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
            if cache_slots is None and cache_bytes is None:
                cache_slots = 16
            reader = ShardReader()
            cache = ExpertCache(capacity_per_layer=cache_slots, max_bytes=cache_bytes)
            provider = CachedExpertProvider(
                root,
                cache,
                reader,
                torch,
                prefetch_workers=prefetch_workers,
                coalesce_gap=coalesce_gap,
            )
            try:
                build = build_qwen36_meta_text_model(
                    root,
                    expert_factory=lambda layer: SparseFlowQwenExperts(
                        layer,
                        provider,
                        prefetch_workers > 0,
                    ),
                )
                materialized = materialize_qwen36_text_model(build, dtype=dtype)
            except Exception:
                provider.close()
                reader.close()
                raise
            return cls(
                materialized.model,
                tokenizer,
                torch,
                root,
                mode="streaming",
                reader=reader,
                cache=cache,
                provider=provider,
                prefetch_workers=prefetch_workers,
                experts_implementation="eager",
                load_mode="memory-native",
                loader_report=materialized.as_dict(),
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
        cache = ExpertCache(capacity_per_layer=cache_slots, max_bytes=cache_bytes)
        provider = CachedExpertProvider(
            root,
            cache,
            reader,
            torch,
            prefetch_workers=prefetch_workers,
            coalesce_gap=coalesce_gap,
        )
        cls._attach_streaming_experts(model, provider, prefetch_workers > 0)
        return cls(
            model,
            tokenizer,
            torch,
            root,
            mode="streaming",
            reader=reader,
            cache=cache,
            provider=provider,
            prefetch_workers=prefetch_workers,
            experts_implementation="eager",
            load_mode="transformers",
            loader_report={"mode": "transformers-full-then-replace"},
        )

    @staticmethod
    def _attach_streaming_experts(model, provider: CachedExpertProvider, prefetch: bool) -> None:
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
                prefetch,
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

        prefill_started = time.perf_counter()
        first = self.prefill(inputs)
        next_token = first.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        prefill_seconds = time.perf_counter() - prefill_started
        generated = [next_token]
        logit_fingerprints = (
            [_logit_fingerprint(first.logits[:, -1, :], self.torch)]
            if record_logit_fingerprints
            else None
        )
        past = getattr(first, "past_key_values", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        decode_durations: list[float] = []

        for _ in range(1, max_new_tokens):
            attention_mask = self.torch.cat(
                [attention_mask, self.torch.ones_like(next_token)],
                dim=-1,
            )
            started = time.perf_counter()
            output = self.decode(next_token, attention_mask, past)
            decode_durations.append(time.perf_counter() - started)
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            if logit_fingerprints is not None:
                logit_fingerprints.append(
                    _logit_fingerprint(output.logits[:, -1, :], self.torch)
                )
            past = getattr(output, "past_key_values", None)
            if stop_on_eos and eos_id is not None and bool((next_token == eos_id).all()):
                break

        generated_ids = self.torch.cat(generated, dim=-1)
        text = self.tokenizer.decode(
            generated_ids[0].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return {
            "mode": self.mode,
            "load_mode": self.load_mode,
            "experts_implementation": self.experts_implementation,
            "prompt": prompt,
            "input_ids": input_ids[0].tolist(),
            "generated_ids": generated_ids[0].tolist(),
            "generated_tokens": int(generated_ids.shape[-1]),
            "text": text,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": sum(decode_durations),
            "decode_token_seconds": decode_durations,
            "logit_fingerprints": logit_fingerprints,
            "cache": self.cache.stats_dict() if self.cache is not None else None,
            "prefetch": self.provider.prefetch_stats() if self.provider is not None else None,
            "loader": self.loader_report,
        }

    def close(self) -> None:
        if self.provider is not None:
            self.provider.close()
        if self.reader is not None:
            self.reader.close()

    def __enter__(self) -> "Qwen36TextRuntime":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


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
