"""Small model-independent helpers for multi-request routed-MoE fixtures.

The helper only combines already captured rows. It does not synthesize routes
and does not own cache, session, or scheduler state.

[Main Dev]
"""

from __future__ import annotations

from typing import Any, Sequence


def combine_layer_rows(
    records: Sequence[dict[str, Any]],
    layer: int,
    torch_module: Any,
):
    """Concatenate independent real rows in stable session order."""

    if not records:
        raise ValueError("at least one session record is required")
    hidden = []
    selected = []
    routing = []
    for record in records:
        layers = record.get("layers")
        if not isinstance(layers, dict) or layer not in layers:
            raise ValueError(f"record is missing layer {layer}")
        item = layers[layer]
        hidden.append(item["hidden_states"])
        selected.append(item["selected_experts"])
        routing.append(item["routing_weights"])
    return (
        torch_module.cat(hidden, dim=0).contiguous(),
        torch_module.cat(selected, dim=0).contiguous(),
        torch_module.cat(routing, dim=0).contiguous(),
    )


# [Main Dev]
