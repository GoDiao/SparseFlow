import json
import tempfile
import unittest
from pathlib import Path

from sparseflow.multirequest_trace import (
    MultiRequestTrace,
    SessionRoute,
    analyze_schedule,
    load_multi_request_trace,
    make_schedule,
)
from sparseflow.trace import TraceGroup


def make_session(session_id: str, first: tuple[int, ...], second: tuple[int, ...]) -> SessionRoute:
    groups = (
        TraceGroup(
            forward=0,
            phase="prefill",
            row=0,
            token_position=2,
            token_id=1,
            requests=tuple((layer, expert) for layer in range(40) for expert in first),
        ),
        TraceGroup(
            forward=1,
            phase="decode",
            row=0,
            token_position=3,
            token_id=2,
            requests=tuple((layer, expert) for layer in range(40) for expert in second),
        ),
        TraceGroup(
            forward=2,
            phase="decode",
            row=0,
            token_position=4,
            token_id=3,
            requests=tuple((layer, expert) for layer in range(40) for expert in second),
        ),
    )
    return SessionRoute(session_id, session_id, session_id, groups, f"sha-{session_id}")


class MultiRequestTraceTest(unittest.TestCase):
    def test_schedule_union_preserves_session_boundaries(self):
        left = make_session("left", (0, 1), (1, 2))
        right = make_session("right", (2, 3), (2, 3))
        schedule = make_schedule((left, right), "sync", max_steps=2)
        trace = MultiRequestTrace((left, right), (schedule,))
        result = analyze_schedule(trace, "sync")

        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["assignments"], 40 * 2 * 2 * 2)
        self.assertEqual(result["unique_experts"], 40 * 3 * 2)
        self.assertAlmostEqual(result["union_ratio"], 320 / 240)
        self.assertEqual(result["steps_detail"][0]["layers"][0]["unique_experts"], 3)

    def test_offset_schedule_omits_not_yet_arrived_session(self):
        left = make_session("left", (0,), (1,))
        right = make_session("right", (0,), (1,))
        schedule = make_schedule(
            (left, right),
            "offset",
            kind="offset",
            max_steps=2,
            arrival_offsets={"right": 1},
        )
        self.assertEqual([step.sessions for step in schedule.steps], [
            (("left", 1),),
            (("left", 2), ("right", 1)),
        ])

    def test_round_trip_is_deterministic(self):
        left = make_session("left", (0,), (1,))
        schedule = make_schedule((left,), "sync", max_steps=2)
        trace = MultiRequestTrace((left,), (schedule,), model={"config_sha256": "x"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            trace.write(path)
            loaded = load_multi_request_trace(path)
            self.assertEqual(trace.as_dict(), loaded.as_dict())
            self.assertEqual(json.loads(path.read_text())["schema_version"], 3)

    def test_duplicate_ids_are_rejected(self):
        left = make_session("same", (0,), (1,))
        right = make_session("same", (2,), (3,))
        with self.assertRaises(ValueError):
            MultiRequestTrace((left, right), ())


if __name__ == "__main__":
    unittest.main()


# [Main Dev]
