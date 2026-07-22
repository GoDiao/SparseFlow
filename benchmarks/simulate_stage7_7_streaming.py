"""CLI alias for the Stage 7.7 raw shared-cache replay.

The implementation lives with route-union analysis so metadata and real raw
replay cannot silently diverge. This entry point keeps the plan's streaming
simulation command explicit.

[Main Dev]
"""

from __future__ import annotations

from analyze_stage7_7_union import main


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]
