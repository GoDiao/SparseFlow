GB = 1_000_000_000
GIB = 1024 ** 3


def format_bytes(value: int | float) -> str:
    value = float(value)
    if value >= GIB:
        return f"{value / GIB:.2f} GiB"
    if value >= 1024 ** 2:
        return f"{value / (1024 ** 2):.2f} MiB"
    if value >= 1024:
        return f"{value / 1024:.2f} KiB"
    return f"{value:.0f} B"


def gb_to_bytes(value: float) -> int:
    return int(value * GB)
