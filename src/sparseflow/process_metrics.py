"""Small, optional-dependency-free process and host telemetry helpers."""

from __future__ import annotations

import os
from pathlib import Path
import platform
from typing import Any


def process_snapshot() -> dict[str, int | float | str]:
    """Return a portable snapshot of this process.

    Windows uses the native process APIs so production runtime reports the
    same kind of RSS data as the benchmark harness. The function deliberately
    imports ctypes lazily to keep inspection-only CLI commands lightweight.
    """

    if os.name == "nt":
        try:
            return _windows_process_snapshot()
        except (OSError, AttributeError, TypeError):
            return _unavailable_snapshot("windows-process-api-unavailable")

    status = _read_proc_file("/proc/self/status")
    io = _read_proc_file("/proc/self/io")
    usage = _posix_resource_usage()
    peak_rss = max(status.get("VmHWM", 0) * 1024, int(usage.get("peak_rss_bytes", 0)))
    return {
        "rss_bytes": status.get("VmRSS", 0) * 1024,
        "peak_rss_bytes": peak_rss,
        "private_bytes": _linux_private_bytes(),
        "read_bytes": io.get("read_bytes", 0),
        "read_chars": io.get("rchar", 0),
        "read_syscalls": io.get("syscr", 0),
        "read_bytes_semantics": "proc-self-io-storage-accounted",
        "user_seconds": usage.get("user_seconds", 0.0),
        "system_seconds": usage.get("system_seconds", 0.0),
        "minor_page_faults": usage.get("minor_page_faults", 0),
        "major_page_faults": usage.get("major_page_faults", 0),
        "voluntary_context_switches": usage.get("voluntary_context_switches", 0),
        "involuntary_context_switches": usage.get("involuntary_context_switches", 0),
        "page_faults": usage.get("minor_page_faults", 0)
        + usage.get("major_page_faults", 0),
        "platform_source": "proc-status-resource",
    }


def current_rss_bytes() -> int:
    return int(process_snapshot().get("rss_bytes", 0))


def peak_rss_bytes() -> int:
    return int(process_snapshot().get("peak_rss_bytes", 0))


def host_memory_snapshot() -> dict[str, int | str | None]:
    """Return physical memory totals without importing a third-party package."""

    if os.name == "nt":
        try:
            total, available = _windows_memory_snapshot()
            return {
                "total_bytes": total,
                "available_bytes": available,
                "source": "windows-global-memory-status",
            }
        except (OSError, AttributeError, TypeError):
            return {
                "total_bytes": 0,
                "available_bytes": 0,
                "source": "windows-memory-api-unavailable",
            }

    memory = _read_proc_file("/proc/meminfo")
    return {
        "total_bytes": memory.get("MemTotal", 0) * 1024,
        "available_bytes": memory.get("MemAvailable", 0) * 1024,
        "source": "proc-meminfo",
    }


def _read_proc_file(path: str) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if not separator:
                continue
            token = value.strip().split()[0] if value.strip() else "0"
            try:
                result[key] = int(token)
            except ValueError:
                continue
    except OSError:
        pass
    return result


def _posix_resource_usage() -> dict[str, int | float]:
    try:
        import resource
        import sys

        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak = int(usage.ru_maxrss)
        if sys.platform == "darwin":
            peak_bytes = peak
        else:
            peak_bytes = peak * 1024
        return {
            "peak_rss_bytes": peak_bytes,
            "user_seconds": usage.ru_utime,
            "system_seconds": usage.ru_stime,
            "minor_page_faults": int(usage.ru_minflt),
            "major_page_faults": int(usage.ru_majflt),
            "voluntary_context_switches": int(usage.ru_nvcsw),
            "involuntary_context_switches": int(usage.ru_nivcsw),
        }
    except (ImportError, AttributeError, ValueError):
        return {}


def _linux_private_bytes() -> int:
    values = _read_proc_file("/proc/self/smaps_rollup")
    return (values.get("Private_Clean", 0) + values.get("Private_Dirty", 0)) * 1024


def _unavailable_snapshot(source: str) -> dict[str, int | float | str]:
    return {
        "rss_bytes": 0,
        "peak_rss_bytes": 0,
        "private_bytes": 0,
        "read_bytes": 0,
        "read_chars": 0,
        "read_syscalls": 0,
        "read_bytes_semantics": "unavailable",
        "user_seconds": 0.0,
        "system_seconds": 0.0,
        "minor_page_faults": 0,
        "major_page_faults": 0,
        "voluntary_context_switches": 0,
        "involuntary_context_switches": 0,
        "page_faults": 0,
        "platform_source": source,
    }


def _windows_process_snapshot() -> dict[str, int | float | str]:
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    handle_type = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = handle_type
    psapi.GetProcessMemoryInfo.argtypes = [
        handle_type,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    kernel32.GetProcessIoCounters.argtypes = [
        handle_type,
        ctypes.POINTER(IoCounters),
    ]
    kernel32.GetProcessIoCounters.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = [
        handle_type,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL

    handle = kernel32.GetCurrentProcess()
    memory = ProcessMemoryCountersEx()
    memory.cb = ctypes.sizeof(memory)
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    io = IoCounters()
    if not kernel32.GetProcessIoCounters(handle, ctypes.byref(io)):
        raise ctypes.WinError(ctypes.get_last_error())
    created = wintypes.FILETIME()
    exited = wintypes.FILETIME()
    kernel = wintypes.FILETIME()
    user = wintypes.FILETIME()
    if not kernel32.GetProcessTimes(
        handle,
        ctypes.byref(created),
        ctypes.byref(exited),
        ctypes.byref(kernel),
        ctypes.byref(user),
    ):
        raise ctypes.WinError(ctypes.get_last_error())

    def seconds(value: wintypes.FILETIME) -> float:
        ticks = (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)
        return ticks / 10_000_000.0

    return {
        "rss_bytes": int(memory.WorkingSetSize),
        "peak_rss_bytes": int(memory.PeakWorkingSetSize),
        "private_bytes": int(memory.PrivateUsage),
        "read_bytes": int(io.ReadTransferCount),
        "read_chars": int(io.ReadTransferCount),
        "read_syscalls": int(io.ReadOperationCount),
        "read_bytes_semantics": "windows-process-read-transfer",
        "user_seconds": seconds(user),
        "system_seconds": seconds(kernel),
        "minor_page_faults": int(memory.PageFaultCount),
        "major_page_faults": 0,
        "voluntary_context_switches": 0,
        "involuntary_context_switches": 0,
        "page_faults": int(memory.PageFaultCount),
        "platform_source": "windows-process-memory-info",
    }


def _windows_memory_snapshot() -> tuple[int, int]:
    import ctypes
    from ctypes import wintypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatusEx)]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def host_snapshot() -> dict[str, Any]:
    memory = host_memory_snapshot()
    cpu_model = platform.processor() or None
    if os.name != "nt":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    cpu_model = line.partition(":")[2].strip()
                    break
        except OSError:
            pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model,
        "memory_total_bytes": memory["total_bytes"],
        "memory_available_bytes": memory["available_bytes"],
        "memory_source": memory["source"],
        "numa_nodes": _read_text("/sys/devices/system/node/online"),
        "pid": os.getpid(),
    }


def _read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
