"""The machine's performance cores. On a hybrid CPU (Intel P-cores + E-cores, Arm big.LITTLE) a pool spread over
every core waits on its slowest workers: the solve runs on the big cores only, all their threads busy.

``big_cpus()``: the logical processors of the most performant core class (all of them on a CPU with one class).
``use_big_cores()``: pins this process to them (worker processes started afterwards inherit it) and returns how
many there are, the default worker count."""
import os
import sys


def _windows():
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.GetSystemCpuSetInformation.argtypes = (ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG),
                                               wintypes.HANDLE, wintypes.ULONG)
    size = wintypes.ULONG(0)
    k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), None, 0)
    buf = ctypes.create_string_buffer(size.value)
    if not k32.GetSystemCpuSetInformation(buf, size, ctypes.byref(size), None, 0):
        return None
    raw, sets, off = buf.raw, [], 0
    while off < size.value:
        # SYSTEM_CPU_SET_INFORMATION: Size, Type, then CpuSet {Id, Group (u16), LogicalProcessorIndex (u8),
        # CoreIndex, LastLevelCacheIndex, NumaNodeIndex, EfficiencyClass (u8; higher = faster), ...}
        entry = int.from_bytes(raw[off:off + 4], 'little')
        group = int.from_bytes(raw[off + 12:off + 14], 'little')
        sets.append((group, raw[off + 14], raw[off + 18]))
        off += entry
    best = max(e for _, _, e in sets)
    # an affinity mask spans one processor group (64 logical processors): this process's
    return sorted(lp for g, lp, e in sets if e == best and g == 0)


def _linux():
    hybrid = '/sys/devices/cpu_core/cpus'       # Intel hybrid: the P-cores' logical processors
    if os.path.exists(hybrid):
        cpus = set()
        for part in open(hybrid).read().strip().split(','):
            a, _, b = part.partition('-')
            cpus.update(range(int(a), int(b or a) + 1))
        return sorted(cpus)
    allowed = sorted(os.sched_getaffinity(0))
    capacity = {}
    for c in allowed:                           # Arm big.LITTLE (and some x86): relative core capacity
        path = f'/sys/devices/system/cpu/cpu{c}/cpu_capacity'
        if os.path.exists(path):
            capacity[c] = int(open(path).read())
    if len(capacity) == len(allowed) and capacity:
        best = max(capacity.values())
        return [c for c in allowed if capacity[c] == best]
    return allowed


def _mac_count():
    import subprocess
    try:     # Apple silicon: performance level 0 = the P-cores (no affinity on macOS: only their count)
        return int(subprocess.run(['sysctl', '-n', 'hw.perflevel0.logicalcpu'], capture_output=True, text=True,
                                  check=True).stdout)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def big_cpus():
    try:
        if sys.platform == 'win32':
            return _windows()
        if sys.platform.startswith('linux'):
            return _linux()
    except OSError:
        pass
    return None


def use_big_cores():
    cpus = big_cpus()
    if cpus:
        if sys.platform == 'win32':
            import ctypes
            k32 = ctypes.WinDLL('kernel32', use_last_error=True)
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
            k32.SetProcessAffinityMask(k32.GetCurrentProcess(), sum(1 << c for c in cpus))
        else:
            os.sched_setaffinity(0, cpus)
        return len(cpus)
    if sys.platform == 'darwin':
        return _mac_count() or os.cpu_count() or 1
    return os.cpu_count() or 1
