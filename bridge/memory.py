"""Read the emulated PC's RAM from a running DOSBox process (read-only).

This is the same idea MarIO used in its emulator: observe the game's variables
while it runs. Nothing is written to the process or to the game's files.
"""
import ctypes
from ctypes import wintypes

import numpy as np

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
PROCESS_VM_READ, PROCESS_QUERY_INFORMATION = 0x0010, 0x0400
MEM_COMMIT = 0x1000
READABLE = {0x02, 0x04, 0x20, 0x40}    # PAGE_READONLY, READWRITE, EXECUTE_READ, EXECUTE_READWRITE
BIOS_DATE_OFFSET = 0xFFFF5             # DOSBox writes its BIOS date here
BIOS_DATE = b"01/01/92"
CONVENTIONAL = 0xA0000                 # the 640 KB a DOS game lives in


class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", wintypes.DWORD), ("PartitionId", wintypes.WORD),
                ("RegionSize", ctypes.c_size_t), ("State", wintypes.DWORD),
                ("Protect", wintypes.DWORD), ("Type", wintypes.DWORD)]


kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.VirtualQueryEx.restype = ctypes.c_size_t
kernel32.VirtualQueryEx.argtypes = [wintypes.HANDLE, ctypes.c_void_p,
                                    ctypes.POINTER(MEMORY_BASIC_INFORMATION), ctypes.c_size_t]
kernel32.ReadProcessMemory.restype = wintypes.BOOL
kernel32.ReadProcessMemory.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]


class EmulatedRAM:
    """Read-only view of a DOSBox process's emulated memory."""

    def __init__(self, pid: int):
        self.handle = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid)
        if not self.handle:
            raise OSError(f"cannot open process {pid}: {ctypes.get_last_error()}")
        self.base = self._find_base()

    def _regions(self):
        addr, info = 0, MEMORY_BASIC_INFORMATION()
        while kernel32.VirtualQueryEx(self.handle, addr, ctypes.byref(info), ctypes.sizeof(info)):
            base = info.BaseAddress or 0
            if info.State == MEM_COMMIT and info.Protect in READABLE:
                yield base, info.RegionSize
            addr = base + info.RegionSize
            if addr >= 0x7FFFFFFFFFFF:
                break

    def _raw(self, addr, n):
        buf = ctypes.create_string_buffer(n)
        got = ctypes.c_size_t()
        if not kernel32.ReadProcessMemory(self.handle, addr, buf, n, ctypes.byref(got)) or got.value != n:
            return None
        return buf.raw

    def _find_base(self):
        # the RAM array sits a little way into its allocation, so look near the expected spot
        for base, size in self._regions():
            if size < BIOS_DATE_OFFSET + 4096:
                continue
            near = self._raw(base + BIOS_DATE_OFFSET, 4096)
            if near:
                i = near.find(BIOS_DATE)
                if i >= 0 and near[max(0, i - 5)] == 0xEA:    # the reset vector's far jump
                    return base + i
        raise OSError("emulated RAM not found")

    def read(self, offset: int, n: int) -> bytes:
        data = self._raw(self.base + offset, n)
        if data is None:
            raise OSError("memory read failed")
        return data

    def snapshot(self, n: int = CONVENTIONAL) -> np.ndarray:
        return np.frombuffer(self.read(0, n), np.uint8)

    def close(self):
        if self.handle:
            kernel32.CloseHandle(self.handle)
            self.handle = None
