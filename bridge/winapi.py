"""Minimal Win32 helpers: find a window, capture it, send scan-code key presses."""
import ctypes
from ctypes import wintypes

import mss
import numpy as np

user32 = ctypes.WinDLL("user32", use_last_error=True)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # real pixel coordinates on scaled displays
except Exception:
    pass

# ---- windows ----------------------------------------------------------------
EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def find_window(title_part: str):
    """First visible top-level window whose title contains title_part (case-insensitive)."""
    found = []

    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if title_part.lower() in buf.value.lower():
                found.append((hwnd, buf.value))
        return True

    user32.EnumWindows(EnumWindowsProc(cb), 0)
    return found[0] if found else (None, None)


def client_rect(hwnd):
    """Client area of the window in screen coordinates: (left, top, width, height)."""
    r = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, r.right - r.left, r.bottom - r.top


def is_foreground(hwnd):
    return user32.GetForegroundWindow() == hwnd


def key_down_now(vk: int) -> bool:
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


VK_F12 = 0x7B


class Grabber:
    def __init__(self, hwnd):
        self.hwnd = hwnd
        self.sct = mss.mss()

    def grab(self) -> np.ndarray:
        """BGR image of the window's client area."""
        x, y, w, h = client_rect(self.hwnd)
        img = np.asarray(self.sct.grab({"left": x, "top": y, "width": w, "height": h}))
        return img[:, :, :3].copy()


# ---- keyboard (scan codes, which DOSBox reads) --------------------------------
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _U(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _U)]


# name -> (scan code, extended)
SCAN = {"up": (0x48, True), "down": (0x50, True), "left": (0x4B, True), "right": (0x4D, True),
        "space": (0x39, False), "enter": (0x1C, False), "esc": (0x01, False), "f11": (0x57, False)}


def _send(scan, extended, up):
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_EXTENDEDKEY if extended else 0) | (KEYEVENTF_KEYUP if up else 0)
    inp = INPUT(INPUT_KEYBOARD, _U(ki=KEYBDINPUT(0, scan, flags, 0, None)))
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


class Keys:
    """Tracks held keys and only sends changes."""

    def __init__(self, names):
        self.held = {n: False for n in names}

    def set(self, name, down: bool):
        if self.held.get(name) != down:
            _send(*SCAN[name], up=not down)
            self.held[name] = down

    def tap(self, name):
        self.set(name, True)
        self.set(name, False)

    def release_all(self):
        for n in list(self.held):
            self.set(n, False)


# ---- background control: no focus needed, so the user's desktop stays free ----
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
user32.GetDC.restype = wintypes.HDC
user32.GetDC.argtypes = [wintypes.HWND]
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.GetBitmapBits.argtypes = [wintypes.HBITMAP, wintypes.LONG, wintypes.LPVOID]

PW_CLIENT_FULL = 1         # PW_CLIENTONLY. Adding PW_RENDERFULLCONTENT queues on the compositor: 157 ms a grab with 12 windows
WM_KEYDOWN, WM_KEYUP = 0x0100, 0x0101
VK = {"up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
      "space": 0x20, "enter": 0x0D, "esc": 0x1B, "f11": 0x7A}


class WindowGrabber:
    """Copies a window's client area even when other windows cover it."""

    def __init__(self, hwnd):
        self.hwnd = hwnd

    def grab(self) -> np.ndarray:
        _, _, w, h = client_rect(self.hwnd)
        wdc = user32.GetDC(self.hwnd)
        mdc = gdi32.CreateCompatibleDC(wdc)
        bmp = gdi32.CreateCompatibleBitmap(wdc, w, h)
        old = gdi32.SelectObject(mdc, bmp)
        try:
            user32.PrintWindow(self.hwnd, mdc, PW_CLIENT_FULL)
            buf = np.empty((h, w, 4), np.uint8)
            gdi32.GetBitmapBits(bmp, w * h * 4, buf.ctypes.data)
            return buf[:, :, :3].copy()
        finally:
            gdi32.SelectObject(mdc, old)
            gdi32.DeleteObject(bmp)
            gdi32.DeleteDC(mdc)
            user32.ReleaseDC(self.hwnd, wdc)


class PostKeys:
    """Same interface as Keys, but posts key messages to one window only."""

    def __init__(self, hwnd, names):
        self.hwnd = hwnd
        self.held = {n: False for n in names}

    def set(self, name, down: bool):
        if self.held.get(name) == down:
            return
        scan, ext = SCAN[name]
        lp = 1 | (scan << 16) | (int(ext) << 24)
        if not down:
            lp |= (1 << 30) | (1 << 31)
        user32.PostMessageW(self.hwnd, WM_KEYDOWN if down else WM_KEYUP, VK[name], lp)
        self.held[name] = down

    def refresh(self):
        """Release and press held keys again; the game forgets them when a road restarts."""
        for n, down in list(self.held.items()):
            if down:
                self.set(n, False)
                self.set(n, True)

    def release_all(self):
        for n in list(self.held):
            self.set(n, False)
