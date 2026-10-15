#!/usr/bin/env python3
"""
Real-Time Command-Line Signal Grapher for Transforms & Signals
==============================================================
Interactive CLI tool that updates a live Matplotlib window in real time as you
add points (e.g., `3,4`), generate signals, apply time/amplitude transformations,
compute convolutions, or inspect the Frequency Domain (FFT).
"""

import copy
import csv
import math
import os
import re
import select
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure Matplotlib caches its font list inside the project directory so startup is instant
_PROJECT_DIR = Path(__file__).resolve().parent
_MPL_CACHE_DIR = _PROJECT_DIR / ".venv" / ".mplconfig"
try:
    _MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
except OSError:
    pass

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    # Automatically re-exec using the project's .venv if available
    venv_python = _PROJECT_DIR / ".venv" / "bin" / "python3"
    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        os.execv(str(venv_python), [str(venv_python), *sys.argv])
    raise


# Colors for multiple signals
SIGNAL_COLORS = [
    "#2563eb",  # Blue
    "#dc2626",  # Red
    "#16a34a",  # Green
    "#9333ea",  # Purple
    "#ea580c",  # Orange
    "#0891b2",  # Cyan
    "#db2777",  # Pink
]


@dataclass
class Signal:
    name: str
    points: Dict[float, float] = field(default_factory=dict)
    mode: str = "stem"  # 'stem', 'line', 'step', 'both'
    color: str = "#2563eb"
    visible: bool = True
    history: List[Dict[float, float]] = field(default_factory=list)

    def save_undo(self) -> None:
        self.history.append(copy.deepcopy(self.points))
        if len(self.history) > 50:
            self.history.pop(0)

    def undo(self) -> bool:
        if not self.history:
            return False
        self.points = self.history.pop()
        return True

    def set_point(self, x: float, y: float) -> None:
        # Clean floating point keys like -0.0 or 2.9999999999999
        x_clean = round(float(x), 8)
        if x_clean == 0.0:
            x_clean = 0.0
        self.points[x_clean] = round(float(y), 8)

    def remove_point(self, x: float) -> bool:
        x_clean = round(float(x), 8)
        if x_clean in self.points:
            del self.points[x_clean]
            return True
        # Fallback approximate match
        for k in list(self.points.keys()):
            if math.isclose(k, x, abs_tol=1e-6):
                del self.points[k]
                return True
        return False

    def get_sorted_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.points:
            return np.array([]), np.array([])
        sorted_items = sorted(self.points.items(), key=lambda item: item[0])
        xs = np.array([item[0] for item in sorted_items], dtype=float)
        ys = np.array([item[1] for item in sorted_items], dtype=float)
        return xs, ys

    def energy(self) -> float:
        if not self.points:
            return 0.0
        _, ys = self.get_sorted_xy()
        return float(np.sum(np.abs(ys) ** 2))

    def power(self) -> float:
        if not self.points:
            return 0.0
        _, ys = self.get_sorted_xy()
        return float(np.mean(np.abs(ys) ** 2))


class SignalGrapherApp:
    def __init__(self, initial_mode: str = "stem") -> None:
        self.signals: Dict[str, Signal] = {}
        self.active_name: str = "x"
        self.default_mode: str = initial_mode
        self.show_fft: bool = False
        self.show_grid: bool = True
        self.annotate_points: bool = True
        self.xlim: Optional[Tuple[float, float]] = None
        self.ylim: Optional[Tuple[float, float]] = None
        self.running: bool = True
        self.dirty: bool = True
        self.save_request: Optional[str] = None

        # Create initial signal 'x'
        self._get_or_create_signal("x")

    def _get_or_create_signal(self, name: str, mode: Optional[str] = None) -> Signal:
        if name not in self.signals:
            color = SIGNAL_COLORS[len(self.signals) % len(SIGNAL_COLORS)]
            self.signals[name] = Signal(
                name=name,
                mode=mode or self.default_mode,
                color=color,
            )
        return self.signals[name]

    @property
    def active(self) -> Signal:
        return self._get_or_create_signal(self.active_name)

    # -------------------------------------------------------------------------
    # Command Parsing & Execution
    # -------------------------------------------------------------------------
    def handle_command(self, raw_line: str) -> None:
        line = raw_line.strip()
        if not line:
            return

        # Support multiple commands separated by ';'
        if ";" in line:
            for part in line.split(";"):
                if part.strip():
                    self.handle_command(part.strip())
            return

        # 1. Direct coordinate shorthand: e.g. "3,4" or "0,1 1,2 2,1" or "-1, 2.5"
        direct_coords = self._try_parse_coordinates(line)
        if direct_coords is not None:
            self.active.save_undo()
            for x_val, y_val in direct_coords:
                self.active.set_point(x_val, y_val)
            pts_str = ", ".join(f"({x:g}, {y:g})" for x, y in direct_coords)
            print(f"  [+] Added to '{self.active.name}': {pts_str}")
            self.dirty = True
            return

        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        try:
            if cmd in ("q", "quit", "exit"):
                print("Closing Signal Grapher...")
                self.running = False

            elif cmd in ("h", "help", "?"):
                self.print_help()

            elif cmd in ("add", "pt", "p"):
                rest = " ".join(args)
                coords = self._try_parse_coordinates(rest)
                if not coords and len(args) == 2:
                    coords = [(float(args[0]), float(args[1]))]
                if not coords:
                    print("  [!] Usage: add <x>,<y> [x2,y2 ...]  OR simply type: 3,4")
                    return
                self.active.save_undo()
                for x_val, y_val in coords:
                    self.active.set_point(x_val, y_val)
                pts_str = ", ".join(f"({x:g}, {y:g})" for x, y in coords)
                print(f"  [+] Added to '{self.active.name}': {pts_str}")
                self.dirty = True

            elif cmd in ("rm", "del", "remove", "delete"):
                if not args:
                    print("  [!] Usage: rm <x1> [x2 ...]")
                    return
                self.active.save_undo()
                for arg in args:
                    x_val = float(arg)
                    if self.active.remove_point(x_val):
                        print(f"  [-] Removed point at x={x_val:g} from '{self.active.name}'")
                    else:
                        print(f"  [!] No point found at x={x_val:g} in '{self.active.name}'")
                self.dirty = True

            elif cmd == "clear":
                if args and args[0].lower() == "all":
                    self.signals.clear()
                    self._get_or_create_signal(self.active_name)
                    print("  [*] Cleared all signals.")
                else:
                    self.active.save_undo()
                    self.active.points.clear()
                    print(f"  [*] Cleared signal '{self.active.name}'.")
                self.dirty = True

            elif cmd == "undo":
                if self.active.undo():
                    print(f"  [*] Undid last action on '{self.active.name}'.")
                    self.dirty = True
                else:
                    print(f"  [!] Nothing to undo for '{self.active.name}'.")

            elif cmd in ("mode", "style"):
                if not args or args[0].lower() not in ("stem", "line", "step", "both"):
                    print("  [!] Usage: mode <stem|line|step|both>")
                    return
                new_mode = args[0].lower()
                self.active.mode = new_mode
                self.default_mode = new_mode
                print(f"  [*] Set '{self.active.name}' plot mode to '{new_mode}'.")
                self.dirty = True

            elif cmd in ("stem", "line", "step", "both"):
                self.active.mode = cmd
                self.default_mode = cmd
                print(f"  [*] Set '{self.active.name}' plot mode to '{cmd}'.")
                self.dirty = True

            elif cmd in ("sig", "use", "select"):
                if not args:
                    print(f"  [*] Active signal: '{self.active.name}' ({len(self.active.points)} points)")
                    return
                sig_name = args[0]
                sig_mode = args[1].lower() if len(args) > 1 else None
                self._get_or_create_signal(sig_name, mode=sig_mode)
                self.active_name = sig_name
                print(f"  [*] Switched active signal to '{sig_name}' (mode: {self.active.mode}).")
                self.dirty = True

            elif cmd in ("list", "ls", "info", "stats"):
                self.print_signals_info()

            elif cmd == "hide":
                target = args[0] if args else self.active.name
                if target in self.signals:
                    self.signals[target].visible = False
                    print(f"  [*] Hid signal '{target}'.")
                    self.dirty = True
                else:
                    print(f"  [!] Signal '{target}' not found.")

            elif cmd == "show":
                target = args[0] if args else self.active.name
                if target in self.signals:
                    self.signals[target].visible = True
                    print(f"  [*] Showing signal '{target}'.")
                    self.dirty = True
                else:
                    print(f"  [!] Signal '{target}' not found.")

            elif cmd == "del_sig":
                if not args:
                    print("  [!] Usage: del_sig <signal_name>")
                    return
                target = args[0]
                if target in self.signals:
                    del self.signals[target]
                    if self.active_name == target:
                        self.active_name = next(iter(self.signals.keys()), "x")
                        self._get_or_create_signal(self.active_name)
                    print(f"  [-] Deleted signal '{target}'. Active signal is '{self.active_name}'.")
                    self.dirty = True
                else:
                    print(f"  [!] Signal '{target}' not found.")

            elif cmd == "labels":
                self.annotate_points = not self.annotate_points
                state = "ON" if self.annotate_points else "OFF"
                print(f"  [*] Point coordinate labels: {state}")
                self.dirty = True

            elif cmd == "fft":
                if args and args[0].lower() in ("on", "true", "1"):
                    self.show_fft = True
                elif args and args[0].lower() in ("off", "false", "0"):
                    self.show_fft = False
                else:
                    self.show_fft = not self.show_fft
                state = "ON" if self.show_fft else "OFF"
                print(f"  [*] Frequency Domain (FFT Magnitude & Phase) panel: {state}")
                self.dirty = True

            # -----------------------------------------------------------------
            # Signal Operations (Transforms & Signals Course Operations)
            # -----------------------------------------------------------------
            elif cmd == "shift":
                # y(t) = x(t - t0) -> shifts signal right by t0 (x_new = x_old + t0)
                if not args:
                    print("  [!] Usage: shift <t0>  (performs y(t) = x(t - t0), moving points to x + t0)")
                    return
                t0 = float(args[0])
                self.active.save_undo()
                new_pts = {}
                for x_val, y_val in self.active.points.items():
                    new_pts[round(x_val + t0, 8)] = y_val
                self.active.points = new_pts
                direction = "right (delay)" if t0 >= 0 else "left (advance)"
                sign_str = f"- {t0:g}" if t0 >= 0 else f"+ {abs(t0):g}"
                print(f"  [*] Time shift {self.active.name}(t {sign_str}): shifted {direction} by {abs(t0):g}.")
                self.dirty = True

            elif cmd in ("reverse", "flip", "reflect"):
                # y(t) = x(-t) -> x_new = -x_old
                self.active.save_undo()
                new_pts = {}
                for x_val, y_val in self.active.points.items():
                    nx = round(-x_val, 8)
                    if nx == 0.0:
                        nx = 0.0
                    new_pts[nx] = y_val
                self.active.points = new_pts
                print(f"  [*] Time reversal {self.active.name}(-t): reflected across y-axis.")
                self.dirty = True

            elif cmd == "tscale":
                # y(t) = x(a * t) -> x_new = x_old / a
                if not args:
                    print("  [!] Usage: tscale <a>  (performs y(t) = x(a*t), scaling time coordinates by 1/a)")
                    return
                a = float(args[0])
                if a == 0:
                    print("  [!] Time scale factor 'a' cannot be 0.")
                    return
                self.active.save_undo()
                new_pts = {}
                for x_val, y_val in self.active.points.items():
                    nx = round(x_val / a, 8)
                    if nx == 0.0:
                        nx = 0.0
                    new_pts[nx] = y_val
                self.active.points = new_pts
                print(f"  [*] Time scaled {self.active.name}({a:g}t): divided x-coordinates by {a:g}.")
                self.dirty = True

            elif cmd in ("scale", "ascale"):
                # y(t) = A * x(t)
                if not args:
                    print("  [!] Usage: scale <A>  (multiplies signal amplitude y by A)")
                    return
                amp = float(args[0])
                self.active.save_undo()
                for x_val in list(self.active.points.keys()):
                    self.active.points[x_val] = round(self.active.points[x_val] * amp, 8)
                print(f"  [*] Amplitude scaled '{self.active.name}' by {amp:g}.")
                self.dirty = True

            elif cmd == "offset":
                # y(t) = x(t) + B
                if not args:
                    print("  [!] Usage: offset <B>  (adds constant DC offset B to y)")
                    return
                dc = float(args[0])
                self.active.save_undo()
                for x_val in list(self.active.points.keys()):
                    self.active.points[x_val] = round(self.active.points[x_val] + dc, 8)
                print(f"  [*] Added DC offset {dc:g} to '{self.active.name}'.")
                self.dirty = True

            elif cmd in ("even", "odd"):
                # Even part: 0.5 * (x(t) + x(-t)), Odd part: 0.5 * (x(t) - x(-t))
                target_name = args[0] if args else f"{self.active.name}_{cmd}"
                src = self.active
                all_x = sorted(set(list(src.points.keys()) + [-k for k in src.points.keys()]))
                target_sig = self._get_or_create_signal(target_name, mode=src.mode)
                target_sig.save_undo()
                target_sig.points.clear()
                for x_val in all_x:
                    v_pos = src.points.get(round(x_val, 8), 0.0)
                    v_neg = src.points.get(round(-x_val, 8), 0.0)
                    val = 0.5 * (v_pos + v_neg) if cmd == "even" else 0.5 * (v_pos - v_neg)
                    target_sig.set_point(x_val, val)
                print(f"  [*] Computed {cmd} component into signal '{target_name}'.")
                self.dirty = True

            elif cmd in ("cumsum", "integrate"):
                self.active.save_undo()
                xs, ys = self.active.get_sorted_xy()
                csum = np.cumsum(ys)
                for x_val, y_val in zip(xs, csum):
                    self.active.set_point(float(x_val), float(y_val))
                print(f"  [*] Applied running sum (discrete integration) to '{self.active.name}'.")
                self.dirty = True

            elif cmd == "diff":
                self.active.save_undo()
                xs, ys = self.active.get_sorted_xy()
                if len(ys) == 0:
                    return
                diffs = np.diff(ys, prepend=0.0)
                for x_val, y_val in zip(xs, diffs):
                    self.active.set_point(float(x_val), float(y_val))
                print(f"  [*] Applied first difference x[n] - x[n-1] to '{self.active.name}'.")
                self.dirty = True

            elif cmd == "conv":
                # Usage: conv <other_sig> [target_sig]  OR  conv <sig1> <sig2> [target_sig]
                if not args:
                    print("  [!] Usage: conv <sig2> [out_sig]  OR  conv <sig1> <sig2> [out_sig]")
                    return
                if len(args) == 1:
                    s1_name, s2_name, out_name = self.active.name, args[0], "y_conv"
                elif len(args) == 2 and args[0] in self.signals and args[1] not in self.signals:
                    s1_name, s2_name, out_name = self.active.name, args[0], args[1]
                else:
                    s1_name = args[0]
                    s2_name = args[1]
                    out_name = args[2] if len(args) > 2 else "y_conv"

                if s1_name not in self.signals or s2_name not in self.signals:
                    print(f"  [!] Both signals '{s1_name}' and '{s2_name}' must exist.")
                    return

                s1 = self.signals[s1_name]
                s2 = self.signals[s2_name]
                if not s1.points or not s2.points:
                    print("  [!] Cannot convolve empty signals.")
                    return

                # Discrete convolution sum_{k} s1[k] * s2[n - k]
                conv_pts: Dict[float, float] = {}
                for x1, y1 in s1.points.items():
                    for x2, y2 in s2.points.items():
                        nx = round(x1 + x2, 8)
                        if nx == 0.0:
                            nx = 0.0
                        conv_pts[nx] = round(conv_pts.get(nx, 0.0) + y1 * y2, 8)

                out_sig = self._get_or_create_signal(out_name, mode="stem")
                out_sig.save_undo()
                out_sig.points = conv_pts
                self.active_name = out_name
                print(
                    f"  [*] Convolved '{s1_name}' * '{s2_name}' -> '{out_name}' "
                    f"({len(conv_pts)} points, active='{out_name}')."
                )
                self.dirty = True

            elif cmd in ("add_sig", "sub_sig", "mul_sig"):
                if len(args) < 2:
                    print(f"  [!] Usage: {cmd} <sig1> <sig2> [out_sig]")
                    return
                s1_name, s2_name = args[0], args[1]
                out_name = args[2] if len(args) > 2 else f"{s1_name}_{cmd[:3]}_{s2_name}"
                if s1_name not in self.signals or s2_name not in self.signals:
                    print(f"  [!] Signals '{s1_name}' and '{s2_name}' must exist.")
                    return
                s1, s2 = self.signals[s1_name], self.signals[s2_name]
                all_x = sorted(set(list(s1.points.keys()) + list(s2.points.keys())))
                out_sig = self._get_or_create_signal(out_name, mode=s1.mode)
                out_sig.save_undo()
                out_sig.points.clear()
                for x_val in all_x:
                    v1 = s1.points.get(x_val, 0.0)
                    v2 = s2.points.get(x_val, 0.0)
                    if cmd == "add_sig":
                        res = v1 + v2
                    elif cmd == "sub_sig":
                        res = v1 - v2
                    else:
                        res = v1 * v2
                    out_sig.set_point(x_val, res)
                self.active_name = out_name
                print(f"  [*] Created '{out_name}' from '{s1_name}' and '{s2_name}'.")
                self.dirty = True

            # -----------------------------------------------------------------
            # Signal Generators (Step, Impulse, Rect, Sinusoids, Expressions)
            # -----------------------------------------------------------------
            elif cmd in ("impulse", "delta"):
                # delta [x0=0] [amp=1]
                x0 = float(args[0]) if len(args) >= 1 else 0.0
                amp = float(args[1]) if len(args) >= 2 else 1.0
                self.active.save_undo()
                self.active.set_point(x0, amp)
                print(f"  [+] Added impulse {amp:g}*delta[n - {x0:g}] at x={x0:g}, y={amp:g}")
                self.dirty = True

            elif cmd == "ustep":
                # ustep <xmin> <xmax> [x0=0] [amp=1]
                if len(args) < 2:
                    print("  [!] Usage: ustep <xmin> <xmax> [x0=0] [amp=1]")
                    return
                xmin, xmax = int(float(args[0])), int(float(args[1]))
                x0 = float(args[2]) if len(args) >= 3 else 0.0
                amp = float(args[3]) if len(args) >= 4 else 1.0
                self.active.save_undo()
                for n in range(min(xmin, xmax), max(xmin, xmax) + 1):
                    self.active.set_point(n, amp if n >= x0 else 0.0)
                print(f"  [+] Generated unit step {amp:g}*u[n - {x0:g}] on [{xmin}, {xmax}].")
                self.dirty = True

            elif cmd == "rect":
                # rect <xmin> <xmax> <half_width> [amp=1]
                if len(args) < 3:
                    print("  [!] Usage: rect <xmin> <xmax> <half_width> [amp=1]")
                    return
                xmin, xmax = int(float(args[0])), int(float(args[1]))
                hw = float(args[2])
                amp = float(args[3]) if len(args) >= 4 else 1.0
                self.active.save_undo()
                for n in range(min(xmin, xmax), max(xmin, xmax) + 1):
                    self.active.set_point(n, amp if abs(n) <= hw else 0.0)
                print(f"  [+] Generated rectangular pulse on [{xmin}, {xmax}] with |x| <= {hw:g}.")
                self.dirty = True

            elif cmd in ("sin", "cos"):
                # sin <freq> <xmin> <xmax> [step=1] [amp=1] [phase_deg=0]
                if len(args) < 3:
                    print(f"  [!] Usage: {cmd} <freq> <xmin> <xmax> [step=0.1] [amp=1] [phase_deg=0]")
                    return
                freq = float(args[0])
                xmin, xmax = float(args[1]), float(args[2])
                step = float(args[3]) if len(args) >= 4 else 0.1
                amp = float(args[4]) if len(args) >= 5 else 1.0
                phase = math.radians(float(args[5])) if len(args) >= 6 else 0.0
                self.active.save_undo()
                ts = np.arange(xmin, xmax + step * 0.5, step)
                fn = np.sin if cmd == "sin" else np.cos
                ys = amp * fn(2 * np.pi * freq * ts + phase)
                for t_val, y_val in zip(ts, ys):
                    self.active.set_point(float(t_val), float(y_val))
                print(
                    f"  [+] Generated {amp:g}*{cmd}(2*pi*{freq:g}*t) on [{xmin:g}, {xmax:g}] "
                    f"(step={step:g}, {len(ts)} pts)."
                )
                self.dirty = True

            elif cmd == "expr":
                # expr <formula> <xmin> <xmax> [step=1]
                if len(args) < 3:
                    print(
                        "  [!] Usage: expr <formula_in_t_or_n> <xmin> <xmax> [step=1]\n"
                        "      Example: expr (n>=0)*(0.8**n) -5 15 1\n"
                        "      Example: expr sin(2*pi*t)*exp(-0.5*t) 0 6 0.1"
                    )
                    return
                # Allow spaces inside formula if last 2 or 3 tokens are numbers
                step = 1.0
                try:
                    # Try parsing last 3 tokens as xmin, xmax, step
                    if len(args) >= 4:
                        try:
                            xmin_try = float(args[-3])
                            xmax_try = float(args[-2])
                            step_try = float(args[-1])
                            formula = " ".join(args[:-3])
                            xmin, xmax, step = xmin_try, xmax_try, step_try
                        except ValueError:
                            xmin = float(args[-2])
                            xmax = float(args[-1])
                            formula = " ".join(args[:-2])
                    else:
                        xmin = float(args[-2])
                        xmax = float(args[-1])
                        formula = " ".join(args[:-2])
                except ValueError:
                    print("  [!] Usage: expr <formula> <xmin> <xmax> [step=1]")
                    return

                ts = np.arange(xmin, xmax + step * 0.5, step)
                safe_env = {
                    "t": ts,
                    "n": ts,
                    "x": ts,
                    "pi": np.pi,
                    "e": np.e,
                    "sin": np.sin,
                    "cos": np.cos,
                    "tan": np.tan,
                    "exp": np.exp,
                    "log": np.log,
                    "sqrt": np.sqrt,
                    "abs": np.abs,
                    "sinc": np.sinc,
                    "sign": np.sign,
                    "u": lambda z: np.where(z >= 0, 1.0, 0.0),
                    "delta": lambda z: np.where(np.isclose(z, 0.0, atol=1e-7), 1.0, 0.0),
                    "rect": lambda z: np.where(np.abs(z) <= 0.5, 1.0, 0.0),
                    "tri": lambda z: np.maximum(0.0, 1.0 - np.abs(z)),
                }
                ys = eval(formula, {"__builtins__": {}}, safe_env)  # noqa: S307
                if np.isscalar(ys):
                    ys = np.full_like(ts, float(ys))
                ys = np.asarray(ys, dtype=float)
                self.active.save_undo()
                for t_val, y_val in zip(ts, ys):
                    if np.isfinite(y_val):
                        self.active.set_point(float(t_val), float(y_val))
                print(f"  [+] Evaluated '{formula}' on [{xmin:g}, {xmax:g}] (step={step:g}, {len(ts)} pts).")
                self.dirty = True

            elif cmd == "xlim":
                if not args or args[0].lower() == "auto":
                    self.xlim = None
                    print("  [*] X-axis limits set to auto.")
                elif len(args) >= 2:
                    self.xlim = (float(args[0]), float(args[1]))
                    print(f"  [*] X-axis limits set to [{self.xlim[0]:g}, {self.xlim[1]:g}].")
                self.dirty = True

            elif cmd == "ylim":
                if not args or args[0].lower() == "auto":
                    self.ylim = None
                    print("  [*] Y-axis limits set to auto.")
                elif len(args) >= 2:
                    self.ylim = (float(args[0]), float(args[1]))
                    print(f"  [*] Y-axis limits set to [{self.ylim[0]:g}, {self.ylim[1]:g}].")
                self.dirty = True

            elif cmd == "save":
                filename = args[0] if args else "signal_plot.png"
                if filename.lower().endswith(".csv"):
                    self._export_csv(filename)
                else:
                    self.save_request = filename
                    self.dirty = True

            else:
                print(
                    f"  [?] Unknown command '{cmd}'. Type '3,4' to add a point, or 'help' for all commands."
                )

        except Exception as exc:
            print(f"  [!] Error: {exc}")

    @staticmethod
    def _try_parse_coordinates(text: str) -> Optional[List[Tuple[float, float]]]:
        """
        Parse coordinate inputs such as:
          - "3,4"
          - "3, 4"
          - "(3, 4)"
          - "3 4"
          - "0,0 1,1 2,4 3,9"
        Returns a list of (x, y) tuples if the entire text is valid coordinate(s), else None.
        """
        cleaned = text.strip()
        if not cleaned:
            return None

        # Case 1: Two space-separated numbers like "3 4" or "-2 5.5"
        space_match = re.fullmatch(
            r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
            cleaned,
        )
        if space_match:
            return [(float(space_match.group(1)), float(space_match.group(2)))]

        # Case 2: One or more comma-separated pairs like "3,4" or "(1, 2) (3, 4)" or "0,1 1,2 2,1"
        pair_pattern = re.compile(
            r"\(?\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*\)?"
        )
        matches = list(pair_pattern.finditer(cleaned))
        if not matches:
            return None

        # Verify the matches cover the whole string (ignoring whitespace and semicolons)
        reconstructed = pair_pattern.sub("", cleaned).strip()
        if reconstructed:
            return None

        return [(float(m.group(1)), float(m.group(2))) for m in matches]

    def _export_csv(self, filename: str) -> None:
        xs, ys = self.active.get_sorted_xy()
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x", self.active.name])
            for x_val, y_val in zip(xs, ys):
                writer.writerow([x_val, y_val])
        print(f"  [*] Exported {len(xs)} points from '{self.active.name}' to {filename}")

    def print_signals_info(self) -> None:
        print("\n  --- Signals Overview ---")
        for name, sig in self.signals.items():
            marker = "*" if name == self.active_name else " "
            vis = "visible" if sig.visible else "hidden"
            xs, ys = sig.get_sorted_xy()
            pts_preview = ", ".join(f"({x:g},{y:g})" for x, y in list(zip(xs, ys))[:8])
            if len(xs) > 8:
                pts_preview += f", ... ({len(xs)} total)"
            print(
                f"  [{marker}] {name} (mode={sig.mode}, {vis}): "
                f"N={len(xs)}, Energy E={sig.energy():.4g}, Power P={sig.power():.4g}"
            )
            if len(xs) > 0:
                print(f"      Points: {pts_preview}")
        print()

    def print_help(self) -> None:
        print(
            """
  ==================== SIGNAL GRAPHER COMMANDS ====================
  ADDING & EDITING POINTS:
    3,4                 Add/update point at x=3, y=4 (or '3 4')
    0,1 1,2 2,1         Add multiple (x,y) points at once
    rm <x1> [x2...]     Remove point(s) at given x coordinate(s)
    clear [all]         Clear active signal (or 'clear all' for all)
    undo                Undo last modification on active signal
    labels              Toggle (x, y) coordinate text labels on graph

  PLOT STYLES & MULTIPLE SIGNALS:
    stem | line | step | both
                        Switch plot style (discrete stem vs continuous line)
    sig <name> [mode]   Switch to (or create) another signal (e.g. 'sig h stem')
    list                Show all signals, points, Energy (E), and Power (P)
    hide / show [name]  Hide or show a signal on the graph
    fft [on|off]        Toggle live Frequency Domain (FFT Magnitude & Phase)

  TRANSFORMS & SIGNALS OPERATIONS:
    shift <t0>          Time shift: y(t) = x(t - t0)  (moves right by t0)
    reverse             Time reversal: y(t) = x(-t)   (flips across y-axis)
    tscale <a>          Time scaling: y(t) = x(a*t)   (scales x by 1/a)
    scale <A>           Amplitude scaling: y(t) = A * x(t)
    offset <B>          DC offset: y(t) = x(t) + B
    even / odd [out]    Extract even x_e(t) or odd x_o(t) component
    conv <sig2> [out]   Discrete convolution: y[n] = (x * sig2)[n]
    add_sig <s1> <s2>   Add two signals: s1(t) + s2(t)
    mul_sig <s1> <s2>   Multiply two signals: s1(t) * s2(t)
    cumsum / diff       Running sum (integral) or first difference

  SIGNAL GENERATORS:
    impulse [x0] [A]    Add discrete impulse A * delta[n - x0]
    ustep <min> <max> [x0] [A]
                        Generate unit step A * u[n - x0]
    rect <min> <max> <half_width> [A]
                        Generate rectangular pulse
    sin <f> <min> <max> [step] [A] [phase_deg]
                        Generate sinusoid (also 'cos')
    expr <formula> <min> <max> [step]
                        Evaluate math expression in t or n (supports u(t),
                        delta(n), rect(t), tri(t), sinc(t), sin, cos, exp...)
                        e.g.: expr (n>=0)*(0.8**n) -4 12 1

  VIEW & EXPORT:
    xlim / ylim <min> <max> | auto
    save <file.png|csv> Save figure image or export points to CSV
    quit / exit         Close the grapher
  =================================================================
"""
        )


def render_plot(fig: plt.Figure, app: SignalGrapherApp) -> None:
    """Redraws the Matplotlib figure based on current application state."""
    fig.clf()

    if app.show_fft:
        gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.0, 1.0], hspace=0.42)
        ax_time = fig.add_subplot(gs[0])
        ax_mag = fig.add_subplot(gs[1])
        ax_phase = fig.add_subplot(gs[2])
    else:
        ax_time = fig.add_subplot(111)
        ax_mag = None
        ax_phase = None

    # Draw origin axes (x=0 and y=0)
    ax_time.axhline(0, color="#64748b", linewidth=1.1, zorder=1)
    ax_time.axvline(0, color="#64748b", linewidth=1.1, zorder=1)

    all_xs: List[float] = []
    all_ys: List[float] = []
    has_any_points = False

    for name, sig in app.signals.items():
        if not sig.visible or not sig.points:
            continue

        xs, ys = sig.get_sorted_xy()
        if len(xs) == 0:
            continue

        has_any_points = True
        all_xs.extend(xs.tolist())
        all_ys.extend(ys.tolist())

        is_active = name == app.active_name
        lw = 2.2 if is_active else 1.5
        alpha = 1.0 if is_active else 0.75
        label = (
            f"{name} [{sig.mode}] (N={len(xs)}, E={sig.energy():.3g})"
            if is_active
            else f"{name} [{sig.mode}]"
        )

        if sig.mode in ("stem", "both"):
            markerline, stemlines, baseline = ax_time.stem(
                xs,
                ys,
                label=label if sig.mode == "stem" else None,
            )
            plt.setp(stemlines, color=sig.color, linewidth=lw, alpha=alpha, zorder=3)
            plt.setp(
                markerline,
                color=sig.color,
                markersize=7 if is_active else 5,
                alpha=alpha,
                zorder=4,
            )
            plt.setp(baseline, visible=False)

        if sig.mode in ("line", "both"):
            ax_time.plot(
                xs,
                ys,
                color=sig.color,
                linewidth=lw,
                alpha=alpha,
                marker="o" if len(xs) <= 40 else None,
                markersize=6 if is_active else 4,
                label=label,
                zorder=4,
            )

        if sig.mode == "step":
            ax_time.step(
                xs,
                ys,
                where="post",
                color=sig.color,
                linewidth=lw,
                alpha=alpha,
                marker="o" if len(xs) <= 40 else None,
                markersize=5,
                label=label,
                zorder=4,
            )

        # Annotate coordinates if enabled and point count is reasonable
        if app.annotate_points and len(xs) <= 25:
            for x_val, y_val in zip(xs, ys):
                offset_y = 8 if y_val >= 0 else -14
                ax_time.annotate(
                    f"({x_val:g}, {y_val:g})",
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=(0, offset_y),
                    ha="center",
                    fontsize=8.5,
                    color=sig.color,
                    weight="bold" if is_active else "normal",
                )

    # Configure Time-Domain Axes
    ax_time.grid(True, which="both", linestyle="--", alpha=0.45)
    ax_time.set_title(
        f"Time Domain Signal — Active: {app.active_name}[n] / {app.active_name}(t)",
        fontsize=12,
        weight="bold",
    )
    ax_time.set_xlabel("Time (t or n)")
    ax_time.set_ylabel("Amplitude")

    if has_any_points:
        ax_time.legend(loc="upper right", framealpha=0.9)
        if app.xlim:
            ax_time.set_xlim(app.xlim)
        else:
            xmin, xmax = min(all_xs), max(all_xs)
            span_x = max(xmax - xmin, 2.0)
            margin_x = max(span_x * 0.15, 1.0)
            ax_time.set_xlim(xmin - margin_x, xmax + margin_x)

        if app.ylim:
            ax_time.set_ylim(app.ylim)
        else:
            ymin, ymax = min(all_ys + [0.0]), max(all_ys + [0.0])
            span_y = max(ymax - ymin, 2.0)
            margin_y = max(span_y * 0.2, 0.5)
            ax_time.set_ylim(ymin - margin_y, ymax + margin_y)
    else:
        ax_time.set_xlim(-5, 5)
        ax_time.set_ylim(-5, 5)
        ax_time.text(
            0,
            0,
            "No points yet — type e.g. '3,4' or '0,1 1,2 2,1' in the terminal!",
            ha="center",
            va="center",
            fontsize=11,
            color="#64748b",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1"),
        )

    # Render FFT Magnitude & Phase if enabled
    if app.show_fft and ax_mag is not None and ax_phase is not None:
        xs, ys = app.active.get_sorted_xy()
        ax_mag.grid(True, linestyle="--", alpha=0.45)
        ax_phase.grid(True, linestyle="--", alpha=0.45)
        ax_mag.set_title(f"Frequency Domain (DFT/FFT of '{app.active_name}')", fontsize=10, weight="bold")
        ax_mag.set_ylabel("|X(f)|")
        ax_phase.set_ylabel("Phase (rad)")
        ax_phase.set_xlabel("Normalized Frequency (cycles/sample) or Hz")

        if len(ys) >= 2:
            n_fft = max(len(ys), 64)
            dts = np.diff(xs)
            dt = float(np.median(dts)) if len(dts) > 0 and np.median(dts) > 0 else 1.0
            spectrum = np.fft.fftshift(np.fft.fft(ys, n=n_fft))
            freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=dt))
            mag = np.abs(spectrum)
            # Zero out tiny magnitude noise before computing phase
            phase = np.where(mag > 1e-6 * np.max(mag), np.angle(spectrum), 0.0)

            ax_mag.plot(freqs, mag, color=app.active.color, linewidth=1.8)
            ax_mag.fill_between(freqs, 0, mag, color=app.active.color, alpha=0.15)
            ax_phase.plot(freqs, phase, color="#9333ea", linewidth=1.4)
            ax_phase.set_ylim(-np.pi - 0.3, np.pi + 0.3)

    if app.save_request:
        try:
            fig.savefig(app.save_request, dpi=150, bbox_inches="tight")
            print(f"  [*] Saved plot figure to '{app.save_request}'")
        except Exception as exc:
            print(f"  [!] Failed to save figure: {exc}")
        app.save_request = None

    fig.canvas.draw_idle()


def _print_prompt(app: SignalGrapherApp) -> None:
    sys.stdout.write(f"[{app.active_name} ({app.active.mode})] > ")
    sys.stdout.flush()


def main() -> None:
    app = SignalGrapherApp()

    # Process any initial CLI arguments as commands (e.g. `python3 signal_grapher.py "0,1 1,2 2,1"`)
    for arg in sys.argv[1:]:
        app.handle_command(arg)

    # Set up non-blocking interactive Matplotlib window on the main thread
    plt.ion()
    plt.rcParams["figure.facecolor"] = "#ffffff"
    plt.rcParams["axes.facecolor"] = "#f8fafc"
    fig = plt.figure(figsize=(9, 6), num="Signal Grapher — Transforms & Signals")

    # Handle user closing the GUI window directly
    def on_close(_event: object) -> None:
        app.running = False

    fig.canvas.mpl_connect("close_event", on_close)

    # Initial draw and show without blocking
    render_plot(fig, app)
    app.dirty = False
    plt.show(block=False)
    fig.canvas.flush_events()

    print("\n=================================================================")
    print("  Real-Time Signal Grapher (Transforms & Signals)")
    print("  Type '3,4' (or '0,1 1,2 2,1') to plot points in real time!")
    print("  Type 'help' for operations (shift, reverse, conv, fft, expr...)")
    print("=================================================================\n")
    _print_prompt(app)

    # Main non-blocking event loop: pumps Matplotlib GUI events and reads stdin lines
    try:
        while app.running:
            if not plt.fignum_exists(fig.number):
                break

            fig.canvas.flush_events()

            ready, _, _ = select.select([sys.stdin], [], [], 0.03)
            if ready:
                line = sys.stdin.readline()
                if not line:  # EOF (Ctrl-D)
                    print("\nClosing Signal Grapher...")
                    break

                app.handle_command(line)

                if app.dirty and app.running:
                    render_plot(fig, app)
                    fig.canvas.flush_events()
                    app.dirty = False

                if app.running:
                    _print_prompt(app)
    except KeyboardInterrupt:
        print("\nClosing Signal Grapher...")
    finally:
        app.running = False
        plt.close("all")


if __name__ == "__main__":
    main()
