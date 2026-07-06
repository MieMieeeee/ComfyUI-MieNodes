"""Tiny VRAM pinner GUI.

Pins a chosen amount of GPU memory via torch.zeros (memset) so other processes
(e.g. ComfyUI) see a smaller free pool.  All visible numbers come
from nvidia-smi; the progress bar shows our own pinned amount.

Run:    python scripts/vram_pin.py

Requires: torch (CUDA build), tkinter (stdlib), nvidia-smi on PATH.
"""

import subprocess
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import torch


POLL_MS = 1500


def fmt_gb(n_bytes):
    return f"{n_bytes / 1024 ** 3:.2f} GB"


def query_driver():
    """(used, free, total) in bytes from nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            timeout=2,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        total_mb, used_mb, free_mb = (int(x) for x in out.split(","))
        mb = 1024 ** 2
        return used_mb * mb, free_mb * mb, total_mb * mb
    except Exception:
        return None


class VramPinApp:
    def __init__(self, root):
        self.root = root
        root.title("VRAM Pin")
        root.geometry("440x320")
        root.resizable(False, False)

        self.pinned_tensor = None
        self.pinned_gb = 0.0
        self.total_gb = 0.0
        self._syncing = False  # avoid slider<->entry feedback loop

        self._build()

        try:
            self.gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            self.gpu_name = "CUDA device"

        driver = query_driver()
        if driver is None:
            self._status("nvidia-smi query failed", error=True)
            return
        _, _, total = driver
        self.total_gb = total / 1024 ** 3
        self.slider.configure(to=round(self.total_gb, 1))
        self.gpu_var.set(f"GPU: {self.gpu_name} ({self.total_gb:.1f} GB)")
        self._tick()

    def _build(self):
        pad = {"padx": 14, "pady": 3}
        mono = ("Consolas", 10)
        bold = ("Segoe UI", 10, "bold")
        sub_c = "#666"

        r = 0
        self.gpu_var = tk.StringVar(value="GPU: detecting...")
        ttk.Label(
            self.root, textvariable=self.gpu_var, font=bold
        ).grid(row=r, column=0, columnspan=2, sticky="w", padx=14, pady=(10, 4))
        r += 1
        ttk.Separator(self.root).grid(
            row=r, column=0, columnspan=2, sticky="ew", padx=14
        )
        r += 1

        self.total_var = tk.StringVar(value="--")
        ttk.Label(self.root, text="Total:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Label(self.root, textvariable=self.total_var, font=mono).grid(
            row=r, column=1, sticky="e", **pad
        )
        r += 1

        self.free_var = tk.StringVar(value="--")
        ttk.Label(self.root, text="Free:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Label(self.root, textvariable=self.free_var, font=mono).grid(
            row=r, column=1, sticky="e", **pad
        )
        r += 1

        self.bar_var = tk.DoubleVar(value=0)
        ttk.Progressbar(
            self.root, length=400, mode="determinate", variable=self.bar_var
        ).grid(row=r, column=0, columnspan=2, sticky="ew", **pad)
        r += 1

        ttk.Separator(self.root).grid(
            row=r, column=0, columnspan=2, sticky="ew", padx=14
        )
        r += 1

        self.pinned_var = tk.StringVar(value="0.00 GB")
        ttk.Label(self.root, text="Pinned:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Label(self.root, textvariable=self.pinned_var, font=mono).grid(
            row=r, column=1, sticky="e", **pad
        )
        r += 1

        ttk.Label(self.root, text="Pin amount:").grid(
            row=r, column=0, sticky="w", **pad
        )
        sf = ttk.Frame(self.root)
        sf.grid(row=r, column=1, sticky="e", **pad)
        self.slider_var = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(
            sf, from_=0, to=16, orient="horizontal",
            variable=self.slider_var, length=200, command=self._on_slide,
        )
        self.slider.pack(side="left")
        self.amount_var = tk.StringVar(value="0.0")
        self.amount_entry = ttk.Entry(
            sf, textvariable=self.amount_var, width=6, justify="right", font=mono
        )
        self.amount_entry.pack(side="left", padx=4)
        ttk.Label(sf, text="GB", font=mono).pack(side="left")
        self.amount_var.trace_add("write", self._on_amount)
        self.amount_entry.bind("<FocusOut>", self._on_amount_focus_out)
        self.amount_entry.bind("<Return>", lambda e: self.apply())
        r += 1

        btns = ttk.Frame(self.root)
        btns.grid(row=r, column=0, columnspan=2, **pad)
        ttk.Button(btns, text="Copy", command=self.copy_stats).pack(side="left", padx=6)
        ttk.Button(btns, text="Apply", command=self.apply).pack(side="left", padx=6)
        ttk.Button(btns, text="Release", command=self.release).pack(
            side="left", padx=6
        )
        r += 1

        self.status_var = tk.StringVar(value="idle")
        ttk.Label(
            self.root, textvariable=self.status_var, foreground=sub_c
        ).grid(row=r, column=0, columnspan=2, sticky="w", **pad)

        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)

    def _on_slide(self, _evt=None):
        if self._syncing:
            return
        v = round(self.slider_var.get(), 1)
        self._syncing = True
        self.amount_var.set(f"{v:.1f}")
        self._syncing = False

    def _on_amount(self, *_):
        if self._syncing:
            return
        try:
            v = round(float(self.amount_var.get()), 1)
        except ValueError:
            return
        v = max(0.0, min(self.total_gb, v))
        if abs(v - self.slider_var.get()) > 0.01:
            self._syncing = True
            self.slider_var.set(v)
            self._syncing = False

    def _on_amount_focus_out(self, _evt=None):
        try:
            v = round(float(self.amount_var.get()), 1)
        except ValueError:
            v = self.slider_var.get()
        v = max(0.0, min(self.total_gb, v))
        self._syncing = True
        self.amount_var.set(f"{v:.1f}")
        self.slider_var.set(v)
        self._syncing = False

    def apply(self):
        gb = round(self.slider_var.get(), 1)
        if gb < 0.1:
            self.release()
            return
        self.release()
        n = int(gb * 1024 ** 3 / 4)
        try:
            self.pinned_tensor = torch.zeros(
                n, dtype=torch.float32, device="cuda:0"
            )
            self.pinned_gb = gb
            self.pinned_var.set(f"{self.pinned_gb:.2f} GB")
            self._status(f"Holding {gb:.1f} GB")
        except Exception as e:
            self.pinned_tensor = None
            self.pinned_gb = 0.0
            self.pinned_var.set(f"{self.pinned_gb:.2f} GB")
            self._status(f"Pin failed: {e}", error=True)
            messagebox.showerror('VRAM pin failed', str(e))

    def copy_stats(self):
        driver = query_driver()
        if driver:
            _, df, total = driver
            total_str = fmt_gb(total)
            free_str = fmt_gb(df)
        else:
            total_str = self.total_var.get()
            free_str = self.free_var.get()
        text = (
            f"Total: {total_str}\n"
            f"Free: {free_str}\n"
            f"Pinned: {self.pinned_gb:.2f} GB"
        )
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        self._status("Copied to clipboard")

    def release(self):
        if self.pinned_tensor is not None:
            del self.pinned_tensor
            self.pinned_tensor = None
            torch.cuda.empty_cache()
        self.pinned_gb = 0.0
        self.pinned_var.set(f"{self.pinned_gb:.2f} GB")
        self._syncing = True
        self.slider_var.set(0)
        self.amount_var.set("0.0")
        self._syncing = False
        self._status("Released")

    def _status(self, msg, error=False):
        self.status_var.set(msg)

    def _refresh(self):
        driver = query_driver()
        if driver:
            du, df, total = driver
            self.total_var.set(fmt_gb(total))
            self.free_var.set(fmt_gb(df))
            self.bar_var.set(df / total * 100 if total else 0)
        self.pinned_var.set(f"{self.pinned_gb:.2f} GB")

    def _tick(self):
        self._refresh()
        self.root.after(POLL_MS, self._tick)

    def on_close(self):
        self.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = VramPinApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()