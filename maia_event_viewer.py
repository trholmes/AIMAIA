#!/usr/bin/env python3
"""Interactive 3D event viewer for MAIA LCIO (.slcio) files."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any

TK_AVAILABLE = True
TK_IMPORT_ERROR: Exception | None = None
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as exc:
    TK_AVAILABLE = False
    TK_IMPORT_ERROR = exc

    class _TkStub:
        class Tk:
            pass

    tk = _TkStub()  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]

# Keep matplotlib state writable inside restricted environments.
if not os.path.isdir(os.environ.get("MPLCONFIGDIR", "")) or not os.access(
    os.environ.get("MPLCONFIGDIR", ""), os.W_OK
):
    os.environ["MPLCONFIGDIR"] = "/tmp"

import matplotlib
if TK_AVAILABLE:
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
if TK_AVAILABLE:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


@dataclass
class CollectionSummary:
    name: str
    n_points: int
    n_tracks: int
    energy_sum: float


class LcioReader:
    """Minimal adapter for LCIO readers (pylcio/pyLCIO variants)."""

    def __init__(self, path: str):
        self.path = path
        self._lib_name = ""
        self._lib = self._import_lcio()
        self._reader = self._create_reader()
        self._events = []
        self._load_events()

    def _import_lcio(self) -> Any:
        last_exc: Exception | None = None
        for name in ("pylcio", "pyLCIO"):
            try:
                module = __import__(name)
                self._lib_name = name
                return module
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(
            "Could not import LCIO Python bindings (tried pylcio, pyLCIO)."
        ) from last_exc

    def _create_reader(self) -> Any:
        # Common binding layout: IOIMPL.LCFactory.getInstance().createLCReader()
        try:
            factory = self._lib.IOIMPL.LCFactory.getInstance()
            return factory.createLCReader()
        except Exception:
            pass

        # Some pylcio builds expose LCFactory at top level.
        try:
            factory = self._lib.LCFactory.getInstance()
            return factory.createLCReader()
        except Exception:
            pass

        ioimpl = getattr(self._lib, "IOIMPL", None)
        if ioimpl is not None:
            lc_factory = getattr(ioimpl, "LCFactory", None)
            if lc_factory is not None:
                return lc_factory.getInstance().createLCReader()

        available = [k for k in dir(self._lib) if not k.startswith("_")]
        raise RuntimeError(
            f"{self._lib_name} imported, but LCReader creation failed. "
            f"Top-level symbols include: {available[:20]}"
        )

    def _load_events(self) -> None:
        self._reader.open(self.path)
        self._events = [evt for evt in self._reader]
        self._reader.close()

    @property
    def n_events(self) -> int:
        return len(self._events)

    def get_event(self, index: int) -> Any:
        return self._events[index]

    def get_collection_names(self, event_index: int) -> list[str]:
        event = self.get_event(event_index)
        try:
            return [str(name) for name in event.getCollectionNames()]
        except Exception:
            return []


class MaiaEventViewer(tk.Tk):
    def __init__(self, initial_file: str | None = None):
        super().__init__()
        self.title("MAIA LCIO 3D Event Viewer")
        self.geometry("1400x900")

        self.reader: LcioReader | None = None
        self.current_event = 0

        self.file_var = tk.StringVar(value=initial_file or "")
        self.event_var = tk.IntVar(value=0)
        self.max_points_var = tk.IntVar(value=8000)
        self.energy_min_var = tk.DoubleVar(value=0.0)
        self.point_size_var = tk.DoubleVar(value=10.0)
        self.show_tracks_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Load an .slcio file to begin.")

        self._build_ui()

        if initial_file:
            self.load_file(initial_file)

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=10)
        controls.grid(row=0, column=0, sticky="nsew")
        controls.columnconfigure(0, weight=1)

        plot_frame = ttk.Frame(self, padding=(0, 10, 10, 10))
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        file_row = ttk.Frame(controls)
        file_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        file_row.columnconfigure(0, weight=1)
        ttk.Entry(file_row, textvariable=self.file_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(file_row, text="Browse", command=self.browse_file).grid(row=0, column=1, padx=4)
        ttk.Button(file_row, text="Load", command=lambda: self.load_file(self.file_var.get())).grid(
            row=0, column=2
        )

        ev_row = ttk.Frame(controls)
        ev_row.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ev_row.columnconfigure(1, weight=1)
        ttk.Label(ev_row, text="Event").grid(row=0, column=0, sticky="w")
        self.event_scale = ttk.Scale(
            ev_row,
            from_=0,
            to=0,
            variable=self.event_var,
            orient="horizontal",
            command=self._on_event_slide,
        )
        self.event_scale.grid(row=0, column=1, sticky="ew", padx=6)
        self.event_spin = ttk.Spinbox(
            ev_row,
            from_=0,
            to=0,
            textvariable=self.event_var,
            width=6,
            command=self.on_event_change,
        )
        self.event_spin.grid(row=0, column=2, sticky="e")

        opts = ttk.LabelFrame(controls, text="Display Options", padding=8)
        opts.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        opts.columnconfigure(1, weight=1)

        ttk.Label(opts, text="Min energy [GeV]").grid(row=0, column=0, sticky="w")
        ttk.Entry(opts, textvariable=self.energy_min_var, width=10).grid(row=0, column=1, sticky="ew")

        ttk.Label(opts, text="Max points / collection").grid(row=1, column=0, sticky="w")
        ttk.Entry(opts, textvariable=self.max_points_var, width=10).grid(row=1, column=1, sticky="ew")

        ttk.Label(opts, text="Point size").grid(row=2, column=0, sticky="w")
        ttk.Entry(opts, textvariable=self.point_size_var, width=10).grid(row=2, column=1, sticky="ew")

        ttk.Checkbutton(opts, text="Draw MC/track lines", variable=self.show_tracks_var).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        cols_frame = ttk.LabelFrame(controls, text="Collections", padding=8)
        cols_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 8))
        controls.rowconfigure(3, weight=1)
        cols_frame.rowconfigure(0, weight=1)
        cols_frame.columnconfigure(0, weight=1)

        self.collections_list = tk.Listbox(cols_frame, selectmode=tk.EXTENDED, exportselection=False, height=20)
        self.collections_list.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(cols_frame, orient="vertical", command=self.collections_list.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.collections_list.configure(yscrollcommand=scroll.set)

        btns = ttk.Frame(cols_frame)
        btns.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        btns.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(btns, text="Select all", command=self.select_all_collections).grid(row=0, column=0, sticky="ew")
        ttk.Button(btns, text="Clear", command=self.clear_selection).grid(row=0, column=1, padx=4, sticky="ew")
        ttk.Button(btns, text="Refresh", command=self.refresh_plot).grid(row=0, column=2, sticky="ew")

        ttk.Label(controls, textvariable=self.status_var, wraplength=360, foreground="#2c3e50").grid(
            row=4, column=0, sticky="ew"
        )

        self.figure = plt.Figure(figsize=(9, 7), dpi=100)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.collections_list.bind("<<ListboxSelect>>", lambda _e: self.refresh_plot())
        self.event_spin.bind("<Return>", lambda _e: self.on_event_change())

    def browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select LCIO file",
            filetypes=[("LCIO files", "*.slcio"), ("All files", "*.*")],
        )
        if path:
            self.file_var.set(path)
            self.load_file(path)

    def load_file(self, path: str) -> None:
        if not path:
            return
        if not os.path.exists(path):
            messagebox.showerror("File not found", f"Cannot find file:\n{path}")
            return

        try:
            self.status_var.set("Loading file (this can take a moment)...")
            self.update_idletasks()
            self.reader = LcioReader(path)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            self.status_var.set("Failed to load file.")
            return

        n_events = self.reader.n_events
        if n_events == 0:
            self.status_var.set("Loaded file but found zero events.")
            return

        self.current_event = 0
        self.event_var.set(0)
        self.event_scale.configure(to=max(0, n_events - 1))
        self.event_spin.configure(to=max(0, n_events - 1))

        self.populate_collections()
        self.refresh_plot()
        self.status_var.set(f"Loaded {os.path.basename(path)} with {n_events} events.")

    def populate_collections(self) -> None:
        self.collections_list.delete(0, tk.END)
        if not self.reader:
            return
        names = self.reader.get_collection_names(self.current_event)
        for name in names:
            self.collections_list.insert(tk.END, name)
        self.select_all_collections()

    def select_all_collections(self) -> None:
        self.collections_list.selection_set(0, tk.END)
        self.refresh_plot()

    def clear_selection(self) -> None:
        self.collections_list.selection_clear(0, tk.END)
        self.refresh_plot()

    def _on_event_slide(self, _value: str) -> None:
        self.on_event_change()

    def on_event_change(self) -> None:
        if not self.reader:
            return
        try:
            idx = int(self.event_var.get())
        except Exception:
            return
        idx = max(0, min(idx, self.reader.n_events - 1))
        if idx != self.current_event:
            self.current_event = idx
            self.event_var.set(idx)
            self.populate_collections()
        self.refresh_plot()

    def _selected_collections(self) -> list[str]:
        return [self.collections_list.get(i) for i in self.collections_list.curselection()]

    def refresh_plot(self) -> None:
        self.ax.clear()
        self.ax.set_title(f"MAIA Event {self.current_event}")
        self.ax.set_xlabel("x [mm]")
        self.ax.set_ylabel("y [mm]")
        self.ax.set_zlabel("z [mm]")

        if not self.reader:
            self.canvas.draw_idle()
            return

        try:
            event = self.reader.get_event(self.current_event)
        except Exception as exc:
            self.status_var.set(f"Could not read event: {exc}")
            self.canvas.draw_idle()
            return

        selected = self._selected_collections()
        if not selected:
            self.status_var.set("No collections selected.")
            self.canvas.draw_idle()
            return

        cmap = plt.get_cmap("tab20")
        max_points = max(1, int(self.max_points_var.get()))
        min_energy = float(self.energy_min_var.get())
        point_size = float(self.point_size_var.get())

        summaries: list[CollectionSummary] = []
        plotted_any = False

        for idx, coll_name in enumerate(selected):
            try:
                coll = event.getCollection(coll_name)
            except Exception:
                continue

            points = []
            energies = []
            tracks = []

            for obj in coll:
                pos = self._extract_position(obj)
                if pos is not None:
                    energy = self._extract_energy(obj)
                    if energy is None:
                        energy = 0.0
                    if energy >= min_energy:
                        points.append(pos)
                        energies.append(energy)

                if self.show_tracks_var.get():
                    line = self._extract_track_line(obj)
                    if line is not None:
                        tracks.append(line)

            if len(points) > max_points:
                stride = max(1, len(points) // max_points)
                points = points[::stride]
                energies = energies[::stride]

            color = cmap(idx % 20)

            if points:
                xs, ys, zs = zip(*points)
                sizes = [point_size + (3.0 * min(e, 5.0)) for e in energies]
                self.ax.scatter(xs, ys, zs, s=sizes, color=color, alpha=0.75, label=coll_name)
                plotted_any = True

            for (x0, y0, z0), (x1, y1, z1) in tracks[:5000]:
                self.ax.plot([x0, x1], [y0, y1], [z0, z1], color=color, alpha=0.4, linewidth=0.6)
                plotted_any = True

            summaries.append(
                CollectionSummary(
                    name=coll_name,
                    n_points=len(points),
                    n_tracks=len(tracks),
                    energy_sum=sum(energies),
                )
            )

        if plotted_any:
            self._set_equal_aspect()
            self.ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
            summary_text = " | ".join(
                f"{s.name}: N={s.n_points}, tracks={s.n_tracks}, E={s.energy_sum:.2f}"
                for s in summaries[:4]
            )
            if len(summaries) > 4:
                summary_text += " | ..."
            self.status_var.set(summary_text)
        else:
            self.status_var.set("Selected collections do not expose plottable positions/tracks.")

        self.canvas.draw_idle()

    @staticmethod
    def _try_call(obj: Any, method_name: str) -> Any:
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                return method()
            except Exception:
                return None
        return None

    def _extract_position(self, obj: Any) -> tuple[float, float, float] | None:
        for method_name in ("getPosition", "getVertex", "getReferencePoint"):
            v = self._try_call(obj, method_name)
            if v is not None:
                try:
                    return (float(v[0]), float(v[1]), float(v[2]))
                except Exception:
                    continue
        return None

    def _extract_track_line(
        self, obj: Any
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        start = self._try_call(obj, "getVertex")
        end = self._try_call(obj, "getEndpoint")
        if start is None or end is None:
            return None
        try:
            p0 = (float(start[0]), float(start[1]), float(start[2]))
            p1 = (float(end[0]), float(end[1]), float(end[2]))
            return p0, p1
        except Exception:
            return None

    def _extract_energy(self, obj: Any) -> float | None:
        for method_name in ("getEnergy", "getEDep", "getAmplitude", "getTime"):
            value = self._try_call(obj, method_name)
            if value is None:
                continue
            try:
                return float(value)
            except Exception:
                continue
        return None

    def _set_equal_aspect(self) -> None:
        xlim = self.ax.get_xlim3d()
        ylim = self.ax.get_ylim3d()
        zlim = self.ax.get_zlim3d()

        xmid = 0.5 * (xlim[0] + xlim[1])
        ymid = 0.5 * (ylim[0] + ylim[1])
        zmid = 0.5 * (zlim[0] + zlim[1])

        radius = max(
            abs(xlim[1] - xlim[0]),
            abs(ylim[1] - ylim[0]),
            abs(zlim[1] - zlim[0]),
            1.0,
        ) / 2.0

        self.ax.set_xlim3d([xmid - radius, xmid + radius])
        self.ax.set_ylim3d([ymid - radius, ymid + radius])
        self.ax.set_zlim3d([zmid - radius, zmid + radius])


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAIA LCIO 3D event viewer")
    parser.add_argument("slcio", nargs="?", help="Path to input .slcio file")
    parser.add_argument(
        "--summary-event",
        type=int,
        default=None,
        help="Headless mode: print collection summary for one event index.",
    )
    return parser.parse_args(argv)


def print_event_summary(path: str, event_index: int) -> int:
    reader = LcioReader(path)
    if reader.n_events == 0:
        print("No events found in file.")
        return 1
    idx = max(0, min(event_index, reader.n_events - 1))
    event = reader.get_event(idx)
    names = reader.get_collection_names(idx)
    print(f"File: {path}")
    print(f"Events: {reader.n_events}")
    print(f"Event: {idx}")
    for name in names:
        try:
            coll = event.getCollection(name)
            n = sum(1 for _ in coll)
        except Exception:
            n = -1
        print(f"- {name}: {n if n >= 0 else 'N/A'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.summary_event is not None:
        if not args.slcio:
            print("Provide a .slcio path when using --summary-event.")
            return 2
        return print_event_summary(args.slcio, args.summary_event)

    if not TK_AVAILABLE:
        print("Tkinter is not available in this Python environment.")
        print(
            "Run with --summary-event for headless inspection, or use a Python build with tkinter for the GUI."
        )
        if TK_IMPORT_ERROR:
            print(f"Import error: {TK_IMPORT_ERROR}")
        return 2

    app = MaiaEventViewer(initial_file=args.slcio)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
