#!/usr/bin/env python3
"""Browser-based MAIA LCIO viewer using Streamlit + Plotly."""

from __future__ import annotations

import glob
import math
import os
from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go
import streamlit as st


@dataclass
class CollectionSummary:
    name: str
    n_points: int
    n_tracks: int
    energy_sum: float


DETECTOR_COMPONENTS = [
    # Dimensions from provided table, converted from cm to mm.
    {
        "name": "Vertex",
        "color": "#9aa0a6",
        "barrel": {"rmin": 30.0, "rmax": 104.0, "zmax": 650.0},
        "endcap": {"rmin": 25.0, "rmax": 112.0, "zmin": 80.0, "zmax": 282.0},
    },
    {
        "name": "InnerTracker",
        "color": "#7e57c2",
        "barrel": {"rmin": 127.0, "rmax": 554.0, "zmax": 692.0},
        "endcap": {"rmin": 405.0, "rmax": 555.0, "zmin": 524.0, "zmax": 2190.0},
    },
    {
        "name": "OuterTracker",
        "color": "#5c6bc0",
        "barrel": {"rmin": 819.0, "rmax": 1486.0, "zmax": 1249.0},
        "endcap": {"rmin": 618.0, "rmax": 1430.0, "zmin": 1310.0, "zmax": 2190.0},
    },
    {
        "name": "Solenoid",
        "color": "#26a69a",
        "barrel": {"rmin": 1500.0, "rmax": 1857.0, "zmax": 2307.0},
        "endcap": None,
    },
    {
        "name": "ECAL",
        "color": "#66bb6a",
        "barrel": {"rmin": 1857.0, "rmax": 2125.0, "zmax": 2307.0},
        "endcap": {"rmin": 310.0, "rmax": 2125.0, "zmin": 2307.0, "zmax": 2575.0},
    },
    {
        "name": "HCAL",
        "color": "#ffa726",
        "barrel": {"rmin": 2125.0, "rmax": 4113.0, "zmax": 2575.0},
        "endcap": {"rmin": 307.0, "rmax": 4113.0, "zmin": 2575.0, "zmax": 4562.0},
    },
    {
        "name": "Muon",
        "color": "#ef5350",
        "barrel": {"rmin": 4150.0, "rmax": 7150.0, "zmax": 4565.0},
        "endcap": {"rmin": 446.0, "rmax": 7150.0, "zmin": 4565.0, "zmax": 6025.0},
    },
]


def import_lcio_module() -> Any:
    last_exc: Exception | None = None
    for name in ("pyLCIO", "pylcio"):
        try:
            return __import__(name)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(
        "Could not import LCIO Python bindings (tried pylcio, pyLCIO)."
    ) from last_exc


def create_lc_reader(module: Any) -> Any:
    # Common binding layout.
    try:
        return module.IOIMPL.LCFactory.getInstance().createLCReader()
    except Exception:
        pass

    # Some pylcio builds expose namespaces slightly differently.
    try:
        return module.LCFactory.getInstance().createLCReader()
    except Exception:
        pass

    ioimpl = getattr(module, "IOIMPL", None)
    if ioimpl is not None:
        lc_factory = getattr(ioimpl, "LCFactory", None)
        if lc_factory is not None:
            return lc_factory.getInstance().createLCReader()

    available = [k for k in dir(module) if not k.startswith("_")]
    raise RuntimeError(
        "Could not find LCReader factory in LCIO module. "
        f"Available top-level symbols include: {available[:20]}"
    )


@st.cache_data(show_spinner=False)
def resolve_input_paths(path_expr: str) -> list[str]:
    wildcard_chars = ("*", "?", "[")
    if any(ch in path_expr for ch in wildcard_chars):
        return sorted(glob.glob(path_expr))
    return [path_expr] if os.path.exists(path_expr) else []


@st.cache_data(show_spinner=False)
def get_event_count_and_collections(path: str) -> tuple[int, list[str]]:
    lcio = import_lcio_module()
    reader = create_lc_reader(lcio)
    reader.open(path)

    n_events = 0
    first_event_collections: list[str] = []
    for event in reader:
        if n_events == 0:
            try:
                first_event_collections = [str(n) for n in event.getCollectionNames()]
            except Exception:
                first_event_collections = []
        n_events += 1

    reader.close()
    return n_events, first_event_collections


def get_event(path: str, event_index: int) -> Any:
    lcio = import_lcio_module()
    reader = create_lc_reader(lcio)
    reader.open(path)

    current = 0
    selected = None
    for event in reader:
        if current == event_index:
            selected = event
            break
        current += 1

    if selected is None:
        reader.close()
        raise IndexError(f"Event index {event_index} out of range")

    return selected, reader


def try_call(obj: Any, method_name: str) -> Any:
    method = getattr(obj, method_name, None)
    if callable(method):
        try:
            return method()
        except Exception:
            return None
    return None


def extract_position(obj: Any) -> tuple[float, float, float] | None:
    for method_name in ("getPosition", "getVertex", "getReferencePoint"):
        v = try_call(obj, method_name)
        if v is not None:
            try:
                return float(v[0]), float(v[1]), float(v[2])
            except Exception:
                continue
    return None


def extract_energy(obj: Any) -> float:
    for method_name in ("getEnergy", "getEDep", "getAmplitude"):
        value = try_call(obj, method_name)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return 0.0


def extract_track_line(
    obj: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    start = try_call(obj, "getVertex")
    end = try_call(obj, "getEndpoint")
    if start is None or end is None:
        return None
    try:
        p0 = (float(start[0]), float(start[1]), float(start[2]))
        p1 = (float(end[0]), float(end[1]), float(end[2]))
        return p0, p1
    except Exception:
        return None


def _circle_xyz(radius: float, z: float, n: int = 72) -> tuple[list[float], list[float], list[float]]:
    xs = []
    ys = []
    zs = []
    for i in range(n + 1):
        phi = (2.0 * math.pi * i) / n
        xs.append(radius * math.cos(phi))
        ys.append(radius * math.sin(phi))
        zs.append(z)
    return xs, ys, zs


def add_detector_wireframe(fig: go.Figure) -> None:
    for comp in DETECTOR_COMPONENTS:
        name = comp["name"]
        color = comp["color"]
        show_legend_for_component = True

        barrel = comp.get("barrel")
        if barrel:
            for r in (barrel["rmin"], barrel["rmax"]):
                for z in (-barrel["zmax"], barrel["zmax"]):
                    xs, ys, zs = _circle_xyz(r, z)
                    fig.add_trace(
                        go.Scatter3d(
                            x=xs,
                            y=ys,
                            z=zs,
                            mode="lines",
                            line={"color": color, "width": 2},
                            opacity=0.35,
                            name=name,
                            legendgroup=name,
                            showlegend=show_legend_for_component,
                            hoverinfo="skip",
                        )
                    )
                    show_legend_for_component = False

            for phi in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0):
                for r in (barrel["rmin"], barrel["rmax"]):
                    x = r * math.cos(phi)
                    y = r * math.sin(phi)
                    fig.add_trace(
                        go.Scatter3d(
                            x=[x, x],
                            y=[y, y],
                            z=[-barrel["zmax"], barrel["zmax"]],
                            mode="lines",
                            line={"color": color, "width": 1},
                            opacity=0.25,
                            legendgroup=name,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

        endcap = comp.get("endcap")
        if endcap:
            for z in (-endcap["zmin"], -endcap["zmax"], endcap["zmin"], endcap["zmax"]):
                for r in (endcap["rmin"], endcap["rmax"]):
                    xs, ys, zs = _circle_xyz(r, z)
                    fig.add_trace(
                        go.Scatter3d(
                            x=xs,
                            y=ys,
                            z=zs,
                            mode="lines",
                            line={"color": color, "width": 1},
                            opacity=0.22,
                            legendgroup=name,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

            for z in (-endcap["zmin"], -endcap["zmax"], endcap["zmin"], endcap["zmax"]):
                for phi in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0):
                    x0 = endcap["rmin"] * math.cos(phi)
                    y0 = endcap["rmin"] * math.sin(phi)
                    x1 = endcap["rmax"] * math.cos(phi)
                    y1 = endcap["rmax"] * math.sin(phi)
                    fig.add_trace(
                        go.Scatter3d(
                            x=[x0, x1],
                            y=[y0, y1],
                            z=[z, z],
                            mode="lines",
                            line={"color": color, "width": 1},
                            opacity=0.2,
                            legendgroup=name,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )


def build_figure(
    event: Any,
    selected_collections: list[str],
    min_energy: float,
    max_points_per_collection: int,
    point_size: float,
    show_tracks: bool,
    show_detector: bool,
    view_revision: int,
) -> tuple[go.Figure, list[CollectionSummary]]:
    fig = go.Figure()
    summaries: list[CollectionSummary] = []
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for idx, coll_name in enumerate(selected_collections):
        try:
            coll = event.getCollection(coll_name)
        except Exception:
            continue

        points: list[tuple[float, float, float]] = []
        energies: list[float] = []
        tracks: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []

        for obj in coll:
            pos = extract_position(obj)
            if pos is not None:
                energy = extract_energy(obj)
                if energy >= min_energy:
                    points.append(pos)
                    energies.append(energy)

            if show_tracks:
                line = extract_track_line(obj)
                if line is not None:
                    tracks.append(line)

        if len(points) > max_points_per_collection:
            stride = max(1, len(points) // max_points_per_collection)
            points = points[::stride]
            energies = energies[::stride]

        color = palette[idx % len(palette)]

        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            zs = [p[2] for p in points]
            sizes = [point_size + min(8.0, e * 2.0) for e in energies]

            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="markers",
                    marker={"size": sizes, "color": color, "opacity": 0.75},
                    name=coll_name,
                )
            )

        if show_tracks and tracks:
            for line in tracks[:2000]:
                (x0, y0, z0), (x1, y1, z1) = line
                fig.add_trace(
                    go.Scatter3d(
                        x=[x0, x1],
                        y=[y0, y1],
                        z=[z0, z1],
                        mode="lines",
                        line={"color": color, "width": 2},
                        opacity=0.35,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        summaries.append(
            CollectionSummary(
                name=coll_name,
                n_points=len(points),
                n_tracks=len(tracks),
                energy_sum=sum(energies),
            )
        )

    if show_detector:
        add_detector_wireframe(fig)

    fig.update_layout(
        uirevision=f"maia-view-{view_revision}",
        scene={
            "xaxis_title": "x [mm]",
            "yaxis_title": "y [mm]",
            "zaxis_title": "z [mm]",
            "aspectmode": "data",
            "uirevision": f"maia-scene-{view_revision}",
            # Initial view: z-axis runs left-right on screen.
            "camera": {
                "eye": {"x": 2.2, "y": 0.0, "z": 0.0},
                "up": {"x": 0.0, "y": 1.0, "z": 0.0},
            },
        },
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )

    return fig, summaries


def main() -> None:
    st.set_page_config(page_title="MAIA LCIO Browser Viewer", layout="wide")
    st.title("MAIA LCIO Browser Viewer")
    st.caption("Interactive event inspection from .slcio files using pylcio")

    example_path = (
        "/data/fmeloni/DataMuC_MAIA_v0/v8/recoBIB/neutronGun_E_0_50/"
        "neutronGun_E_0_50_reco_0.slcio"
    )
    example_glob = (
        "/data/fmeloni/DataMuC_MAIA_v0/v8/recoBIB/neutronGun_E_0_50/"
        "neutronGun_E_0_50_reco_*.slcio"
    )
    path_expr = st.sidebar.text_input("LCIO path or glob", value=example_glob)
    st.sidebar.markdown(f"Example file: `{example_path}`")
    st.sidebar.markdown(f"Example glob: `{example_glob}`")

    if not path_expr:
        st.info("Enter an LCIO file path or glob pattern in the sidebar.")
        return

    matched_paths = resolve_input_paths(path_expr)
    if not matched_paths:
        st.error("No files matched this path/pattern.")
        return

    if len(matched_paths) == 1:
        path = matched_paths[0]
        st.sidebar.info("Matched 1 file.")
    else:
        file_index = st.sidebar.number_input(
            "Matched file index",
            min_value=0,
            max_value=len(matched_paths) - 1,
            value=0,
            step=1,
        )
        path = matched_paths[int(file_index)]
        st.sidebar.info(
            f"Matched {len(matched_paths)} files. Loading index {int(file_index)} only."
        )
    st.sidebar.caption(f"Selected file: {path}")

    try:
        n_events, default_collections = get_event_count_and_collections(path)
    except Exception as exc:
        st.error(f"Failed to read file: {exc}")
        return

    if n_events == 0:
        st.warning("No events found in this file.")
        return

    if n_events == 1:
        event_index = 0
        st.sidebar.info("This file contains 1 event. Showing event 0.")
    else:
        event_index = st.sidebar.slider("Event", min_value=0, max_value=n_events - 1, value=0)
    min_energy = st.sidebar.number_input("Min energy [GeV]", value=0.0, step=0.01)
    max_points = st.sidebar.number_input("Max points / collection", value=10000, min_value=100, step=100)
    point_size = st.sidebar.number_input("Base point size", value=3.0, min_value=1.0, max_value=20.0, step=0.5)
    show_tracks = st.sidebar.checkbox("Show track/MC lines", value=True)
    show_detector = st.sidebar.checkbox("Show detector wireframe", value=True)
    if "plot_reset_nonce" not in st.session_state:
        st.session_state.plot_reset_nonce = 0
    if "view_revision" not in st.session_state:
        st.session_state.view_revision = 0
    if st.sidebar.button("Reset 3D view"):
        st.session_state.plot_reset_nonce += 1
        st.session_state.view_revision += 1

    try:
        event, reader = get_event(path, event_index)
    except Exception as exc:
        st.error(f"Could not load event {event_index}: {exc}")
        return

    try:
        event_collections = [str(n) for n in event.getCollectionNames()]
    except Exception:
        event_collections = default_collections

    selected = st.sidebar.multiselect(
        "Collections",
        options=event_collections,
        default=event_collections[: min(8, len(event_collections))],
    )

    if not selected:
        st.warning("Select at least one collection.")
        reader.close()
        return

    fig, summaries = build_figure(
        event=event,
        selected_collections=selected,
        min_energy=float(min_energy),
        max_points_per_collection=int(max_points),
        point_size=float(point_size),
        show_tracks=show_tracks,
        show_detector=show_detector,
        view_revision=int(st.session_state.view_revision),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"maia_plot_{st.session_state.plot_reset_nonce}",
    )

    st.subheader("Collection Summary")
    st.dataframe(
        [
            {
                "collection": s.name,
                "points": s.n_points,
                "tracks": s.n_tracks,
                "energy_sum_GeV": round(s.energy_sum, 4),
            }
            for s in summaries
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(f"File events: {n_events} | Showing event: {event_index}")
    reader.close()


if __name__ == "__main__":
    main()
