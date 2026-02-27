#!/usr/bin/env python3
"""Browser-based MAIA LCIO viewer using Streamlit + Plotly."""

from __future__ import annotations

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


def import_pylcio() -> Any:
    try:
        return __import__("pylcio")
    except Exception as exc:
        raise RuntimeError(
            "Could not import pylcio. Run this in your MuonCollider container environment."
        ) from exc


def get_event_count_and_collections(path: str) -> tuple[int, list[str]]:
    pylcio = import_pylcio()
    reader = pylcio.IOIMPL.LCFactory.getInstance().createLCReader()
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
    pylcio = import_pylcio()
    reader = pylcio.IOIMPL.LCFactory.getInstance().createLCReader()
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


def build_figure(
    event: Any,
    selected_collections: list[str],
    min_energy: float,
    max_points_per_collection: int,
    point_size: float,
    show_tracks: bool,
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

    fig.update_layout(
        scene={
            "xaxis_title": "x [mm]",
            "yaxis_title": "y [mm]",
            "zaxis_title": "z [mm]",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )

    return fig, summaries


def main() -> None:
    st.set_page_config(page_title="MAIA LCIO Browser Viewer", layout="wide")
    st.title("MAIA LCIO Browser Viewer")
    st.caption("Interactive event inspection from .slcio files using pylcio")

    path = st.sidebar.text_input("LCIO file path", value="")
    st.sidebar.markdown("Example: `/data/.../neutronGun_E_0_50_reco_0.slcio`")

    if not path:
        st.info("Enter a .slcio path in the sidebar.")
        return

    try:
        n_events, default_collections = get_event_count_and_collections(path)
    except Exception as exc:
        st.error(f"Failed to read file: {exc}")
        return

    if n_events == 0:
        st.warning("No events found in this file.")
        return

    event_index = st.sidebar.slider("Event", min_value=0, max_value=n_events - 1, value=0)
    min_energy = st.sidebar.number_input("Min energy [GeV]", value=0.0, step=0.01)
    max_points = st.sidebar.number_input("Max points / collection", value=10000, min_value=100, step=100)
    point_size = st.sidebar.number_input("Base point size", value=3.0, min_value=1.0, max_value=20.0, step=0.5)
    show_tracks = st.sidebar.checkbox("Show track/MC lines", value=True)

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
    )

    st.plotly_chart(fig, use_container_width=True)

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
