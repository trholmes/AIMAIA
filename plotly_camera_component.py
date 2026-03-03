from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).resolve().parent / "plotly_camera_component" / "frontend" / "build"
_PLOTLY_CAMERA_COMPONENT = components.declare_component(
    "plotly_camera_component",
    path=str(_COMPONENT_DIR),
)


def plotly_camera_chart(
    fig: go.Figure,
    *,
    key: str,
    height: int = 700,
    camera: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Render a Plotly figure and return current scene camera from client relayout events."""
    fig_dict = fig.to_plotly_json()
    if camera is not None:
        layout = fig_dict.setdefault("layout", {})
        scene = layout.setdefault("scene", {})
        scene["camera"] = camera

    value = _PLOTLY_CAMERA_COMPONENT(
        fig=fig_dict,
        height=height,
        key=key,
        default=camera,
    )
    return value if isinstance(value, dict) else camera
