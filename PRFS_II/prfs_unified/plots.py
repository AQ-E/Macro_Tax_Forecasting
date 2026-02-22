"""
prfs_unified/plots.py
=====================
Reusable Plotly chart builders for the unified PRFS app.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def forecast_plot(
    hist: pd.Series,
    fore: pd.DataFrame,
    title: str,
    y_label: str = "PKR Million",
) -> go.Figure:
    """
    Build a standard forecast plot with historical line + forecast + CI bands.
    """
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist.index.to_timestamp(),
        y=hist.values,
        mode="lines+markers",
        name="Historical",
        line=dict(color="#2d3436"),
    ))

    x_f = fore.index.to_timestamp()

    # 95 % CI band
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_f, x_f[::-1]]),
        y=np.concatenate([fore["hi95"].values, fore["lo95"].values[::-1]]),
        fill="toself",
        fillcolor="rgba(9,132,227,0.08)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="95% CI",
    ))

    # 80 % CI band
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_f, x_f[::-1]]),
        y=np.concatenate([fore["hi80"].values, fore["lo80"].values[::-1]]),
        fill="toself",
        fillcolor="rgba(9,132,227,0.18)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="80% CI",
    ))

    # Point forecast
    fig.add_trace(go.Scatter(
        x=x_f,
        y=fore["yhat"],
        mode="lines+markers",
        name="Forecast",
        line=dict(dash="dash", color="#d63031"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=y_label,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def forecast_table(fore: pd.DataFrame, unit_label: str = "PKR Billion") -> pd.DataFrame:
    """Convert a forecast df (PKR Million) into a display table (PKR Billion)."""
    show = (fore / 1000.0).copy()
    show["80% interval"] = show.apply(lambda r: f"[{r.lo80:,.2f}, {r.hi80:,.2f}]", axis=1)
    show["95% interval"] = show.apply(lambda r: f"[{r.lo95:,.2f}, {r.hi95:,.2f}]", axis=1)
    return show[["yhat", "80% interval", "95% interval"]].rename(columns={"yhat": f"Forecast ({unit_label})"})
