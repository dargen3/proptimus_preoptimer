import html
import re
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
from plotly.subplots import make_subplots

from .data import atomic_types

COLOR = "#636EFA"
TARGET_COLOR = "red"
HISTOGRAM_BINS = 30
POLAR_BINS = 120


def _subplot_titles(res_name: str, at_name: str) -> tuple:
    """Titles stating which atoms the bond length, bond angle and dihedral angle are measured between."""
    a1 = at_name.split("/")[0]
    a2, a3, a4 = atomic_types[res_name][a1]["ancestors"]
    return (f"Bond Lengths ({a1}-{a2})",
            f"Bond Angles ({a1}-{a2}-{a3})",
            f"Dihedral Angles ({a1}-{a2}-{a3}-{a4})")


def _create_figure(res_name: str, at_name: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=3,
        shared_xaxes=True,
        row_heights=[0.95, 0.05],
        vertical_spacing=0.03,
        horizontal_spacing=0.06,
        subplot_titles=_subplot_titles(res_name, at_name),
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "polar", "rowspan": 2}],
            [{"type": "xy"}, {"type": "xy"}, None],
        ],
    )
    fig.update_layout(barmode="overlay")
    return fig


def _add_histogram_and_rug(fig: go.Figure, values: list, ids: list, targets: list,
                           col: int, title: str, unit: str) -> None:
    """Adds a histogram (top row), a rug plot (bottom row) and target lines to the given column."""
    min_val, max_val = min(values), max(values)

    # Avoid division by zero if all values are the same
    bin_size = (max_val - min_val) / HISTOGRAM_BINS if max_val > min_val else 1

    fig.add_trace(
        go.Histogram(
            x=values,
            autobinx=False,
            xbins=dict(
                start=min_val - 0.5 * bin_size,
                end=max_val + 0.5 * bin_size,
                size=bin_size,
            ),
            name=title,
            marker_color=COLOR,
            marker_line_width=0,  # Prevents narrow bins from looking like lines
            hovertemplate=f"{title}: %{{x}}{unit}<br>Count: %{{y}}<extra></extra>",
        ),
        row=1, col=col,
    )

    hover_texts = [f"ID: {i}<br>{title}: {v:.3f}{unit}" for i, v in zip(ids, values)]
    fig.add_trace(
        go.Scatter(
            x=values,
            y=[0] * len(values),
            mode="markers",
            marker=dict(symbol="line-ns-open", color=COLOR, size=20, line_width=1.5),
            text=hover_texts,
            hoverinfo="text",
            showlegend=False,
            name=f"{title} Rug",
        ),
        row=2, col=col,
    )

    for target in targets:
        for row in (1, 2):
            fig.add_vline(x=target, line_color=TARGET_COLOR, line_width=2, row=row, col=col)


def _add_polar_histogram(fig: go.Figure, angles: list, ids: list, targets: list) -> None:
    """Adds a polar histogram of dihedral angles with a rug and target lines."""
    counts, bin_edges = np.histogram(angles, bins=POLAR_BINS, range=(-180, 180))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = (bin_edges[-1] - bin_edges[0]) / POLAR_BINS

    fig.add_trace(
        go.Barpolar(
            r=counts,
            theta=bin_centers,
            width=bin_width,
            name="Count",
            marker_color=COLOR,
            marker_line_color="white",
            marker_line_width=0,
            opacity=0.8,
        ),
        row=1, col=3,
    )

    max_count = max(counts) if max(counts) > 0 else 1
    rug_start = max_count * 1.05
    rug_end = max_count * 1.15

    # Rug as a single line trace; None separates individual segments
    r_rug, theta_rug, hover_texts = [], [], []
    for angle, atom_id in zip(angles, ids):
        r_rug.extend([rug_start, rug_end, None])
        theta_rug.extend([angle, angle, None])
        hover_str = f"ID: {atom_id}<br>Angle: {angle:.2f}°"
        hover_texts.extend([hover_str, hover_str, None])

    fig.add_trace(
        go.Scatterpolar(
            r=r_rug,
            theta=theta_rug,
            mode="lines",
            line=dict(color=COLOR),
            showlegend=False,
            text=hover_texts,
            hoverinfo="text",
            name="Rug",
        ),
        row=1, col=3,
    )

    for target in targets:
        fig.add_trace(
            go.Scatterpolar(
                r=[0, rug_end],
                theta=[target, target],
                mode="lines",
                line=dict(color=TARGET_COLOR, width=2),
                showlegend=False,
                hoverinfo="text",
                text="Target",
                name="Target Angle",
            ),
            row=1, col=3,
        )


def _format_figure(fig: go.Figure, res_name: str, at_name: str, n_atoms: int) -> None:
    fig.update_layout(
        title_text=f"Residue: {res_name}; Atom type: {at_name} (n={n_atoms})",
        title_x=0.5,
        title_font_size=20,
        showlegend=False,
        polar=dict(
            radialaxis=dict(showline=False, ticks=""),
            angularaxis=dict(
                rotation=90,
                direction="clockwise",
                tickmode="array",
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=["0°", "45°", "90°", "135°", "±180°", "-135°", "-90°", "-45°"],
            ),
        ),
    )

    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Bond Length [Å]", row=2, col=1)
    fig.update_xaxes(title_text="Bond Angle [°]", row=2, col=2)

    # Minimal look for the rug plot rows
    for col in (1, 2):
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=col)


def _create_figure_for_atom(data: dict, parameters: dict, res_name: str, at_name: str) -> go.Figure:
    atom_data = data[res_name][at_name]
    atom_params = parameters[res_name][at_name]
    ids = atom_data["ID"]

    fig = _create_figure(res_name, at_name)
    _add_histogram_and_rug(fig, atom_data["bond_length"], ids, atom_params.get("bond_length", []),
                           col=1, title="Bond Length", unit="Å")
    _add_histogram_and_rug(fig, atom_data["bond_angle"], ids, atom_params.get("bond_angle", []),
                           col=2, title="Bond Angle", unit="°")
    _add_polar_histogram(fig, atom_data["dihedral_angle"], ids, atom_params.get("dihedral_angle", []))
    _format_figure(fig, res_name, at_name, len(ids))
    return fig


def _file_name(res_name: str, at_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", f"{res_name}_{at_name}") + ".html"


def _write_index(index_file: Path, files: dict) -> None:
    """Writes an HTML page with links to the graphs, grouped by residue."""
    sections = []
    for res_name, atoms in files.items():
        links = "".join(f'<li><a href="{html.escape(file_name)}">{html.escape(at_name)}</a></li>'
                        for at_name, file_name in atoms.items())
        sections.append(f"<h2>{html.escape(res_name)}</h2><ul>{links}</ul>")
    index_file.write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>PROPTIMUS PREOPTIMER graphs</title>"
        "<style>body{font-family:sans-serif;margin:2em}ul{display:flex;flex-wrap:wrap;gap:.5em 1.5em;"
        "list-style:none;padding:0}</style></head><body>"
        "<h1>Atom types</h1>" + "".join(sections) + "</body></html>",
        encoding="utf-8",
    )


def make_html_graphs(data: dict, parameters: dict, output_dir: str, open_browser: bool = False) -> None:
    """
    Saves one HTML graph per atom type of every residue to output_dir, together with
    index.html linking to all of them. The index is opened in the default browser if open_browser is set.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    total = sum(len(atoms) for atoms in data.values())
    with tqdm(total=total, desc="Creating HTML graphs", unit="graph") as progress:
        for res_name, atoms in data.items():
            files[res_name] = {}
            for at_name in atoms:
                progress.set_postfix_str(f"{res_name} {at_name}")
                file_name = _file_name(res_name, at_name)
                fig = _create_figure_for_atom(data, parameters, res_name, at_name)
                # plotly.min.js is written once to the directory and shared by all graphs
                fig.write_html(out / file_name, include_plotlyjs="directory")
                files[res_name][at_name] = file_name
                progress.update()
    index_file = out / "index.html"
    _write_index(index_file, files)
    if open_browser:
        webbrowser.open(index_file.resolve().as_uri())
