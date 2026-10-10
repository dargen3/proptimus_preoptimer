import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_graph(data: dict, parameters: dict, res_name: str, at_name: str) -> None:
    """
    Creates a Plotly figure with subplots for Bond Lengths, Bond Angles,
    and Dihedral Angles.
    """
    # Vytažení konkrétních dat pro přehlednější přístup a lepší výkon
    atom_data = data[res_name][at_name]
    atom_params = parameters[res_name][at_name]
    ids = atom_data["ID"]

    # Inicializace subplotů
    fig = make_subplots(
        rows=2,
        cols=3,
        shared_xaxes=True,
        row_heights=[0.95, 0.05],
        vertical_spacing=0.03,
        horizontal_spacing=0.06,
        subplot_titles=("Bond Lengths", "Bond Angles", "Dihedral Angles"),
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "polar", "rowspan": 2}],
            [{"type": "xy"}, {"type": "xy"}, None]
        ]
    )

    fig.update_layout(barmode='overlay')

    def add_hist_and_rug(metric_key: str, col: int, title: str, unit: str) -> None:
        """Pomocná funkce pro vytvoření histogramu a rug plotu pro první 2 sloupce."""
        values = atom_data[metric_key]
        min_val, max_val = min(values), max(values)

        # Ochrana proti dělení nulou, pokud by byly všechny hodnoty stejné
        bin_size = (max_val - min_val) / 30 if max_val > min_val else 1

        # 1. Hlavní histogram
        fig.add_trace(
            go.Histogram(
                x=values,
                autobinx=False,
                xbins=dict(
                    start=min_val - 0.5 * bin_size,
                    end=max_val + 0.5 * bin_size,
                    size=bin_size
                ),
                name=title,
                marker_color='#636EFA',
                marker_line_width=0,  # Zabrání tomu, aby úzké biny vypadaly jen jako čáry
                hovertemplate=f"{title}: %{{x}}{unit}<br>Count: %{{y}}<extra></extra>"
            ),
            row=1, col=col
        )

        # 2. Rug plot pomocí Scatter grafu
        hover_texts = [f"ID: {i}<br>{title}: {v:.3f}{unit}" for i, v in zip(ids, values)]
        fig.add_trace(
            go.Scatter(
                x=values,
                y=[0] * len(values),
                mode='markers',
                marker=dict(symbol='line-ns-open', color='#636EFA', size=20, line_width=1.5),
                text=hover_texts,
                hoverinfo='text',
                showlegend=False,
                name=f"{title} Rug"
            ),
            row=2, col=col
        )

        # Referenční svislé čáry z parametrů
        for parameter in atom_params.get(metric_key, []):
            fig.add_vline(x=parameter, line_color="red", line_width=2, row=1, col=col)
            fig.add_vline(x=parameter, line_color="red", line_width=2, row=2, col=col)

    # Přidání prvních dvou sloupců
    add_hist_and_rug("bond_length", col=1, title="Bond Length", unit="Å")
    add_hist_and_rug("bond_angle", col=2, title="Bond Angle", unit="°")

    # --- Dihedral angles subplot (polární graf) ---
    dihedral_angles = atom_data["dihedral_angle"]
    num_bins = 120
    counts, bin_edges = np.histogram(dihedral_angles, bins=num_bins, range=(-180, 180))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = (bin_edges[-1] - bin_edges[0]) / num_bins

    # Barpolar stopa
    fig.add_trace(
        go.Barpolar(
            r=counts,
            theta=bin_centers,
            width=bin_width,
            name='Count',
            marker_color='#636EFA',
            marker_line_color='white',
            marker_line_width=0,
            opacity=0.8,
        ),
        row=1, col=3
    )

    # Vlastní "rug" pro polární graf
    r_rug, theta_rug, hover_texts_polar = [], [], []
    max_count = max(counts) if max(counts) > 0 else 1

    # Zjednodušený výpočet poloměrů
    outer_radius_start = max_count * 1.05
    outer_radius_end = max_count * 1.15

    for angle, val_id in zip(dihedral_angles, ids):
        r_rug.extend([outer_radius_start, outer_radius_end, None])
        theta_rug.extend([angle, angle, None])
        hover_str = f"ID: {val_id}<br>Angle: {angle:.2f}°"
        hover_texts_polar.extend([hover_str, hover_str, None])

    fig.add_trace(
        go.Scatterpolar(
            r=r_rug,
            theta=theta_rug,
            mode='lines',
            line=dict(color='#636EFA'),
            showlegend=False,
            text=hover_texts_polar,
            hoverinfo='text',
            name='Rug',
        ),
        row=1, col=3
    )

    # Přidání červené čáry na cílové hodnoty dihedrálních úhlů
    for parameter in atom_params.get("dihedral_angle", []):
        fig.add_trace(
            go.Scatterpolar(
                r=[0, outer_radius_end],
                theta=[parameter, parameter],
                mode='lines',
                line=dict(color='red', width=2),
                showlegend=False,
                hoverinfo='text',
                text="Target",
                name='Target Angle'
            ),
            row=1, col=3
        )

    # Společný formát celého obrázku
    fig.update_layout(
        title_text=f'Analysis of bond lengths and angles for atom {res_name} {at_name}',
        title_x=0.5,
        title_font_size=20,
        showlegend=False,
        polar=dict(
            radialaxis=dict(showline=False, ticks=''),
            angularaxis=dict(
                rotation=90,
                direction='clockwise',
                tickmode='array',
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['0°', '45°', '90°', '135°', '±180°', '-135°', '-90°', '-45°']
            ),
        )
    )

    # Popisky os
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Bond Length [Å]", row=2, col=1)
    fig.update_xaxes(title_text="Bond Angle [°]", row=2, col=2)

    # Čistší formát pro miniaturní řádek s Rug ploty
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=2)

    fig.show()
#
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
#
# def create_graph(data, parameters, res_name, at_name):
#
#     fig = make_subplots(
#         rows=2, cols=3,
#         shared_xaxes=True,
#         row_heights=[0.95, 0.05],
#         vertical_spacing=0.03,
#         horizontal_spacing=0.06,
#         subplot_titles=("Bond Lengths", "Bond Angles", "Dihedral Angles"),
#         specs=[
#             [{"type": "xy"}, {"type": "xy"}, {"type": "polar", "rowspan": 2}],
#             [{"type": "xy"}, {"type": "xy"}, None]
#         ]
#     )
#
#     fig.update_layout(barmode='overlay')
#
#     # Pomocná funkce pro vytvoření histogramu a rug plotu pro prvni 2 sloupce
#     def add_hist_and_rug(metric_key, col, title, unit, parameters):
#         values = data[res_name][at_name][metric_key]
#         ids = data[res_name][at_name]["ID"]
#
#         min_val = min(values)
#         max_val = max(values)
#         # Ochrana proti dělení nulou, pokud by byly všechny hodnoty stejné
#         bin_size = (max_val - min_val) / 30 if max_val > min_val else 1
#
#         # 1. Hlavní histogram
#         fig.add_trace(
#             go.Histogram(
#                 x=values,
#                 autobinx=False,
#                 xbins=dict(
#                     start=min_val - 0.5 * bin_size,
#                     end=max_val + 0.5 * bin_size,
#                     size=bin_size
#                 ),
#                 name=title,
#                 marker_color='#636EFA',
#                 marker_line_width=0,  # Klíčové: zabrání tomu, aby úzké biny vypadaly jen jako čáry
#                 hovertemplate=f"{title}: %{{x}}{unit}<br>Count: %{{y}}<extra></extra>"
#             ),
#             row=1, col=col
#         )
#
#         # 2. Rug plot pomocí Scatter grafu
#         hover_texts = [f"ID: {i}<br>{title}: {v:.3f}{unit}" for i, v in zip(ids, values)]
#         fig.add_trace(
#             go.Scatter(
#                 x=values,
#                 y=[0] * len(values),  # Všechny body na jedné linii
#                 mode='markers',
#                 marker=dict(symbol='line-ns-open', color='#636EFA', size=20, line_width=1.5),
#                 text=hover_texts,
#                 hoverinfo='text',
#                 showlegend=False,
#                 name=f"{title} Rug"
#             ),
#             row=2, col=col
#         )
#
#         ats_type_parameters = parameters[res_name][at_name][metric_key]
#         for parameter in ats_type_parameters:
#             fig.add_vline(x=parameter, line_color="red", line_width=2, row=1, col=col)
#             fig.add_vline(x=parameter, line_color="red", line_width=2, row=2, col=col)
#
#
#     # Přidání prvních dvou sloupců
#     add_hist_and_rug("bond_length", col=1, title="Bond Length", unit="Å", parameters=parameters)
#     add_hist_and_rug("bond_angle", col=2, title="Bond Angle", unit="°", parameters=parameters)
#
#     # --- dihedral angles subplot (polární graf z vašeho původního kódu) ---
#     dihedral_angles = data[res_name][at_name]["dihedral_angle"]
#     ids_dihedrals = data[res_name][at_name]["ID"]
#     num_bins = 120
#     counts, bin_edges = np.histogram(dihedral_angles, bins=num_bins, range=(-180, 180))
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#     bin_width = (bin_edges[-1] - bin_edges[0]) / num_bins
#
#     # Barpolar stopa
#     fig.add_trace(
#         go.Barpolar(
#             r=counts,
#             theta=bin_centers,
#             width=bin_width,
#             name='Count',
#             marker_color='#636EFA',
#             marker_line_color='white',
#             marker_line_width=0,
#             opacity=0.8,
#         ),
#         row=1, col=3
#     )
#
#     # Vlastní "rug" pro polární graf
#     r_rug = []
#     theta_rug = []
#     hover_texts_polar = []
#     max_count = max(counts) if max(counts) > 0 else 1
#     outer_radius_start = max_count + (max_count * 0.05)
#     outer_radius_end = outer_radius_start + (max_count * 0.1)
#
#     for t, val_id in zip(dihedral_angles, ids_dihedrals):
#         r_rug.extend([outer_radius_start, outer_radius_end, None])
#         theta_rug.extend([t, t, None])
#         hover_str = f"ID: {val_id}<br>Angle: {t:.2f}°"
#         hover_texts_polar.extend([hover_str, hover_str, None])
#
#
#     fig.add_trace(
#         go.Scatterpolar(
#             r=r_rug,
#             theta=theta_rug,
#             mode='lines',
#             line=dict(color='#636EFA'),
#             showlegend=False,
#             text=hover_texts_polar,
#             hoverinfo='text',
#             name='Rug',
#         ),
#         row=1, col=3
#     )
#
#     # Přidání červené čáry na hodnotu úhlu 20°
#     if "dihedral_angle" in parameters[res_name][at_name].keys():
#         for parameter in parameters[res_name][at_name]["dihedral_angle"]:
#             fig.add_trace(
#                 go.Scatterpolar(
#                     r=[0, outer_radius_end],
#                     theta=[parameter, parameter],
#                     mode='lines',
#                     line=dict(color='red', width=2),
#                     showlegend=False,
#                     hoverinfo='text',
#                     text=f"Target",
#                     name='Target Angle'
#                 ),
#                 row=1, col=3
#             )
#
#     # Společný formát celého obrázku
#     fig.update_layout(
#         title_text=f'Analysis of bond lengths and angles for atom {res_name} {at_name}',
#         title_x=0.5,
#         title_font_size=20,
#         showlegend=False,
#         polar=dict(
#             radialaxis=dict(showline=False, ticks=''),
#             angularaxis=dict(
#                 rotation=90,
#                 direction='clockwise',
#                 tickmode='array',
#                 tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
#                 ticktext=['0°', '45°', '90°', '135°', '±180°', '-135°', '-90°', '-45°']
#             ),
#         )
#     )
#
#     # Popisky os
#     fig.update_yaxes(title_text="Count", row=1, col=1)
#     fig.update_yaxes(title_text="Count", row=1, col=2)
#     fig.update_xaxes(title_text="Bond Length [Å]", row=2, col=1)
#     fig.update_xaxes(title_text="Bond Angle [°]", row=2, col=2)
#
#     # Čistší formát pro miniaturní řádek s Rug ploty
#     fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=1)
#     fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=2)
#
#     fig.show()
