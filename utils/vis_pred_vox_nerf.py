import plotly.graph_objects as go
import numpy as np
import torch
import numba as nb
from torch.utils import data
from dataloader.transform_3d import PadMultiViewImage, NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage, RandomScaleImageMultiViewImage
from typing import List, Union
from torch import Tensor
import torch.nn.functional as F
import os
import numpy as np
from torch.utils import data
import yaml
import pickle
from pyquaternion import Quaternion

def vis_occ_plotly(
    vis_aabb: List[Union[int, float]],
    coords: np.array,
    colors: np.array,
    dynamic_coords: List[np.array] = None,
    dynamic_colors: List[np.array] = None,
    x_ratio: float = 1.0,
    y_ratio: float = 1.0,
    z_ratio: float = 0.125,
    size: int = 5,
    black_bg: bool = False,
    title: str = None,
) -> go.Figure:  # type: ignore
    fig = go.Figure()  # start with an empty figure

    # Add static trace
    static_trace = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker=dict(
            size=size,
            color=colors,
            symbol="square",
        ),
    )
    fig.add_trace(static_trace)

    # Add temporal traces
    if dynamic_coords is not None:
        for i in range(len(dynamic_coords)):
            fig.add_trace(
                go.Scatter3d(
                    x=dynamic_coords[i][:, 0],
                    y=dynamic_coords[i][:, 1],
                    z=dynamic_coords[i][:, 2],
                    mode="markers",
                    marker=dict(
                        size=size,
                        color=dynamic_colors[i],
                        symbol="diamond",
                    ),
                )
            )
        steps = []
        for i in range(len(dynamic_coords)):
            step = dict(
                method="restyle",
                args=[
                    "visible",
                    [False] * (len(dynamic_coords) + 1),
                ],  # Include the static trace
                label="Second {}".format(i),
            )
            step["args"][1][0] = True  # Make the static trace always visible
            step["args"][1][i + 1] = True  # Toggle i'th temporal trace to "visible"
            steps.append(step)

        sliders = [
            dict(
                active=0,
                pad={"t": 1},
                steps=steps,
                font=dict(color="white") if black_bg else {},  # Update for font color
            )
        ]
        fig.update_layout(sliders=sliders)
    title_font_color = "white" if black_bg else "black"
    if not black_bg:
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="x",
                    showspikes=False,
                    range=[vis_aabb[0], vis_aabb[3]],
                ),
                yaxis=dict(
                    title="y",
                    showspikes=False,
                    range=[vis_aabb[1], vis_aabb[4]],
                ),
                zaxis=dict(
                    title="z",
                    showspikes=False,
                    range=[vis_aabb[2], vis_aabb[5]],
                ),
                aspectmode="manual",
                aspectratio=dict(x=x_ratio, y=y_ratio, z=z_ratio),
            ),
            margin=dict(r=0, b=10, l=0, t=10),
            hovermode=False,
            title=dict(
                text=title,
                font=dict(color=title_font_color),
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top",
            )
            if title
            else None,  # Title addition
        )
    else:
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="x",
                    showspikes=False,
                    range=[vis_aabb[0], vis_aabb[3]],
                    backgroundcolor="rgb(0, 0, 0)",
                    gridcolor="gray",
                    showbackground=True,
                    zerolinecolor="gray",
                    tickfont=dict(color="gray"),
                ),
                yaxis=dict(
                    title="y",
                    showspikes=False,
                    range=[vis_aabb[1], vis_aabb[4]],
                    backgroundcolor="rgb(0, 0, 0)",
                    gridcolor="gray",
                    showbackground=True,
                    zerolinecolor="gray",
                    tickfont=dict(color="gray"),
                ),
                zaxis=dict(
                    title="z",
                    showspikes=False,
                    range=[vis_aabb[2], vis_aabb[5]],
                    backgroundcolor="rgb(0, 0, 0)",
                    gridcolor="gray",
                    showbackground=True,
                    zerolinecolor="gray",
                    tickfont=dict(color="gray"),
                ),
                aspectmode="manual",
                aspectratio=dict(x=x_ratio, y=y_ratio, z=z_ratio),
            ),
            margin=dict(r=0, b=10, l=0, t=10),
            hovermode=False,
            paper_bgcolor="black",
            plot_bgcolor="rgba(0,0,0,0)",
            title=dict(
                text=title,
                font=dict(color=title_font_color),
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top",
            )
            if title
            else None,  # Title addition
        )
    return fig

def voxel_coords_to_world_coords(
    aabb_min: Union[Tensor, List[float]],
    aabb_max: Union[Tensor, List[float]],
    voxel_resolution: Union[Tensor, List[int]],
    points: Union[Tensor, List[float]] = None,
) -> Tensor:
    aabb_min = torch.tensor(aabb_min) if isinstance(aabb_min, List) else aabb_min
    aabb_max = torch.tensor(aabb_max) if isinstance(aabb_max, List) else aabb_max
    voxel_resolution = (
        torch.tensor(voxel_resolution)
        if isinstance(voxel_resolution, List)
        else voxel_resolution
    )

    if points is None:
        x, y, z = torch.meshgrid(
            torch.linspace(aabb_min[0], aabb_max[0], voxel_resolution[0]),
            torch.linspace(aabb_min[1], aabb_max[1], voxel_resolution[1]),
            torch.linspace(aabb_min[2], aabb_max[2], voxel_resolution[2]),
        )
        return torch.stack([x, y, z], dim=-1)
    else:
        points = torch.tensor(points) if isinstance(points, List) else points

        # Compute voxel size
        voxel_size = (aabb_max - aabb_min) / voxel_resolution

        # Convert voxel coordinates to world coordinates
        world_coords = aabb_min.to(points.device) + points * voxel_size.to(points.device)

        return world_coords

def world_coords_to_voxel_coords(
    point: Union[Tensor, List[float]],
    aabb_min: Union[Tensor, List[float]],
    aabb_max: Union[Tensor, List[float]],
    voxel_resolution: Union[Tensor, List[int]],
) -> Tensor:
    # Convert lists to tensors if necessary
    point = torch.tensor(point) if isinstance(point, List) else point
    aabb_min = torch.tensor(aabb_min) if isinstance(aabb_min, List) else aabb_min
    aabb_max = torch.tensor(aabb_max) if isinstance(aabb_max, List) else aabb_max
    voxel_resolution = (
        torch.tensor(voxel_resolution)
        if isinstance(voxel_resolution, List)
        else voxel_resolution
    )

    # Compute the size of each voxel
    voxel_size = (aabb_max - aabb_min) / voxel_resolution

    # Compute the voxel index for the given point
    voxel_idx = ((point - aabb_min) / voxel_size).long()

    return voxel_idx