from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

class RadialBasisFunction(nn.Module):

    def __init__(
        self,
        grid_min=-1.5,
        grid_max=1.5,
        num_grids=8,
        denominator=None
    ):
        super().__init__()

        grid = torch.linspace(
            grid_min,
            grid_max,
            num_grids
        )

        self.grid = nn.Parameter(
            grid,
            requires_grad=False
        )

        self.denominator = (
            denominator
            or
            (grid_max - grid_min) / (num_grids - 1)
        )

    def forward(self, x):

        return torch.exp(
            -(
                (
                    x[..., None]
                    - self.grid
                )
                / self.denominator
            ) ** 2
        )

class BSRBF_KANLayer(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        grid_size=5,
        spline_order=3,
        base_activation=nn.SiLU,
        grid_range=(-1.5, 1.5)
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layernorm = nn.LayerNorm(
            input_dim
        )

        self.base_activation = (
            base_activation()
        )

        # ================================================================
        # Base branch
        # ================================================================

        self.base_weight = nn.Parameter(
            torch.empty(
                output_dim,
                input_dim
            )
        )

        nn.init.kaiming_uniform_(
            self.base_weight,
            a=math.sqrt(5)
        )

        # ================================================================
        # BSRBF branch
        # ================================================================

        self.spline_weight = nn.Parameter(
            torch.empty(
                output_dim,
                input_dim * (
                    grid_size
                    + spline_order
                )
            )
        )

        nn.init.kaiming_uniform_(
            self.spline_weight,
            a=math.sqrt(5)
        )

        self.rbf = RadialBasisFunction(
            grid_min=grid_range[0],
            grid_max=grid_range[1],
            num_grids=(
                grid_size
                + spline_order
            )
        )

        h = (
            grid_range[1]
            - grid_range[0]
        ) / grid_size

        grid = (
            (
                torch.arange(
                    -spline_order,
                    grid_size
                    + spline_order
                    + 1
                )
                * h
                + grid_range[0]
            )
            .expand(
                input_dim,
                -1
            )
            .contiguous()
        )

        self.register_buffer(
            "grid",
            grid
        )

    # =====================================================================
    # B-SPLINE
    # =====================================================================

    def b_splines(self, x):

        assert x.dim() == 2
        assert x.size(1) == self.input_dim

        grid = self.grid

        x = x.unsqueeze(-1)

        bases = (
            (
                x >= grid[:, :-1]
            )
            &
            (
                x < grid[:, 1:]
            )
        ).to(x.dtype)

        for k in range(
            1,
            self.spline_order + 1
        ):

            bases = (
                (
                    (
                        x
                        - grid[:, :-(k + 1)]
                    )
                    /
                    (
                        grid[:, k:-1]
                        - grid[:, :-(k + 1)]
                    )
                )
                * bases[:, :, :-1]
            ) + (
                (
                    (
                        grid[:, k + 1:]
                        - x
                    )
                    /
                    (
                        grid[:, k + 1:]
                        - grid[:, 1:-k]
                    )
                )
                * bases[:, :, 1:]
            )

        return bases.contiguous()

    # =====================================================================
    # FORWARD
    # =====================================================================

    def forward(self, x):

        x = self.layernorm(x)

        # Base branch
        base_output = F.linear(
            self.base_activation(x),
            self.base_weight
        )

        # B-spline branch
        bs_output = (
            self
            .b_splines(x)
            .reshape(
                x.size(0),
                -1
            )
        )

        # RBF branch
        rbf_output = (
            self
            .rbf(x)
            .reshape(
                x.size(0),
                -1
            )
        )

        # Combined BSRBF
        bsrbf_output = (
            bs_output
            + rbf_output
        )

        bsrbf_output = F.linear(
            bsrbf_output,
            self.spline_weight
        )

        return (
            base_output
            + bsrbf_output
        )

class BSRBF_KAN(nn.Module):

    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        base_activation=nn.SiLU
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        for input_dim, output_dim in zip(
            layers_hidden,
            layers_hidden[1:]
        ):

            self.layers.append(
                BSRBF_KANLayer(
                    input_dim,
                    output_dim,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_activation=base_activation
                )
            )

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x


