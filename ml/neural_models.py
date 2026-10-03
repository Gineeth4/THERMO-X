"""
Neural-network models for THERMO-X.

Contains:

1. Deep MLP
2. LSTM sequence predictor

Both models are implemented with PyTorch.

The LSTM is particularly important because thermal systems
are dynamic systems whose present state depends on their
recent history.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


# ============================================================
# DEVICE
# ============================================================


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DEEP MLP
# ============================================================


class ThermalMLP(
    nn.Module
):
    """
    Deep feed-forward neural network for thermal prediction.

    Architecture:

        Input
          ↓
        128
          ↓
        64
          ↓
        32
          ↓
        16
          ↓
        Output temperature
    """

    def __init__(
        self,
        input_size: int,
    ) -> None:

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_size,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                128,
                64,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                64,
                32,
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                16,
            ),

            nn.ReLU(),

            nn.Linear(
                16,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(
            x
        )


# ============================================================
# LSTM
# ============================================================


class ThermalLSTM(
    nn.Module
):
    """
    LSTM sequence model for THERMO-X.

    Input:

        [batch, sequence_length, feature_count]

    Output:

        future temperature
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:

        super().__init__()

        # Dropout is only active between LSTM layers
        # when more than one layer exists.
        effective_dropout = (
            dropout
            if num_layers > 1
            else 0.0
        )

        self.lstm = nn.LSTM(
            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True,

            dropout=effective_dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(
                hidden_size,
                32,
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        output, _ = self.lstm(
            x
        )

        # Only the final sequence state is used
        # for future-temperature prediction.
        final_state = (
            output[
                :,
                -1,
                :
            ]
        )

        return self.head(
            final_state
        )