"""
Model Architecture (The Transformer) for Green Hydrogen Digital Twin.
Implements the Neuro-Symbolic Transformer (NatureClassTFT).
"""
import torch
import torch.nn as nn
from physics_layer import ButlerVolmerLayer


class NatureClassTFT(nn.Module):
    """
    Physics-Informed Temporal Fusion Transformer with Bayesian Uncertainty Head.
    """

    def __init__(self, input_dim, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()

        # 1. Physics Encoder
        self.physics_layer = ButlerVolmerLayer()
        self.physics_projection = nn.Linear(1, d_model)

        # 2. Data Encoder (Learnable)
        self.input_projection = nn.Linear(input_dim, d_model)

        # 3. Temporal Fusion Encoder (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Critical for convergence
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # 4. Bayesian Head (Flipout / Variational Approximation)
        # Simplified here as separate Mean and Sigma heads
        self.head_mu = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)  # Predicts the Residual Mean
        )
        self.head_sigma = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Variance must be positive
        )

    def forward(self, x_raw):
        """
        Args:
            x_raw: [Batch, Sequence, Features]
        Returns:
            v_final: [Batch, 1 (Last Step)]
            residual_sigma: [Batch, 1 (Last Step)]
        """

        # Path A: Physics (The "Prior")
        # Physics layer expects [B, T, F]. Output is [B, T]
        v_theory = self.physics_layer(x_raw).unsqueeze(-1)  # [B, T, 1]
        phys_embed = self.physics_projection(v_theory)

        # Path B: Data (The "Correction")
        data_embed = self.input_projection(x_raw)

        # Fusion: Physics + Data
        combined_embed = phys_embed + data_embed

        # Attention Mechanism
        latent = self.transformer(combined_embed)

        # Bayesian Output
        # We take the last time step for forecasting
        latent_last = latent[:, -1, :]

        residual_mu = self.head_mu(latent_last)
        residual_sigma = self.head_sigma(latent_last)

        # Final Prediction = Theory + Predicted Residual
        # This ensures the model CANNOT violate physics grossly
        v_theory_last = v_theory[:, -1, :]
        v_final = v_theory_last + residual_mu

        return v_final, residual_sigma
