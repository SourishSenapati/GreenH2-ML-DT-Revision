"""
Physics Layer (The Immutable Laws) for Green Hydrogen Digital Twin.
Encodes Butler-Volmer and Nernst equations as a differentiable manifold.
"""
# pylint: disable=import-error
import torch
import torch.nn as nn
# pylint: enable=import-error


class ButlerVolmerLayer(nn.Module):
    """
    Differentiable Physics Layer.
    Encodes the theoretical behavior of the electrolyzer.
    """

    def __init__(self, faraday_constant=96485, gas_constant=8.314):
        super().__init__()
        self.faraday_constant = faraday_constant  # Faraday Constant
        self.gas_constant = gas_constant  # Gas Constant

    def forward(self, inputs):
        """
        Args:
            inputs: [Batch, Time, Features]
            Features map: 0:Current, 1:Temp, 2:Pressure (if avail, else 30 bar default)
        """
        # Ensure inputs are at least 3D
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)

        current = inputs[:, :, 0]
        temp = inputs[:, :, 1]

        # 1. Nernst Potential (Thermodynamic Baseline)
        # E_rev = 1.229 - 0.9e-3 * (T - 298.15)
        # Note: Using 0.0009 coefficient per prompt specification
        e_rev = 1.229 - 0.0009 * (temp - 298.15)

        # 2. Activation Overpotential (Butler-Volmer Approx)
        # V_act = (RT/2F) * ln(I / I_0)
        # Note: Added epsilon to avoid log(0) and I/I0 simplified to proportional to I
        # Ideally I0 is T-dependent. The Prompt uses a simplified ln(I) form.
        # We stick to the Prompt's logic for consistency.
        v_act = (
            self.gas_constant * temp / (2 * self.faraday_constant)
        ) * torch.log(current + 1e-6)

        # 3. Ohmic Loss
        # R_mem is often a function of T, simplified here
        # R_mem = 0.15 * exp(1268 * (1/303 - 1/T))
        r_mem = 0.15 * torch.exp(1268 * (1/303 - 1/(temp + 1e-6)))
        v_ohm = current * r_mem

        return e_rev + v_act + v_ohm
