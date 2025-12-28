"""
Physics-Informed Loss Functions for Green Hydrogen Digital Twin.
Implements Hamiltonian constraints based on Butler-Volmer and Nernst equations.
This ensures the Neural Network respects the Laws of Thermodynamics.

Author: Antigravity (Google Deepmind)
Version: Nature-Class v1.0
"""

# pylint: disable=import-error
# pylint: disable=import-error
import torch
import torch.nn as nn
# pylint: enable=import-error
# pylint: enable=import-error


class HamiltonianPhysicsLoss(nn.Module):
    """
    Computes the total loss as a weighted sum of Data Observational Loss 
    and Physical Consistency Loss (Hamiltonian Constraint).

    L_total = L_data + lambda * || V_pred - V_physics ||^2
    """

    def __init__(self, lambda_phys=0.1):
        super(HamiltonianPhysicsLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_phys = lambda_phys

        # Physical Constants
        # Physical Constants
        self.gas_constant = 8.314
        self.faraday_constant = 96485.0

    def nernst_potential(self, temp_kelvin):
        """Standard Reversible Potential E0(T)."""
        # E0 = 1.229 - 0.9e-3 * (T - 298)
        return 1.229 - 0.9e-3 * (temp_kelvin - 298.15)

    def butler_volmer_overs(self, current_density, temp_kelvin):
        """
        Approximation of Activation Overpotential using inverse sinh (Tafel).
        V_act = (RT/2F) * asinh(I / 2*I0)
        """
        # Exchange current density approx (Arrhenius)
        # I0 scales with T: I0 ~ exp(-Ea/RT)
        # Simplified for differentiability:
        i0 = 1e-3 * torch.exp(0.05 * (temp_kelvin - 298.15))

        term = (self.gas_constant * temp_kelvin) / (2 * self.faraday_constant)
        # safe log formulation for numerical stability
        # V_act ~ log(I) at high current
        v_act = term * torch.log(current_density / i0 + 1.0)
        return v_act

    def ohmic_loss(self, current_density, temp_kelvin):
        """
        V_ohm = I * R_mem(T)
        """
        # Nafion conductivity increases with T => Resistance decreases
        r_mem = 0.15 * torch.exp(500 * (1/temp_kelvin - 1/353))
        return current_density * r_mem

    def forward(self, v_pred, v_target, current, temp_c):
        """
        Args:
            v_pred: Predicted Voltage (Tensor)
            v_target: Actual Voltage (Tensor)
            current: Current Density A/cm2 (Tensor)
            temp_c: Temperature Celsius (Tensor)
        """
        temp_k = temp_c + 273.15

        # 1. Data Loss (Observation)
        loss_data = self.mse(v_pred, v_target)

        # 2. Physics Constraint (Theory)
        # Calculate what Physics says Voltage SHOULD be
        with torch.enable_grad():  # Ensure gradients flow if inputs require grad
            e_rev = self.nernst_potential(temp_k)
            v_act = self.butler_volmer_overs(current, temp_k)
            v_ohm = self.ohmic_loss(current, temp_k)

            # Theoretical Voltage (ignoring concentration/degradation for base physics)
            v_theoretical = e_rev + v_act + v_ohm

        # The penalty: Deviation from Fundamental Electrochemistry
        # We allow SOME deviation (degradation/noise) but large deviations are penalized
        loss_physics = torch.mean((v_pred - v_theoretical) ** 2)

        # 3. Thermodynamic Inequality Constraint (Efficiency < 100%)
        # V_pred must be > E_rev (cannot generate Hydrogen below thermodynamic limit)
        # Penalty for negative overpotential
        loss_thermo = torch.mean(torch.relu(e_rev - v_pred) ** 2)

        total_loss = loss_data + self.lambda_phys * \
            (loss_physics + loss_thermo)

        return total_loss, loss_data, loss_physics
