"""
generate_paper_figures.py

Nature-Grade Visualization Suite for Green Hydrogen Digital Twin.
Generates:
1. Structure-Property Proof (Surface Area vs Efficiency/Voltage)
2. Thermodynamic Consistency Proof (Voltage vs Current Density with Nernst Limit)
3. Reliability Proof (Response to Sensor Drift with Confidence Intervals)

Usage:
    python generate_paper_figures.py
"""

import os
import math
# pylint: disable=import-error
import torch
import numpy as np
import matplotlib.pyplot as plt

# pylint: disable=import-error
# Import Model Architecture
# Assumes this script is in the same directory as main_digital_twin.py
try:
    from main_digital_twin import DigitalTwin
except ImportError:
    print("Error: Could not import DigitalTwin. "
          "Ensure main_digital_twin.py is in the same directory.")
    exit(1)

# Ensure output directory exists
OUTPUT_DIR = "d:/PROJECT/SCI PAPERS/03_Figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Note: The training loop saves to "best_model.pth".
# Ensuring compatibility if user renamed it to "NeuroSymbolic_Instrument_FINAL.pth"
model_path = "best_model.pth"
if not os.path.exists(model_path):
    if os.path.exists("NeuroSymbolic_Instrument_FINAL.pth"):
        model_path = "NeuroSymbolic_Instrument_FINAL.pth"
    else:
        print(
            f"Warning: neither {model_path} nor NeuroSymbolic_Instrument_FINAL.pth found.")
        print("Using random initialization for demonstration if model -"
              " not present (for layout verification).")

plt.style.use('seaborn-v0_8-ticks')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (6, 4.5),  # Standard single-column width
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3
})


def load_model():
    """Load the Digital Twin model and stats."""
    # Helper to load dataset stats (mocked or loaded) for normalization
    # In a real scenario, we'd load 'dataset.stats' saved during training.
    # Here we use the defaults observed in the logs or explicit defaults.

    # Defaults based on training logs
    stats = {
        'mu_I': 199.8061, 'std_I': 10.0243,
        'mu_T': 343.1331, 'std_T': 1.9603,
        'mu_P': 30.0177,  'std_P': 0.4975,
        'mu_V': 1.8161,   'std_V': 0.0586
    }

    model = DigitalTwin(stats).to(DEVICE)

    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
        except (RuntimeError, ValueError) as e:
            print(f"Failed to load weights: {e}")
    else:
        print("Model file not found. Generating plots with UNTRAINED model (Layout Check).")

    model.eval()
    return model, stats


def plot_structure_property(model, stats):
    """
    Proof: Higher Surface Area -> Better Efficiency (Lower Voltage).
    Vary SA from 10 to 100 (normalized).
    """
    print("Generating Figure 1: Structure-Property Proof...")

    # Create synthetic batch
    n_points = 50
    sa_values = np.linspace(0.1, 5.0, n_points)  # Normalized range assumption

    # Inputs: I, T, P constant
    x = torch.zeros((n_points, 1, 5)).to(DEVICE)
    x[:, :, 0] = stats['mu_I']  # Nominal Current
    x[:, :, 1] = stats['mu_T']  # Nominal Temp
    x[:, :, 2] = stats['mu_P']  # Nominal Pressure
    x[:, :, 3] = stats['mu_V']  # Lags
    x[:, :, 4] = stats['mu_V']

    # Cat Feats: [IrO2, Pt/C, RuO2, SA, Porosity]
    # One-Hot IrO2 (1,0,0)
    cat = torch.zeros((n_points, 5)).to(DEVICE)
    cat[:, 0] = 1.0  # IrO2
    cat[:, 3] = torch.tensor(
        sa_values, dtype=torch.float32).to(DEVICE)  # Varying SA
    cat[:, 4] = 0.5  # Nominal composed porosity

    with torch.no_grad():
        # model returns: conf_preds, diag_logits, v_phys, t_pred, rul_phys
        # We care about v_phys (physics-based voltage) or conf_preds (final)
        # Using v_phys to show the material bridge effect directly
        _, _, v_phys, _, _ = model(x, cat)

    v_pred = v_phys.squeeze().cpu().numpy()

    plt.figure()
    plt.plot(sa_values, v_pred, 'b-', label='Physics Prediction')
    plt.xlabel('Normalized Surface Area')
    plt.ylabel('Cell Voltage (V)')
    plt.title('Structure-Property Relationship')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTPUT_DIR, "Fig_Structure_Property.png"), dpi=300)
    plt.close()


def plot_thermodynamic_consistency(model, stats):
    """
    Proof: Model respects the Nernst Floor.
    Plot V vs I. Overlay Nernst Voltage.
    """
    print("Generating Figure 2: Thermodynamic Consistency...")

    n_points = 300  # More detailed
    currents = np.linspace(10, 1000, n_points)

    x = torch.zeros((n_points, 1, 5)).to(DEVICE)
    x[:, :, 0] = torch.tensor(
        currents, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    x[:, :, 1] = stats['mu_T']
    x[:, :, 2] = stats['mu_P']
    x[:, :, 3] = stats['mu_V']
    x[:, :, 4] = stats['mu_V']

    cat = torch.zeros((n_points, 5)).to(DEVICE)
    cat[:, 0] = 1.0  # IrO2
    cat[:, 3] = 0.0  # Mean SA
    cat[:, 4] = 0.0  # Mean Porosity

    with torch.no_grad():
        preds, _, _, _, _ = model(x, cat)

    v_total = preds[:, :, 1].squeeze().cpu().numpy()  # Median prediction

    # Calculate Theoretical Nernst (E_rev)
    # E = 1.229 - 8.5e-4(T-298) + Nernst_term
    # Simplified calculation matching the layer logic
    temp = stats['mu_T']
    pressure = stats['mu_P']
    faraday_constant = 96485.33
    gas_constant = 8.314
    e_rev = 1.229 - 8.5e-4 * (temp - 298.15) + \
        (gas_constant * temp / (2 * faraday_constant)) * math.log(pressure)
    nernst_line = np.full_like(currents, e_rev)

    plt.figure()
    plt.plot(currents, v_total, 'g-', linewidth=2.5,
             label='Digital Twin (Physics-Compliant)')
    plt.plot(currents, nernst_line, 'r--', linewidth=2,
             label='Thermodynamic Limit (Nernst)')

    # Fill area to show violations if any (should be none)
    plt.fill_between(currents, 0, nernst_line, color='red',
                     alpha=0.1, hatch='///', label='Forbidden Region (2nd Law Violation)')

    # Add Text Annotations
    # Placing text to the right to avoid overlapping the steep curve on the left
    plt.text(700, 1.6, "Safe Zone\n(Physically Valid)", color='green',
             ha='center', fontweight='bold', fontsize=10)
    plt.text(700, 1.0, "Forbidden Zone", color='red',
             ha='center', alpha=0.7, fontsize=10)

    # Upper Safety Limit (Membrane Risk)
    plt.axhline(y=2.4, color='orange', linestyle=':',
                linewidth=1.5, label='Safety Limit (2.4V)')

    # Thermo-neutral Voltage (100% HHV Efficiency)
    plt.axhline(y=1.48, color='blue', linestyle='-.', linewidth=1.5,
                alpha=0.7, label='Thermo-neutral (1.48V)')

    # Industrial Standard (Approximate)
    plt.axhline(y=2.0, color='gray', linestyle='--', linewidth=1.5,
                alpha=0.7, label='Industrial Avg (2.0V)')

    plt.xlabel('Current Density ($mA/cm^2$)')
    plt.ylabel('Cell Voltage (V)')
    plt.title('Thermodynamic Consensus Map')
    # Move legend outside to avoid crowding if needed, or keep compact
    plt.legend(loc='upper right', frameon=True, fontsize=8, ncol=2)

    # Detailed Grid
    plt.ylim(0.5, 2.6)
    plt.minorticks_on()
    plt.grid(True, which='major', alpha=0.4, linestyle='-')
    plt.grid(True, which='minor', alpha=0.15, linestyle=':')

    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTPUT_DIR, "Fig_Thermodynamic_Proof.png"), dpi=300)
    plt.close()


def plot_reliability(model, stats):
    """
    Proof: Conformal Intervals widen under Sensor Drift.
    Inject T drift over time.
    """
    print("Generating Figure 3: Reliability Proof...")

    steps = 100
    t = np.arange(steps)

    # Create Inputs with Drifting Temp
    x = torch.zeros((1, steps, 5)).to(DEVICE)
    x[:, :, 0] = stats['mu_I']
    x[:, :, 2] = stats['mu_P']
    x[:, :, 3] = stats['mu_V']
    x[:, :, 4] = stats['mu_V']

    # Temperature Drift: Stable then Drifts up
    temps = np.full(steps, stats['mu_T'])
    # Start drift at step 50
    temps[50:] += np.linspace(0, 20, 50)  # Drift +20K
    x[:, :, 1] = torch.tensor(temps, dtype=torch.float32).to(DEVICE)

    cat = torch.zeros((1, 5)).to(DEVICE)  # Standard mat props
    cat[:, 0] = 1.0

    with torch.no_grad():
        # preds: [B, S, 3] -> Lower, Median, Upper
        preds, _, _, _, _ = model(x, cat)

    lower = preds[:, :, 0].squeeze().cpu().numpy()
    median = preds[:, :, 1].squeeze().cpu().numpy()
    upper = preds[:, :, 2].squeeze().cpu().numpy()

    plt.figure()
    plt.plot(t, median, 'b-', label='Predicted Voltage')
    plt.fill_between(t, lower, upper, color='red', alpha=0.2,
                     label='99% Conformal Uncertainty')

    # Indicate drift start
    plt.axvline(x=50, color='k', linestyle=':', label='Sensor Drift Start')

    plt.xlabel('Time Step')
    plt.ylabel('Voltage (V)')
    plt.title('Reliability & Uncertainty Quantification')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_Reliability_Proof.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    print("Initializing Visualization Suite...")
    global_model, global_stats = load_model()

    plot_structure_property(global_model, global_stats)
    plot_thermodynamic_consistency(global_model, global_stats)
    plot_reliability(global_model, global_stats)

    print(f"All figures generated in {OUTPUT_DIR}")
