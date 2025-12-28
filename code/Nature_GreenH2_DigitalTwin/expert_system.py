"""
Expert System for GreenH2 Digital Twin.
Wraps the scientific model to provide engineering reports.
"""
import os
import sys
import json
from typing import Dict
# pylint: disable=import-error
import torch  # type: ignore
# pylint: enable=import-error
from main_digital_twin import (
    DigitalTwin, ExperimentConfig, RealWorldDataset, AdaptiveConformalInference
)

# Add path to code directory to import main_digital_twin
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class GreenH2ExpertSystem:
    """
    Phase 5: Cognitive Digital Twin Wrapper.
    Translates Scientific Twin outputs into Engineering Reports.
    """

    def __init__(self, model_path: str = "best_model.pth",
                 calibration_path: str = "conformal_calibration.json"):
        self.c = ExperimentConfig
        self.device = self.c.DEVICE

        # Load Data & Stats (needed for initialization)
        print(">> [System] Initializing Digital Twin Core...")
        self.dataset = RealWorldDataset(self.c.DATA_PATH, self.c.SEQ_LEN)

        # Load Model
        self.model = DigitalTwin(self.dataset.stats).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(
                model_path, map_location=self.device))
            print(f">> [System] Loaded ScientificTwin v9.0 from {model_path}")
        else:
            print(
                "!! [Warning] No model checkpoint found. "
                "Running with uninitialized weights (Demo Mode)."
            )

        self.model.eval()

        # Load Calibration
        if os.path.exists(calibration_path):
            with open(calibration_path, "r", encoding='utf-8') as f:
                self.q_hat = json.load(f)["q_hat"]
            print(
                f">> [System] Loaded Conformal Calibration (q_hat={self.q_hat:.4f})")
            self.aci = AdaptiveConformalInference(self.q_hat)
        else:
            self.q_hat = 0.0
            print("!! [Warning] Uncalibrated. 95% CI will be approximate.")
            self.aci = AdaptiveConformalInference(0.0)

        # Diagnostic Map
        self.diag_map = {0: "Healthy", 1: "Membrane Degradation",
                         2: "Catalyst Poisoning", 3: "Sensor Drift"}

    def process_query(self, query: str, sensor_data: Dict[str, float]) -> str:
        """
        Input: Natural Language Query + Sensor Snapshot
        Output: Engineering Report
        """
        # 1. Parse Intent (Regex/Fuzzy)
        query = query.lower()
        if "voltage" in query or "prediction" in query:
            intent = "predict"
        elif "fault" in query or "health" in query or "status" in query:
            intent = "diagnose"
        elif "physics" in query or "alpha" in query or "material" in query:
            intent = "physics"
        else:
            intent = "full_report"

        # 2. Prepare Input Tensor
        # We need a sequence. For single point query, we might need to pad or fetch history.
        # For this expert system, let's assume 'sensor_data' contains valid history
        # or we replicate it. Ideally, we would fetch from a database.
        # Here we simulate a sequence by repeating the point.
        # Inputs: [Current, Temp, Pressure, V_lag1, V_lag2]
        # Catalyst: [IrO2, Pt/C, RuO2, SA, Porosity]

        seq_len = self.c.SEQ_LEN
        x_in = torch.zeros(1, seq_len, 5).to(self.device)

        # Parse Sensor Data
        i_val = sensor_data.get("current", 100.0)
        t_val = sensor_data.get("temperature", 350.0)
        p_val = sensor_data.get("pressure", 30.0)

        # Fill Sequence (Steady State assumption for simplicity of this demo wrapper)
        x_in[:, :, 0] = i_val
        x_in[:, :, 1] = t_val
        x_in[:, :, 2] = p_val
        x_in[:, :, 3:] = 1.8  # Dummy lags

        # Catalyst Features from Request or Default
        cat_type = sensor_data.get("catalyst", "IrO2")
        cat_feats = torch.zeros(1, 5).to(self.device)

        # One-Hot
        if cat_type == "IrO2":
            cat_feats[0, 0] = 1.0
        elif cat_type == "Pt/C":
            cat_feats[0, 1] = 1.0
        elif cat_type == "RuO2":
            cat_feats[0, 2] = 1.0

        # Props
        cat_feats[0, 3] = sensor_data.get("surface_area", 0.5)  # Normalized
        cat_feats[0, 4] = sensor_data.get("porosity", 0.5)  # Normalized

        # 3. Run Inference
        with torch.no_grad():
            preds, diag_logits, v_phys, t_recon = self.model(x_in, cat_feats)
        _ = v_phys  # Suppress unused variable

        # 4. Reliability Check (Virtual Sensor)
        t_inferred = t_recon[0, -1, 0].item()
        # Ensure we are comparing Kelvin
        # T_val from sensor_data is Kelvin (default 350)
        t_sensor = x_in[0, -1, 1].item()  # Should equal T_val

        sensor_error = abs(t_inferred - t_sensor) / (t_sensor + 1e-6)
        sensor_fault = False
        if sensor_error > 0.05:  # 5% Threshold
            sensor_fault = True

        # 5. Extract Key Metrics
        med_pred = preds[0, -1, 1].item()
        # Use ACI q_hat
        current_q_hat = self.aci.q_hat
        lower_pred = preds[0, -1, 0].item() - current_q_hat
        upper_pred = preds[0, -1, 2].item() + current_q_hat

        phys_params = self.model.physics.get_params(cat_feats)
        phys_alpha = phys_params["alpha"]
        phys_k = phys_params.get("k_decay", 0.0)

        probs = torch.softmax(diag_logits[0, -1], dim=0)
        fault_id = torch.argmax(probs).item()
        fault_prob = probs[fault_id].item()
        fault_name = self.diag_map.get(fault_id, "Unknown")

        # 5. Generate Report
        response = []
        response.append(
            f"--- ENGINEERING REPORT [Intent: {intent.upper()}] ---")

        if intent in ["predict", "full_report"]:
            response.append(f"Predicted Voltage: {med_pred:.3f}V")
            response.append(
                f"95% Conformal Interval: [{lower_pred:.3f}V - {upper_pred:.3f}V]")
            if upper_pred - lower_pred > 0.1:
                response.append(
                    "⚠️ Wide Uncertainty Interval - Check Sensor Calibration.")

            if sensor_fault:
                response.append(
                    f"⛔ CRITICAL: Virtual Sensor Mismatch! (Sensor: {t_sensor:.1f}K, "
                    f"Inferred: {t_inferred:.1f}K). Trusting Inferred."
                )

        if intent in ["physics", "full_report"]:
            response.append(f"Physics State (Alpha): {phys_alpha:.4f}")
            response.append(f"Degradation Rate (k): {phys_k:.2e}")
            if cat_type == "IrO2" and phys_alpha < 0.4:
                response.append(
                    "NOTE: Alpha is low for IrO2. Possible surface degradation.")
            response.append(f"Catalyst Context: {cat_type}")

        if intent in ["diagnose", "full_report"]:
            status_emoji = "✅" if fault_id == 0 else "❌"
            response.append(f"Diagnostics: {status_emoji} {fault_name}")
            response.append(f"Confidence: {fault_prob*100:.1f}%")
            if fault_id != 0:
                response.append(
                    f"ACTION REQUIRED: Initiate {fault_name} mitigation protocols.")

        return "\n".join(response)


def interactive_session():
    """Run an interactive session mimicking a CLI."""
    system = GreenH2ExpertSystem()
    print("\n>> Expert System Ready. Type 'exit' to quit.")
    print("Example Query: 'Check system health' or 'Predict voltage'")

    # Default Dummy Sensor Reading
    sensor_state = {
        "current": 250.0,
        "temperature": 333.15,  # 60C
        "pressure": 30.0,
        "catalyst": "IrO2",
        "surface_area": 0.8,
        "porosity": 0.4
    }

    # Simple CLI Loop (Simulated for non-interactive execution)
    # In a real run, this would be input()
    queries = [
        "What is the current voltage prediction?",
        "Diagnose potential faults",
        "Explain the physics variables",
        "Full system report"
    ]

    for q in queries:
        print(f"\nUser: {q}")
        report = system.process_query(q, sensor_state)
        print(report)


if __name__ == "__main__":
    interactive_session()
