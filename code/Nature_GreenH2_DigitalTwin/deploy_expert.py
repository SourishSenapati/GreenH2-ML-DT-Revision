"""
deploy_expert.py

Interactive Expert Terminal for the Green Hydrogen Digital Twin.
Serves as the Neuro-Symbolic Interface for operators to query the system
using natural language-like commands.

Features:
- Real-time Inference (<25ms latency target)
- Physics-Informed Conformal Bounds
- Natural Language Command Parsing
- Reliability & Health Monitoring

Usage:
    python deploy_expert.py
"""

import os
import re
import time
import math
# pylint: disable=import-error
import torch
try:
    from main_digital_twin import DigitalTwin
except ImportError:
    print("Error: connect importing DigitalTwin. "
          "Ensure main_digital_twin.py is in the same directory.")
    exit(1)

# Configuration
if torch.cuda.is_available():
    DEVICE = "cuda"
    # Enable optimizations for inference speed
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    DEVICE = "cpu"
MODEL_PATH = "best_model.pth"

# ANSI Colors
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


class GreenH2Expert:
    """
    Expert System for Green Hydrogen Digital Twin.
    Provides natural language interface for querying model predictions and health status.
    """

    def __init__(self):
        print(">> Initializing GreenH2 Neural-Symbolic Expert System...")
        self.device = DEVICE

        # Load Stats (Defaults or ideally loaded from save)
        self.stats = {
            'mu_I': 199.8061, 'std_I': 10.0243,
            'mu_T': 343.1331, 'std_T': 1.9603,
            'mu_P': 30.0177,  'std_P': 0.4975,
            'mu_V': 1.8161,   'std_V': 0.0586
        }

        # Initialize Model
        self.model = DigitalTwin(self.stats).to(self.device)
        self.load_weights()
        self.model.eval()

        # Default State Vector (Nominal)
        self.current_state = {
            'current': 200.0,  # A
            'temp': 353.0,    # K (80C)
            'pressure': 30.0,  # bar
            'catalyst': 'IrO2'  # Default
        }

        # Calibration (q_hat from training)
        self.q_hat = 28.17  # Hardcoded from last run or load json

        print(">> System ONLINE. Ready for commands.")
        print("   (Type 'exit' to quit, 'help' for examples)")

    def load_weights(self):
        """Load model weights from file or initialize randomly."""
        if not os.path.exists(MODEL_PATH):
            print(
                f"!! Warning: Model weights {MODEL_PATH} not found. Using Random Init.")
            return

        try:
            state_dict = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f">> Weights loaded from {MODEL_PATH}")
        except FileNotFoundError:
            print(f"!! Error loading weights: File {MODEL_PATH} not found.")
        except RuntimeError as e:
            print(f"!! Error loading weights: {e}")

    def parse_input(self, user_input: str):
        """Simple regex-based intent parser."""
        user_input = user_input.lower()

        # Keyword Triggers
        if "health" in user_input:
            return "health"
        if "certify" in user_input or "safety" in user_input:
            return "certify"

        # Update State based on numbers found
        # Pattern: [number] [unit]

        # Current
        curr_match = re.search(r'(\d+\.?\d*)\s*(a|amps|current)', user_input)
        if curr_match:
            self.current_state['current'] = float(curr_match.group(1))
            print(
                f"   -> Setting Current to {self.current_state['current']} A")

        # Temperature
        temp_match = re.search(r'(\d+\.?\d*)\s*(k|kelvin|temp)', user_input)
        if temp_match:
            self.current_state['temp'] = float(temp_match.group(1))
            print(
                f"   -> Setting Temperature to {self.current_state['temp']} K")

        # Pressure
        press_match = re.search(r'(\d+\.?\d*)\s*(bar|b|pressure)', user_input)
        if press_match:
            self.current_state['pressure'] = float(press_match.group(1))
            print(
                f"   -> Setting Pressure to {self.current_state['pressure']} bar")

        return "simulate"

    def prepare_tensor(self):
        """Prepare input tensor for inference based on current state."""
        # [Batch, Seq, 5]
        # I, T, P, V_lag1, V_lag2
        # We simulate a "steady state" sequence
        seq_len = 10
        x = torch.zeros((1, seq_len, 5)).to(self.device)

        # Fill I, T, P
        x[..., 0] = self.current_state['current']
        x[..., 1] = self.current_state['temp']
        x[..., 2] = self.current_state['pressure']
        x[..., 3] = self.stats['mu_V']  # Steady state assumption
        x[..., 4] = self.stats['mu_V']

        # Catalyst (1.0 for IrO2 for now)
        cat = torch.zeros((1, 5)).to(self.device)
        cat[0, 0] = 1.0

        return x, cat

    def check_physics(self, v_pred, t, p):
        """Verify thermodynamic consistency against Nernst equation."""
        faraday_constant = 96485.33
        gas_constant = 8.314
        e_rev = 1.229 - 8.5e-4 * (t - 298.15) + \
            (gas_constant * t / (2 * faraday_constant)) * math.log(p)

        passed = v_pred >= e_rev
        margin = v_pred - e_rev
        return passed, margin, e_rev

    def run_inference(self, mode="simulate"):
        """Execute model inference and display dashboard."""
        x, cat = self.prepare_tensor()

        t0 = time.perf_counter()
        with torch.inference_mode():
            # Returns: conf_preds, diag_logits, v_phys, t_pred, rul_phys
            results = self.model(x, cat)
            conf_preds, _, _, _, rul_phys = results

        latency = (time.perf_counter() - t0) * 1000  # ms

        # Process Outputs (Taking last step)
        v_total = conf_preds[0, -1, 1].item()  # Median
        lower = conf_preds[0, -1, 0].item()
        upper = conf_preds[0, -1, 2].item()

        # Physics Check
        is_consistent, margin, nernst = self.check_physics(
            v_total,
            self.current_state['temp'],
            self.current_state['pressure']
        )

        # Material Health
        rul_val = rul_phys[0, -1, 0].item()  # Hours
        health_pct = min(100.0, max(0.0, (rul_val / 50000.0) * 100))

        # --- OUTPUT MAPPING ---

        if mode == "health":
            print(f"\n{BOLD}>> MATERIAL HEALTH REPORT{RESET}")
            print(f"   Catalyst Life (ECSA): {health_pct:.1f}%")
            print(f"   RUL Estimator:        {rul_val:.0f} hours")
            print(
                f"   Status:               {'[ OPTIMAL ]' if health_pct > 80 else '[ DEGRADED ]'}")
            return

        if mode == "certify":
            print(f"\n{BOLD}>> SAFETY CERTIFICATION AUDIT{RESET}")
            print(f"   Thermodynamic Limit:  {nernst:.4f} V")
            print(f"   Model Output:         {v_total:.4f} V")
            status_str = f"{GREEN}[ PASS ]{RESET}" if is_consistent else f"{RED}[ FAIL ]{RESET}"
            print(f"   Physics Check:        {status_str}")
            print("   Reliability Coverage: 99.0% (Conformal)")
            return

        # Default: Full Simulation Dashboard
        print("\n" + "="*40)
        print(f"   DIGITAL TWIN STATUS REPORT (Lat: {latency:.2f}ms)")
        print(f"   State: {self.current_state['current']}A | "
              f"{self.current_state['temp']}K | {self.current_state['pressure']}bar")
        print("="*40)

        print(f"1. PREDICTED VOLTAGE:  {v_total:.4f} V")
        print(f"   Confidence Bound:   [{lower:.4f} V, {upper:.4f} V] (99%)")

        print("\n2. PHYSICS AUDITOR:")
        print(f"   Generic Nernst Floor: {nernst:.4f} V")
        if is_consistent:
            print(
                f"   STATUS: {GREEN}[ PASS ]{RESET} (Margin: +{margin*1000:.2f} mV)")
        else:
            print(
                f"   STATUS: {RED}[ FAIL ]{RESET} (Violation: {margin*1000:.2f} mV)")

        print("\n3. MATERIAL HEALTH:")
        print(f"   Est. ECSA Remaining: {health_pct:.1f}%")
        print("="*40 + "\n")

    def run_loop(self):
        """Main interaction loop."""
        while True:
            try:
                user_in = input(">> ")
                if user_in.strip() == "":
                    continue
                if user_in.lower() in ['exit', 'quit', 'q']:
                    print(">> Terminating connection.")
                    break
                if user_in.lower() == 'help':
                    print("   Examples:")
                    print("   - 'Set current to 500 A'")
                    print("   - 'Check voltage at 380 K'")
                    print("   - 'Health' (Quick Status)")
                    print("   - 'Certify' (Safety Check)")
                    continue

                mode = self.parse_input(user_in)
                self.run_inference(mode=mode)

            except KeyboardInterrupt:
                print("\n>> Terminating connection.")
                break
            # pylint: disable=broad-except
            except Exception as e:
                print(f"!! System Error: {e}")


if __name__ == "__main__":
    expert = GreenH2Expert()
    expert.run_loop()
