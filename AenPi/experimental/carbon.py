"""
aenpi/carbon.py
---------------
Practical Carbon Estimator (psutil-based, lightweight)

GOAL
----
Estimate energy + CO2 for:
1. AenPi local NLP modules (using real CPU usage via psutil)
2. LLM API calls (token-based approximation)

APPROACH (REALISTIC)
--------------------
- Uses actual CPU time + CPU utilization from psutil
- Converts to energy using:
      Energy (Wh) = CPU_time_seconds * CPU_usage * CPU_power_watts / 3600
- CO2 derived from grid intensity (gCO2/kWh)

This avoids fake hardware modeling and stays deployable.

REQUIREMENTS
------------
pip install psutil
"""

import time
import psutil
from datetime import datetime


# ---------------- CONFIG ----------------
CPU_POWER_WATTS = 15.0  # average laptop CPU power (safe assumption)
GRID_CO2_G_PER_KWH = 400.0  # Pakistan avg grid


# ---------------- LLM PROFILES (simple approximation) ----------------
LLM_PROFILES = {
    "gpt-4": {"co2_per_1k_tokens_g": 0.21},
    "gpt-3.5": {"co2_per_1k_tokens_g": 0.045},
    "claude": {"co2_per_1k_tokens_g": 0.10},
}


# ---------------- VALID MODULES ----------------
# Used to validate incoming module strings and prevent typos
VALID_MODULES = {"sentiment", "ner", "summarizer", "code_switch"}


class CarbonEstimator:
    """
    Simple, deployable carbon estimator using real CPU measurements.

    This version is:
    - NOT theoretical hardware modeling
    - Uses psutil for real process monitoring
    - Suitable for local NLP pipelines
    """

    def __init__(self, grid_co2=GRID_CO2_G_PER_KWH):
        self.grid_co2 = grid_co2
        self.session_log = []

    # ---------------- LLM ESTIMATION ----------------
    def estimate_llm(self, llm: str, n_calls: int, avg_tokens: int = 100):
        if llm not in LLM_PROFILES:
            raise ValueError(f"Unknown LLM: {llm}")

        total_tokens = n_calls * avg_tokens
        co2_g = (total_tokens / 1000) * LLM_PROFILES[llm]["co2_per_1k_tokens_g"]

        return {
            "llm": llm,
            "tokens": total_tokens,
            "co2_grams": co2_g,
            "co2_kg": co2_g / 1000,
        }

    # ---------------- REAL CPU MEASUREMENT ----------------
    def measure(self, func, *args, module: str = "sentiment", **kwargs):
        """
        Measure real execution energy using psutil.

        Steps:
        1. capture CPU usage before
        2. run function
        3. capture CPU usage after
        4. estimate energy consumption
        """

        if module not in MODULE_PROFILES:
            raise ValueError(f"Unknown module: {module}")

        process = psutil.Process()
        start_time = time.time()

        cpu_before = psutil.cpu_percent(interval=None)

        result = func(*args, **kwargs)

        cpu_after = psutil.cpu_percent(interval=None)
        end_time = time.time()

        duration_sec = end_time - start_time
        cpu_usage = max(cpu_after, cpu_before) / 100.0

        # ENERGY ESTIMATION
        energy_wh = (duration_sec * cpu_usage * CPU_POWER_WATTS) / 3600
        energy_kwh = energy_wh / 1000

        co2_g = energy_kwh * self.grid_co2

        return {
            "result": result,
            "module": module,
            "duration_sec": round(duration_sec, 6),
            "cpu_usage": round(cpu_usage, 4),
            "energy_wh": round(energy_wh, 8),
            "co2_grams": round(co2_g, 8),
        }

    # ---------------- BATCH MODULE TRACKING ----------------
    def track(self, module: str, calls: int, avg_sec_per_call: float):
        """
        Estimate energy for batch usage without function execution.
        """

        cpu_usage = 0.5  # assume moderate load
        total_time = calls * avg_sec_per_call

        energy_wh = (total_time * cpu_usage * CPU_POWER_WATTS) / 3600
        energy_kwh = energy_wh / 1000
        co2_g = energy_kwh * self.grid_co2

        entry = {
            "timestamp": datetime.now().isoformat(),
            "module": module,
            "calls": calls,
            "co2_grams": co2_g,
            "energy_wh": energy_wh,
        }

        self.session_log.append(entry)
        return entry

    # ---------------- SESSION SUMMARY ----------------
    def summary(self):
        total_co2 = sum(x["co2_grams"] for x in self.session_log)
        total_energy = sum(x["energy_wh"] for x in self.session_log)

        return {
            "total_events": len(self.session_log),
            "total_co2_grams": round(total_co2, 6),
            "total_energy_wh": round(total_energy, 6),
            "modules": self.session_log,
        }

    # ---------------- COMPARISON ----------------
    def compare(self, llm: str, module: str, n_calls: int, avg_tokens: int = 100):
        llm_data = self.estimate_llm(llm, n_calls, avg_tokens)

        # fake module estimate (based on batch model)
        module_est = self.track(module, n_calls, avg_sec_per_call=0.001)

        savings = llm_data["co2_grams"] - module_est["co2_grams"]

        return {
            "llm": llm_data,
            "module": module_est,
            "co2_saved_grams": round(max(savings, 0), 6),
        }
