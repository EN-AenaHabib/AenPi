"""
AenPi.urdu.green_metrics
-------------------------
Green AI metrics: measure CPU usage, RAM consumption, and execution time
for any Python function. Uses only psutil + time (no heavy frameworks).

Install: pip install psutil
"""

import time
import os

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


class GreenMetrics:
    """
    Context manager and utility class to measure resource usage of a code block.

    Tracks:
        - Execution time (seconds)
        - RAM usage before / after (MB)
        - CPU percent during execution
        - Estimated energy proxy score (lower is greener)

    Example:
        >>> with GreenMetrics() as gm:
        ...     result = some_heavy_function()
        >>> gm.report()
    """

    def __init__(self, label: str = "Task"):
        """
        Args:
            label (str): Human-readable label shown in the report.
        """
        self.label = label
        self.start_time = None
        self.end_time = None
        self.ram_before_mb = 0.0
        self.ram_after_mb = 0.0
        self.cpu_percent = 0.0
        self._process = None

    def _get_process(self):
        if _PSUTIL_AVAILABLE:
            return psutil.Process(os.getpid())
        return None

    def _get_ram_mb(self) -> float:
        if self._process:
            return self._process.memory_info().rss / (1024 ** 2)
        return 0.0

    def __enter__(self):
        self._process = self._get_process()
        self.ram_before_mb = self._get_ram_mb()
        if _PSUTIL_AVAILABLE:
            self._process.cpu_percent(interval=None)  # prime the CPU counter
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.ram_after_mb = self._get_ram_mb()
        if _PSUTIL_AVAILABLE and self._process:
            self.cpu_percent = self._process.cpu_percent(interval=None)

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def elapsed_seconds(self) -> float:
        """Total wall-clock execution time in seconds."""
        if self.start_time and self.end_time:
            return round(self.end_time - self.start_time, 4)
        return 0.0

    @property
    def ram_delta_mb(self) -> float:
        """RAM consumed during execution (MB)."""
        return round(self.ram_after_mb - self.ram_before_mb, 4)

    @property
    def green_score(self) -> float:
        """
        Heuristic Green Score (lower = greener / more efficient).
        Combines time and RAM delta with equal weight.
        Score = time_s * 0.5 + ram_delta_MB * 0.5
        """
        return round(self.elapsed_seconds * 0.5 + max(self.ram_delta_mb, 0) * 0.5, 4)

    # ── Report ────────────────────────────────────────────────────────────────

    def report(self, verbose: bool = True) -> dict:
        """
        Print and return a dictionary of all metrics.

        Args:
            verbose (bool): Print to console if True.

        Returns:
            dict: Keys: label, elapsed_seconds, ram_before_mb,
                        ram_after_mb, ram_delta_mb, cpu_percent, green_score
        """
        metrics = {
            "label":           self.label,
            "elapsed_seconds": self.elapsed_seconds,
            "ram_before_mb":   round(self.ram_before_mb, 2),
            "ram_after_mb":    round(self.ram_after_mb, 2),
            "ram_delta_mb":    self.ram_delta_mb,
            "cpu_percent":     round(self.cpu_percent, 2),
            "green_score":     self.green_score,
        }

        if verbose:
            psutil_note = "" if _PSUTIL_AVAILABLE else " (psutil not installed — RAM/CPU unavailable)"
            print(f"\n{'─'*45}")
            print(f"  🌿 AenPi Green Metrics: {self.label}{psutil_note}")
            print(f"{'─'*45}")
            print(f"  ⏱  Execution time : {metrics['elapsed_seconds']} s")
            print(f"  🧠 RAM before     : {metrics['ram_before_mb']} MB")
            print(f"  🧠 RAM after      : {metrics['ram_after_mb']} MB")
            print(f"  📈 RAM delta      : {metrics['ram_delta_mb']} MB")
            print(f"  💻 CPU usage      : {metrics['cpu_percent']} %")
            print(f"  🌱 Green score    : {metrics['green_score']} (lower = greener)")
            print(f"{'─'*45}\n")

        return metrics


# ─── Convenience function ─────────────────────────────────────────────────────

def green_metrics(func, *args, label: str = None, verbose: bool = True, **kwargs):
    """
    Measure Green AI metrics for a single function call.

    Args:
        func (callable): The function to measure.
        *args: Positional arguments passed to func.
        label (str): Optional label for the report. Defaults to func name.
        verbose (bool): Print metrics report. Default True.
        **kwargs: Keyword arguments passed to func.

    Returns:
        tuple: (function_result, metrics_dict)

    Example:
        >>> def my_func(text):
        ...     return text.split()
        >>> result, metrics = green_metrics(my_func, "میرا نام احمد ہے")
        >>> print(result)
        ['میرا', 'نام', 'احمد', 'ہے']
    """
    task_label = label or getattr(func, '__name__', 'function')

    with GreenMetrics(label=task_label) as gm:
        result = func(*args, **kwargs)

    report = gm.report(verbose=verbose)
    return result, report
