"""Sample host resource bounds at explicit training/cache checkpoints."""
import time


class TrainingBudget:
    def __init__(self, config):
        import psutil
        self.config = config
        self.process = psutil.Process()
        self.virtual_memory = psutil.virtual_memory
        self.started = time.monotonic()
        self.peak_rss_bytes = 0
        self.minimum_available_bytes = None

    def check(self):
        rss = self.process.memory_info().rss
        available = self.virtual_memory().available
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        self.minimum_available_bytes = min(self.minimum_available_bytes or available, available)
        if rss > self.config.max_process_rss_gb * 1e9:
            raise MemoryError("Training process RSS exceeded its configured budget")
        if available < self.config.min_available_ram_gb * 1e9:
            raise MemoryError("Insufficient host RAM for safe continued training")
        if time.monotonic() - self.started > self.config.max_total_runtime_seconds:
            raise TimeoutError("Training including preparation exceeded its total wall-clock budget")

    def report(self):
        return {"sampled_peak_process_rss_bytes": self.peak_rss_bytes,
                "sampled_minimum_host_available_bytes": self.minimum_available_bytes,
                "sampling_scope": "explicit preparation and optimization checkpoints; not an OS hard limit"}
