import threading
from typing import Any

import pynvml


BYTES_PER_MIB = 1024 * 1024


class GpuMemoryMonitor:
    """
    Sample GPU memory usage while an operation is running.

    The reported values are device-level GPU memory usage,
    not memory attributed exclusively to the Ollama process.
    """

    def __init__(
        self,
        device_index: int = 0,
        interval_seconds: float = 0.1,
    ) -> None:
        self.device_index = device_index
        self.interval_seconds = interval_seconds

        self._handle: Any = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.baseline_mb: float = 0.0
        self.peak_mb: float = 0.0

    def _read_used_mb(self) -> float:
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(
            self._handle
        )

        return memory_info.used / BYTES_PER_MIB

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(
            self.interval_seconds
        ):
            used_mb = self._read_used_mb()

            if used_mb > self.peak_mb:
                self.peak_mb = used_mb

    def __enter__(self) -> "GpuMemoryMonitor":
        pynvml.nvmlInit()

        self._handle = pynvml.nvmlDeviceGetHandleByIndex(
            self.device_index
        )

        self.baseline_mb = self._read_used_mb()
        self.peak_mb = self.baseline_mb

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._sample_loop,
            daemon=True,
        )

        self._thread.start()

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join()

        # Capture one final sample before shutting down NVML.
        final_mb = self._read_used_mb()

        if final_mb > self.peak_mb:
            self.peak_mb = final_mb

        pynvml.nvmlShutdown()

    @property
    def incremental_peak_mb(self) -> float:
        return max(
            0.0,
            self.peak_mb - self.baseline_mb,
        )

if __name__ == "__main__":
    import time

    with GpuMemoryMonitor() as gpu:
        time.sleep(1)

    print(
        "Baseline GPU memory (MiB):",
        round(gpu.baseline_mb, 2),
    )

    print(
        "Peak GPU memory used (MiB):",
        round(gpu.peak_mb, 2),
    )

    print(
        "Incremental peak (MiB):",
        round(gpu.incremental_peak_mb, 2),
    )