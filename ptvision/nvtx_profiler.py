import torch
from torch.cuda import nvtx


class NvtxProfiler:
    def __init__(self, steps=20, warmup=1000, sync=False, enabled=False):
        self.steps = steps
        self.warmup = warmup
        self.sync = sync
        self.enabled = enabled
        self.iter = 0

        self._running = False

    def start_profiler(self):
        if not self.enabled or self._running:
            return

        if self.iter == self.warmup:
            if self.sync:
                torch.cuda.synchronize()
            self._running = True
            torch.cuda.cudart().cudaProfilerStart()

    def stop_profiler(self):
        if not self.enabled or not self._running:
            return

        if self.iter == self.warmup + self.steps:
            torch.cuda.cudart().cudaProfilerStop()

    def range_push(self, name):
        if not self.enabled or not self._running:
            return

        if self.iter >= self.warmup and self.iter <= self.warmup + self.steps:
            if self.sync:
                torch.cuda.synchronize()
            nvtx.range_push(name)

    def range_pop(self):
        if not self.enabled or not self._running:
            return

        if self.iter >= self.warmup and self.iter <= self.warmup + self.steps:
            nvtx.range_pop()
            if self.sync:
                torch.cuda.synchronize()

    def step(self, iter=None):
        if not self.enabled:
            return

        if iter is not None:
            self.iter = iter
        else:
            self.iter += 1


__NVTX_PROFILER__ = None


def get_nvtx_profiler(cfg=None):
    global __NVTX_PROFILER__
    if __NVTX_PROFILER__ is None:
        if cfg is None:
            __NVTX_PROFILER__ = NvtxProfiler(enabled=False)
        else:
            print("Initialized NvtxProfiler")
            __NVTX_PROFILER__ = NvtxProfiler(
                steps=cfg.profile_steps,
                warmup=cfg.profile_warmup,
                sync=cfg.profile_sync,
                enabled=bool(cfg.profile_nvtx)
            )

    return __NVTX_PROFILER__
