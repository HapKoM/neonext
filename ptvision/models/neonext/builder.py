import os
from extension_builder.builder import CUDAOpBuilder


class NeoNeXtBuilder(CUDAOpBuilder):
    NAME = "neonext"
    SRC_ROOT = os.path.dirname(__file__)

    def __init__(self, name=None, src_root=None):
        name = self.NAME if name is None else name
        src_root = self.SRC_ROOT if src_root is None else src_root
        super().__init__(name=name, src_root=src_root)

    def absolute_name(self):
        return f"op_builder.{self.NAME}_op"

    def sources(self):
        return [
            "csrc/neocell.cpp"
        ]

    def include_paths(self):
        return []

    def nvcc_args(self):
        args = [
            "-O3",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ]

        return args + self.compute_capability_args()

    def cxx_args(self):
        return ["-O3", "-g", "-Wno-reorder"]
