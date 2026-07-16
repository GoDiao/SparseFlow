import shutil
import unittest
from pathlib import Path

import torch

from sparseflow.native_int8 import load_native_int8, reference_dynamic_linear


def has_native_requirements() -> bool:
    cpuinfo = Path("/proc/cpuinfo")
    return (
        cpuinfo.is_file()
        and "avx512_vnni" in cpuinfo.read_text(encoding="utf-8")
        and shutil.which("g++") is not None
        and shutil.which("ninja") is not None
    )


@unittest.skipUnless(has_native_requirements(), "AVX-512 VNNI build environment required")
class NativeInt8Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_native_int8()

    def test_dynamic_linear_matches_scalar_quantization_reference(self):
        generator = torch.Generator().manual_seed(1234)
        input_tensor = torch.randn(3, 64, generator=generator)
        weight = torch.randint(-127, 128, (16, 64), generator=generator, dtype=torch.int8)
        scales = torch.rand(16, generator=generator) * 0.01 + 0.001
        row_sums = torch.ops.sparseflow_native.row_sums(weight)

        native = torch.ops.sparseflow_native.dynamic_linear(
            input_tensor, weight, scales, row_sums
        )
        reference = reference_dynamic_linear(
            input_tensor, weight, scales, row_sums, torch
        )
        self.assertTrue(torch.allclose(native, reference, atol=1e-5, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()


# [Main Dev]
