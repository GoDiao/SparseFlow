import unittest

from benchmarks.run_stage7_5_quality_matrix import BACKENDS, LEVELS, cells


class Stage75QualityMatrixTest(unittest.TestCase):
    def test_quality_matrix_crosses_frozen_levels_and_backends(self):
        matrix = cells()
        self.assertEqual(len(matrix), 12)
        self.assertEqual(set(matrix), {(level, backend) for level in LEVELS for backend in BACKENDS})


if __name__ == "__main__":
    unittest.main()


# [Main Dev]
