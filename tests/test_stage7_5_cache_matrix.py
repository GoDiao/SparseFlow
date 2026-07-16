import unittest

from benchmarks.run_stage7_5_cache_matrix import calibration_matrix


class Stage75CacheMatrixTest(unittest.TestCase):
    def test_matrix_covers_budget_policy_state_and_io_boundaries(self):
        cells = calibration_matrix()
        self.assertEqual(len(cells), 28)
        core = [cell for cell in cells if cell.tag == "core"]
        self.assertEqual(len(core), 16)
        self.assertEqual(
            {
                (cell.variant, cell.cache_bytes // 1024**2)
                for cell in core
                if cell.variant != "C3-S0"
            },
            {
                (variant, budget)
                for variant in ("C3-S1", "C3-S3", "C3-S4")
                for budget in (512, 1024, 2048, 4096, 8192)
            },
        )
        self.assertEqual(sum(cell.tag == "cold" for cell in cells), 4)
        self.assertEqual(sum(cell.tag == "io" for cell in cells), 4)
        self.assertEqual(sum(cell.tag == "length" for cell in cells), 4)
        self.assertEqual(len({cell.cell_id for cell in cells}), len(cells))


if __name__ == "__main__":
    unittest.main()


# [Main Dev]
