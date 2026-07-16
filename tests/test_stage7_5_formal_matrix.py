import unittest

from benchmarks.run_stage7_5_formal_matrix import formal_matrix


class Stage75FormalMatrixTest(unittest.TestCase):
    def test_formal_matrix_freezes_precision_storage_and_cold_repeats(self):
        cells = formal_matrix()
        self.assertEqual(len(cells), 10)
        warm = [cell for cell in cells if cell.cache_state == "workload-warm"]
        cold = [cell for cell in cells if cell.cache_state == "model-cold"]
        self.assertEqual(len(warm), 7)
        self.assertEqual(len(cold), 3)
        self.assertTrue(all(cell.warmup == 1 and cell.runs == 3 for cell in warm))
        self.assertEqual({cell.replicate for cell in cold}, {0, 1, 2})
        self.assertEqual(
            {cell.expert_storage for cell in warm},
            {"bf16", "int8-reference", "int8-native"},
        )
        self.assertEqual(len({cell.cell_id for cell in cells}), len(cells))


if __name__ == "__main__":
    unittest.main()


# [Main Dev]
