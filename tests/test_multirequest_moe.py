import unittest

import torch

from sparseflow.multirequest_moe import combine_layer_rows


class MultiRequestMoeTest(unittest.TestCase):
    def test_rows_are_concatenated_in_session_order(self):
        records = [
            {
                "layers": {
                    0: {
                        "hidden_states": torch.tensor([[1.0, 2.0]]),
                        "selected_experts": torch.tensor([[1, 2]]),
                        "routing_weights": torch.tensor([[0.5, 0.5]]),
                    }
                }
            },
            {
                "layers": {
                    0: {
                        "hidden_states": torch.tensor([[3.0, 4.0]]),
                        "selected_experts": torch.tensor([[2, 3]]),
                        "routing_weights": torch.tensor([[0.25, 0.75]]),
                    }
                }
            },
        ]
        hidden, selected, routing = combine_layer_rows(records, 0, torch)
        self.assertTrue(torch.equal(hidden, torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
        self.assertTrue(torch.equal(selected, torch.tensor([[1, 2], [2, 3]])))
        self.assertTrue(torch.equal(routing, torch.tensor([[0.5, 0.5], [0.25, 0.75]])))

    def test_missing_layer_is_rejected(self):
        with self.assertRaises(ValueError):
            combine_layer_rows([{"layers": {}}], 0, torch)


if __name__ == "__main__":
    unittest.main()


# [Main Dev]
