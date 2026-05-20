import unittest

import torch

from src.model_tiny import TinyISLRModel


class TinyISLRModelTest(unittest.TestCase):
    def test_forward_loss_and_backward(self) -> None:
        torch.manual_seed(123)
        x = torch.randn(4, 64, 708)
        mask = torch.ones(4, 64, dtype=torch.bool)
        y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        model = TinyISLRModel()
        logits = model(x, mask)

        self.assertEqual(tuple(logits.shape), (4, 250))
        self.assertFalse(torch.isnan(logits).any().item())

        loss = torch.nn.CrossEntropyLoss()(logits, y)
        self.assertFalse(torch.isnan(loss).any().item())
        loss.backward()

        self.assertTrue(any(param.grad is not None for param in model.parameters()))


if __name__ == "__main__":
    unittest.main()
