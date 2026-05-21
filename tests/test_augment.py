import unittest

import torch

from src.augment import apply_augmentation


class AugmentationTest(unittest.TestCase):
    def test_disabled_returns_original_batch_objects_unchanged(self) -> None:
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        mask = torch.tensor([[True, True, False], [True, False, False]])

        x_aug, mask_aug = apply_augmentation(x, mask, {"enabled": False})

        self.assertTrue(torch.equal(x_aug, x))
        self.assertTrue(torch.equal(mask_aug, mask))

    def test_temporal_shift_keeps_shapes_and_padding_zero(self) -> None:
        torch.manual_seed(123)
        x = torch.ones((4, 6, 8), dtype=torch.float32)
        mask = torch.tensor(
            [
                [True, True, True, False, False, False],
                [True, True, True, True, False, False],
                [True, True, False, False, False, False],
                [True, True, True, True, True, False],
            ]
        )

        x_aug, mask_aug = apply_augmentation(
            x,
            mask,
            {
                "enabled": True,
                "temporal_shift_prob": 1.0,
                "temporal_shift_max": 2,
                "temporal_mask_prob": 0.0,
                "feature_dropout_prob": 0.0,
                "gaussian_noise_prob": 0.0,
            },
        )

        self.assertEqual(tuple(x_aug.shape), tuple(x.shape))
        self.assertEqual(tuple(mask_aug.shape), tuple(mask.shape))
        self.assertTrue(torch.all(x_aug[~mask_aug] == 0.0))

    def test_feature_dropout_zeros_some_feature_values_without_changing_mask(self) -> None:
        torch.manual_seed(123)
        x = torch.ones((3, 5, 12), dtype=torch.float32)
        mask = torch.ones((3, 5), dtype=torch.bool)

        x_aug, mask_aug = apply_augmentation(
            x,
            mask,
            {
                "enabled": True,
                "temporal_shift_prob": 0.0,
                "temporal_mask_prob": 0.0,
                "feature_dropout_prob": 1.0,
                "feature_dropout_width": 4,
                "gaussian_noise_prob": 0.0,
            },
        )

        self.assertTrue(torch.equal(mask_aug, mask))
        self.assertGreater(int((x_aug == 0.0).sum().item()), 0)
        self.assertTrue(torch.equal(x, torch.ones_like(x)))

    def test_noise_keeps_padding_and_existing_zero_values_zero(self) -> None:
        torch.manual_seed(123)
        x = torch.ones((1, 4, 6), dtype=torch.float32)
        x[:, 1, :] = 0.0
        mask = torch.tensor([[True, True, False, False]])

        x_aug, mask_aug = apply_augmentation(
            x,
            mask,
            {
                "enabled": True,
                "temporal_shift_prob": 0.0,
                "temporal_mask_prob": 0.0,
                "feature_dropout_prob": 0.0,
                "gaussian_noise_prob": 1.0,
                "gaussian_noise_std": 0.015,
            },
        )

        self.assertTrue(torch.equal(mask_aug, mask))
        self.assertTrue(torch.all(x_aug[:, 1:, :] == 0.0))
        self.assertFalse(torch.equal(x_aug[:, 0, :], x[:, 0, :]))


if __name__ == "__main__":
    unittest.main()
