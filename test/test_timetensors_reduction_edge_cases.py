# -*- coding: utf-8 -*-

import torch
import echotorch

from . import EchoTorchTestCase


class Test_TimeTensors_ReductionEdgeCases(EchoTorchTestCase):
    def _sample(self):
        torch.manual_seed(11)
        return echotorch.rand(4, 3, length=6)

    def test_max_min_tuple_outputs_follow_time_rules(self):
        x = self._sample()

        max_vals, max_idx = torch.max(x, dim=1)
        min_vals, min_idx = torch.min(x, dim=1)

        self.assertTrue(isinstance(max_vals, echotorch.TimeTensor))
        self.assertTrue(isinstance(max_idx, echotorch.TimeTensor))
        self.assertTrue(isinstance(min_vals, echotorch.TimeTensor))
        self.assertTrue(isinstance(min_idx, echotorch.TimeTensor))
        self.assertEqual(max_vals.time_dim, 0)
        self.assertEqual(max_idx.time_dim, 0)
        self.assertEqual(min_vals.time_dim, 0)
        self.assertEqual(min_idx.time_dim, 0)

        ref_max_vals, ref_max_idx = torch.max(x.tensor, dim=1)
        ref_min_vals, ref_min_idx = torch.min(x.tensor, dim=1)
        self.assertTensorAlmostEqual(max_vals.tensor, ref_max_vals, 1e-7)
        self.assertTensorEqual(max_idx.tensor, ref_max_idx)
        self.assertTensorAlmostEqual(min_vals.tensor, ref_min_vals, 1e-7)
        self.assertTensorEqual(min_idx.tensor, ref_min_idx)

    def test_time_dim_reduction_returns_tensor_unless_keepdim(self):
        x = self._sample()

        vals, idx = torch.max(x, dim=0)
        self.assertFalse(isinstance(vals, echotorch.TimeTensor))
        self.assertFalse(isinstance(idx, echotorch.TimeTensor))

        vals_k, idx_k = torch.max(x, dim=0, keepdim=True)
        self.assertTrue(isinstance(vals_k, echotorch.TimeTensor))
        self.assertTrue(isinstance(idx_k, echotorch.TimeTensor))
        self.assertEqual(vals_k.time_dim, 0)
        self.assertEqual(idx_k.time_dim, 0)
        self.assertTensorSize(vals_k.tensor, [1, 4, 3])
        self.assertTensorSize(idx_k.tensor, [1, 4, 3])

    def test_topk_and_kthvalue_keep_consistent_timetensor_outputs(self):
        x = self._sample()

        top_vals, top_idx = torch.topk(x, k=2, dim=1)
        kth_vals, kth_idx = torch.kthvalue(x, k=1, dim=1)

        self.assertTrue(isinstance(top_vals, echotorch.TimeTensor))
        self.assertTrue(isinstance(top_idx, echotorch.TimeTensor))
        self.assertTrue(isinstance(kth_vals, echotorch.TimeTensor))
        self.assertTrue(isinstance(kth_idx, echotorch.TimeTensor))
        self.assertEqual(top_vals.time_dim, 0)
        self.assertEqual(top_idx.time_dim, 0)
        self.assertEqual(kth_vals.time_dim, 0)
        self.assertEqual(kth_idx.time_dim, 0)

        ref_top_vals, ref_top_idx = torch.topk(x.tensor, k=2, dim=1)
        ref_kth_vals, ref_kth_idx = torch.kthvalue(x.tensor, k=1, dim=1)
        self.assertTensorAlmostEqual(top_vals.tensor, ref_top_vals, 1e-7)
        self.assertTensorEqual(top_idx.tensor, ref_top_idx)
        self.assertTensorAlmostEqual(kth_vals.tensor, ref_kth_vals, 1e-7)
        self.assertTensorEqual(kth_idx.tensor, ref_kth_idx)

    def test_tuple_and_negative_dim_reductions(self):
        x = self._sample()

        neg = torch.sum(x, dim=-1)
        self.assertTrue(isinstance(neg, echotorch.TimeTensor))
        self.assertEqual(neg.time_dim, 0)
        self.assertTensorSize(neg.tensor, [6, 4])
        self.assertTensorAlmostEqual(neg.tensor, torch.sum(x.tensor, dim=-1), 1e-7)

        multi_with_time = torch.sum(x, dim=(0, 1))
        self.assertFalse(isinstance(multi_with_time, echotorch.TimeTensor))
        self.assertTensorSize(multi_with_time, [3])
        self.assertTensorAlmostEqual(multi_with_time, torch.sum(x.tensor, dim=(0, 1)), 1e-7)

        multi_without_time = torch.sum(x, dim=(1, 2), keepdim=True)
        self.assertTrue(isinstance(multi_without_time, echotorch.TimeTensor))
        self.assertEqual(multi_without_time.time_dim, 0)
        self.assertTensorSize(multi_without_time.tensor, [6, 1, 1])
        self.assertTensorAlmostEqual(
            multi_without_time.tensor,
            torch.sum(x.tensor, dim=(1, 2), keepdim=True),
            1e-7,
        )
