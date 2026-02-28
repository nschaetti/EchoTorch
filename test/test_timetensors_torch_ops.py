# -*- coding: utf-8 -*-

import torch
import echotorch

from . import EchoTorchTestCase


class Test_TimeTensors_TorchOps(EchoTorchTestCase):
    def _sample(self):
        torch.manual_seed(7)
        return echotorch.rand(4, 3, length=6)

    def test_shape_ops_preserve_time_semantics(self):
        x = self._sample()
        y = self._sample()

        cat_out = torch.cat((x, y), dim=x.time_dim)
        self.assertTrue(isinstance(cat_out, echotorch.TimeTensor))
        self.assertEqual(cat_out.time_dim, 0)
        self.assertTensorSize(cat_out.tensor, [12, 4, 3])

        stack_out = torch.stack((x, y), dim=0)
        self.assertTrue(isinstance(stack_out, echotorch.TimeTensor))
        self.assertEqual(stack_out.time_dim, 1)
        self.assertTensorSize(stack_out.tensor, [2, 6, 4, 3])

        unsq = torch.unsqueeze(x, 0)
        sq = torch.squeeze(unsq, 0)
        self.assertEqual(unsq.time_dim, 1)
        self.assertEqual(sq.time_dim, 0)

        tr = torch.transpose(x, 0, 1)
        pm = torch.permute(x, (1, 0, 2))
        mv = torch.movedim(x, 0, 2)
        self.assertEqual(tr.time_dim, 1)
        self.assertEqual(pm.time_dim, 1)
        self.assertEqual(mv.time_dim, 2)

        flat = torch.flatten(x, start_dim=1)
        resh = torch.reshape(x, (6, 12))
        self.assertEqual(flat.time_dim, 0)
        self.assertEqual(resh.time_dim, 0)
        self.assertTensorSize(flat.tensor, [6, 12])
        self.assertTensorSize(resh.tensor, [6, 12])

    def test_indexing_tuple_ops(self):
        x = self._sample()

        idx = torch.index_select(x, x.time_dim, torch.tensor([0, 2, 4]))
        nar = torch.narrow(x, x.time_dim, 1, 3)
        self.assertEqual(idx.time_dim, 0)
        self.assertEqual(nar.time_dim, 0)
        self.assertTensorSize(idx.tensor, [3, 4, 3])
        self.assertTensorSize(nar.tensor, [3, 4, 3])

        chunks = torch.chunk(x, 3, dim=x.time_dim)
        splits = torch.split(x, 2, dim=x.time_dim)
        ub = torch.unbind(x, dim=1)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(splits), 3)
        self.assertEqual(len(ub), 4)
        self.assertTrue(all(isinstance(t, echotorch.TimeTensor) for t in chunks))
        self.assertTrue(all(isinstance(t, echotorch.TimeTensor) for t in splits))
        self.assertTrue(all(isinstance(t, echotorch.TimeTensor) for t in ub))
        self.assertEqual(chunks[0].time_dim, 0)
        self.assertEqual(splits[0].time_dim, 0)
        self.assertEqual(ub[0].time_dim, 0)

    def test_pointwise_and_linalg_numerical_parity(self):
        x = self._sample()
        y = self._sample()
        w = torch.randn(3, 2)

        relu_out = torch.relu(x)
        sig_out = torch.sigmoid(x)
        mm_out = torch.matmul(x, w)
        wh_out = torch.where(x.tensor > 0.5, x, y)

        self.assertTensorAlmostEqual(relu_out.tensor, torch.relu(x.tensor), 1e-7)
        self.assertTensorAlmostEqual(sig_out.tensor, torch.sigmoid(x.tensor), 1e-7)
        self.assertTensorAlmostEqual(mm_out.tensor, torch.matmul(x.tensor, w), 1e-7)
        self.assertTensorAlmostEqual(wh_out.tensor, torch.where(x.tensor > 0.5, x.tensor, y.tensor), 1e-7)

    def test_clone_and_detach(self):
        x = self._sample()

        c = torch.clone(x)
        d = torch.detach(x)
        self.assertTrue(isinstance(c, echotorch.TimeTensor))
        self.assertTrue(isinstance(d, echotorch.TimeTensor))
        self.assertEqual(c.time_dim, x.time_dim)
        self.assertEqual(d.time_dim, x.time_dim)
        self.assertTensorAlmostEqual(c.tensor, x.tensor, 1e-7)
        self.assertTensorAlmostEqual(d.tensor, x.tensor, 1e-7)
