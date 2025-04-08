"""
Tests for rtorch.Tensor functionality.
"""

import unittest
import numpy as np
import rtorch as torch # Use alias similar to pytorch

class TestTensor(unittest.TestCase):

    def test_creation_from_list(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        t = torch.tensor(data)
        self.assertIsInstance(t, torch.Tensor)
        self.assertEqual(t.shape, (2, 2))
        np.testing.assert_array_almost_equal(t.numpy(), np.array(data, dtype=np.float32))

    def test_creation_from_numpy(self):
        data_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = torch.tensor(data_np)
        self.assertIsInstance(t, torch.Tensor)
        self.assertEqual(t.shape, (2, 2))
        # Check if data is copied or shared (should be copied based on current impl)
        data_np[0, 0] = 99.0
        self.assertNotEqual(t.numpy()[0, 0], 99.0)
        np.testing.assert_array_almost_equal(t.numpy(), [[1.0, 2.0], [3.0, 4.0]])

    def test_requires_grad(self):
        t1 = torch.tensor([1.0], requires_grad=False)
        t2 = torch.tensor([1.0], requires_grad=True)
        self.assertFalse(t1.requires_grad)
        self.assertTrue(t2.requires_grad)

    def test_zeros_ones(self):
        t_zeros = torch.zeros(2, 3)
        t_ones = torch.ones(4, 1)
        self.assertEqual(t_zeros.shape, (2, 3))
        self.assertEqual(t_ones.shape, (4, 1))
        np.testing.assert_array_equal(t_zeros.numpy(), np.zeros((2, 3), dtype=np.float32))
        np.testing.assert_array_equal(t_ones.numpy(), np.ones((4, 1), dtype=np.float32))

    def test_rand_randn(self):
        # Check if feature is available at runtime (optional)
        try:
            t_rand = torch.rand(5, 5)
            t_randn = torch.randn(3, 2)
            self.assertEqual(t_rand.shape, (5, 5))
            self.assertEqual(t_randn.shape, (3, 2))
            # Check values are roughly in expected range (basic check)
            self.assertTrue(np.all(t_rand.numpy() >= 0.0) and np.all(t_rand.numpy() <= 1.0))
        except RuntimeError as e:
            # Handle case where rand feature wasn't compiled
            if "compiled with the 'rand' feature" in str(e):
                self.skipTest("RTorch not compiled with 'rand' feature.")
            else:
                raise e # Re-raise unexpected runtime errors

    def test_basic_ops(self):
        t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        t2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

        # Add
        t_add = t1 + t2
        np.testing.assert_array_almost_equal(t_add.numpy(), [[6.0, 8.0], [10.0, 12.0]])
        t_add_m = t1.add(t2) # Method call
        np.testing.assert_array_almost_equal(t_add_m.numpy(), [[6.0, 8.0], [10.0, 12.0]])

        # Sub
        t_sub = t1 - t2
        np.testing.assert_array_almost_equal(t_sub.numpy(), [[-4.0, -4.0], [-4.0, -4.0]])

        # Mul
        t_mul = t1 * t2
        np.testing.assert_array_almost_equal(t_mul.numpy(), [[5.0, 12.0], [21.0, 32.0]])

        # Matmul
        t_matmul = t1.matmul(t2) # Assuming T2 is compatible - needs reshape maybe?
        t2_T_like = torch.tensor([[5.0, 7.0], [6.0, 8.0]]) # t2 transposed equivalent
        t_matmul_corr = t1.matmul(t2_T_like)
        expected_matmul = np.array([[1.0, 2.0], [3.0, 4.0]]) @ np.array([[5.0, 7.0], [6.0, 8.0]])
        np.testing.assert_array_almost_equal(t_matmul_corr.numpy(), expected_matmul)

    def test_autograd_simple_add(self):
        a = torch.tensor([2.0], requires_grad=True)
        b = torch.tensor([3.0], requires_grad=True)
        c = a + b # c = 5.0
        c.backward()

        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)
        np.testing.assert_array_almost_equal(a.grad.numpy(), [1.0]) # dc/da = 1
        np.testing.assert_array_almost_equal(b.grad.numpy(), [1.0]) # dc/db = 1

    def test_autograd_simple_mul(self):
        a = torch.tensor([2.0], requires_grad=True)
        b = torch.tensor([3.0], requires_grad=True)
        c = a * b # c = 6.0
        c.backward()

        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)
        np.testing.assert_array_almost_equal(a.grad.numpy(), [3.0]) # dc/da = b = 3
        np.testing.assert_array_almost_equal(b.grad.numpy(), [2.0]) # dc/db = a = 2

    def test_autograd_chain(self):
        a = torch.tensor([2.0], requires_grad=True)
        b = torch.tensor([3.0], requires_grad=True)
        c = a * b # c = 6
        d = torch.tensor([4.0], requires_grad=True)
        e = c + d # e = 10
        e.backward()

        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)
        self.assertIsNotNone(d.grad)
        self.assertIsNone(c.grad) # Intermediate non-leaf node

        # de/da = de/dc * dc/da = 1 * b = 3
        np.testing.assert_array_almost_equal(a.grad.numpy(), [3.0])
        # de/db = de/dc * dc/db = 1 * a = 2
        np.testing.assert_array_almost_equal(b.grad.numpy(), [2.0])
        # de/dd = 1
        np.testing.assert_array_almost_equal(d.grad.numpy(), [1.0])

    # TODO: Add tests for broadcasting
    # TODO: Add tests for more ops (reshape, sum, mean, etc.)
    # TODO: Add tests for detach
    # TODO: Add tests for gradient accumulation

if __name__ == '__main__':
    unittest.main()