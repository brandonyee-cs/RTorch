"""
Tests for rtorch.nn module functionality.
"""

import unittest
import numpy as np
import rtorch as torch
import rtorch.nn as nn

class TestNN(unittest.TestCase):

    def test_linear_layer(self):
        layer = nn.Linear(5, 2) # in=5, out=2
        self.assertIsInstance(layer, nn.Linear)
        self.assertEqual(layer.weight.shape, (2, 5))
        self.assertIsNotNone(layer.bias)
        self.assertEqual(layer.bias.shape, (2,))

        # Check parameter count
        params = layer.parameters()
        self.assertEqual(len(params), 2)
        self.assertTrue("weight" in params)
        self.assertTrue("bias" in params)

        # Forward pass
        input_tensor = torch.randn(3, 5) # batch=3, features=5
        output = layer(input_tensor) # Use __call__
        self.assertEqual(output.shape, (3, 2)) # batch=3, out_features=2

        # Check requires_grad propagation
        self.assertTrue(output.requires_grad)

        # Check autograd through layer
        output_sum = output.sum()
        output_sum.backward()

        self.assertIsNotNone(layer.weight.grad)
        self.assertIsNotNone(layer.bias.grad)
        self.assertEqual(layer.weight.grad.shape, layer.weight.shape)
        self.assertEqual(layer.bias.grad.shape, layer.bias.shape)

    def test_relu_layer(self):
        layer = nn.ReLU()
        input_tensor = torch.tensor([[-1.0, 2.0], [0.0, -4.0]])
        output = layer(input_tensor)
        expected_output = np.array([[0.0, 2.0], [0.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(output.numpy(), expected_output)
        self.assertEqual(len(layer.parameters()), 0) # No parameters

        # Check autograd
        input_tensor.requires_grad = True
        output = layer(input_tensor)
        output_sum = output.sum()
        output_sum.backward()
        expected_grad = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        self.assertIsNotNone(input_tensor.grad)
        np.testing.assert_array_almost_equal(input_tensor.grad.numpy(), expected_grad)

    def test_sequential_container(self):
        # Note: This relies on the Python implementation of Sequential,
        # as the Rust one had FFI issues.
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        input_tensor = torch.randn(4, 10) # batch=4, features=10
        output = model(input_tensor)
        self.assertEqual(output.shape, (4, 1))

        # TODO: Test parameter access (if implemented for PySequential)
        # TODO: Test autograd through sequential

    def test_mse_loss_functional(self):
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.1, 1.9], [3.5, 4.2]])
        loss = nn.functional.mse_loss(pred, target)
        self.assertTrue(loss.is_scalar()) # Should be scalar result
        expected_loss = np.mean((pred.numpy() - target.numpy())**2)
        np.testing.assert_almost_equal(loss.numpy()[0], expected_loss, decimal=6)

    # def test_cross_entropy_loss_functional(self):
        # Requires integer target tensor support
        # logits = torch.randn(4, 5, requires_grad=True) # batch=4, classes=5
        # target_indices = ??? # Need integer tensor [4]
        # loss = nn.functional.cross_entropy_loss(logits, target_indices)
        # self.assertTrue(loss.is_scalar())
        # TODO: Add backward check

    # TODO: Add tests for dropout, other activations, etc.

if __name__ == '__main__':
    unittest.main()