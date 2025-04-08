"""
Tests for rtorch.optim module functionality.
"""

import unittest
import numpy as np
import rtorch as torch
import rtorch.nn as nn
import rtorch.optim as optim

class TestOptim(unittest.TestCase):

    def test_sgd_step(self):
        # Create a simple parameter
        param = torch.tensor([10.0], requires_grad=True)
        optimizer = optim.SGD([param], lr=0.1)

        # Simulate loss and backward pass
        # loss = param * 2 # Example loss
        # loss.backward()
        # Manually set gradient for testability
        param.grad = torch.tensor([5.0]) # dloss/dparam = 5.0

        # Store original value
        original_value = param.numpy().copy()

        # Perform optimizer step
        optimizer.step()

        # Check new value: param = param - lr * grad = 10.0 - 0.1 * 5.0 = 9.5
        np.testing.assert_almost_equal(param.numpy(), [9.5], decimal=6)

        # Check gradient is still present (step doesn't zero it)
        self.assertIsNotNone(param.grad)

    def test_sgd_zero_grad(self):
        param = torch.tensor([10.0], requires_grad=True)
        param.grad = torch.tensor([5.0])
        optimizer = optim.SGD([param], lr=0.1)

        self.assertIsNotNone(param.grad)
        optimizer.zero_grad()
        # Check grad attribute directly if possible, or check internal state
        # Assuming zero_grad clears the Option<Tensor> or zeroes the data
        # Need a way to verify tensor.grad is None or zero after zero_grad call.
        # Let's assume param.grad reflects the internal state (might not if grad() clones)
        # A better check involves running backward again after zero_grad.
        loss = param * 2
        loss.backward() # Calculate gradient again (should be 2.0)
        self.assertIsNotNone(param.grad)
        np.testing.assert_array_almost_equal(param.grad.numpy(), [2.0]) # Check if zero_grad worked before backward

    def test_sgd_momentum(self):
        param = torch.tensor([10.0], requires_grad=True)
        optimizer = optim.SGD([param], lr=0.1, momentum=0.9)

        # Step 1
        param.grad = torch.tensor([5.0])
        optimizer.step()
        # buf = 0 * 0.9 + 5.0 * (1-0) = 5.0
        # param = 10.0 - 0.1 * 5.0 = 9.5
        np.testing.assert_almost_equal(param.numpy(), [9.5], decimal=6)

        # Step 2
        param.grad = torch.tensor([2.0]) # New gradient
        optimizer.step()
        # buf = 5.0 * 0.9 + 2.0 * (1-0) = 4.5 + 2.0 = 6.5
        # param = 9.5 - 0.1 * 6.5 = 9.5 - 0.65 = 8.85
        np.testing.assert_almost_equal(param.numpy(), [8.85], decimal=6)

    def test_adam_step(self):
        param = torch.tensor([1.0], requires_grad=True)
        optimizer = optim.Adam([param], lr=0.1)

        # Simulate loss and backward
        param.grad = torch.tensor([2.0])

        # Store original value
        original_value = param.numpy().copy()

        # Perform step
        optimizer.step()

        # Manual calculation (approximate, without precise bias correction check)
        # t=1, beta1=0.9, beta2=0.999, eps=1e-8, lr=0.1
        # m1 = (1-0.9)*2.0 = 0.2
        # v1 = (1-0.999)*2.0^2 = 0.001 * 4 = 0.004
        # bias_corr1 = 1 - 0.9^1 = 0.1
        # bias_corr2 = 1 - 0.999^1 = 0.001
        # m_hat = m1 / bias_corr1 = 0.2 / 0.1 = 2.0
        # v_hat = v1 / bias_corr2 = 0.004 / 0.001 = 4.0
        # update = lr * m_hat / (sqrt(v_hat) + eps) = 0.1 * 2.0 / (sqrt(4.0) + eps)
        # update = 0.2 / (2.0 + 1e-8) approx 0.1
        # new_param = 1.0 - update = 1.0 - 0.1 = 0.9
        self.assertTrue(np.abs(param.numpy()[0] - 0.9) < 1e-6)

    def test_optimizer_with_nn(self):
        # Simple linear regression test
        model = nn.Sequential(nn.Linear(1, 1))
        optimizer = optim.SGD(model.parameters().values(), lr=0.01) # Pass parameter values
        criterion = nn.functional.mse_loss

        # Dummy data
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        y_true = torch.tensor([[2.0], [4.0], [6.0], [8.0]]) # y = 2x

        initial_loss = float('inf')
        for _ in range(100): # Training loop
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y_true)
            if loss.numpy()[0] > initial_loss: # Basic check loss decreases
                print(f"Warning: Loss increased during optimization test ({initial_loss} -> {loss.numpy()[0]})")
            initial_loss = loss.numpy()[0]

            loss.backward()
            optimizer.step()

        # Check if loss decreased significantly
        self.assertTrue(initial_loss < 0.1) # Check loss threshold
        # Optionally check learned parameters
        # learned_w = list(model.parameters().values())[0].numpy() # Weight
        # learned_b = list(model.parameters().values())[1].numpy() # Bias
        # np.testing.assert_almost_equal(learned_w[0,0], 2.0, decimal=1)


if __name__ == '__main__':
    unittest.main()