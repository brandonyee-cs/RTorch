#!/usr/bin/env python
"""
Simple neural network example using RTorch
This example creates a small network to classify the XOR problem.
"""

import numpy as np
import rtorch as torch
import rtorch.nn as nn
import rtorch.optim as optim

def main():
    """Create and train a simple neural network to learn XOR"""
    print("RTorch Simple Neural Network Example - XOR Problem")
    
    # Create input data (XOR problem)
    x = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    
    # Create target output
    y = torch.tensor([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ])
    
    # Define model
    model = nn.Sequential(
        nn.Linear(2, 4),  # Input features: 2, Hidden units: 4
        nn.ReLU(),
        nn.Linear(4, 1)   # Output: 1
    )
    
    # Loss function
    mse_loss = nn.functional.mse_loss
    
    # Optimizer
    optimizer = optim.SGD(model.parameters().values(), lr=0.1)
    
    # Train the model
    print("\nTraining...")
    for epoch in range(1000):
        # Forward pass
        optimizer.zero_grad()
        output = model(x)
        loss = mse_loss(output, y)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy()[0]:.4f}")
    
    # Test the model
    print("\nTesting...")
    output = model(x)
    print("Input (x1, x2) -> Output (prediction):")
    for i in range(4):
        x1, x2 = x.numpy()[i]
        pred = output.numpy()[i][0]
        actual = y.numpy()[i][0]
        print(f"{x1:.0f}, {x2:.0f} -> {pred:.4f} (actual: {actual:.0f})")
    
    return 0

if __name__ == "__main__":
    main()