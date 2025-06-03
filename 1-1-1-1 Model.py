import numpy as np

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    """Derivative of sigmoid (using activation value)"""
    return a * (1 - a)

def forward(input_value, weight, bias):
    """Generic forward pass through a layer"""
    pre_activation = weight * input_value + bias
    activation = sigmoid(pre_activation)
    return pre_activation, activation

def backward(error_from_next_layer, activation, input_to_this_layer, weight):
    """Generic backward pass through a layer"""
    # Through activation function
    error_before_activation = error_from_next_layer * sigmoid_derivative(activation)
    
    # Gradients for this layer's parameters
    gradient_weight = error_before_activation * input_to_this_layer
    gradient_bias = error_before_activation
    
    # Error to pass to previous layer
    error_to_previous_layer = error_before_activation * weight
    
    return error_before_activation, gradient_weight, gradient_bias, error_to_previous_layer

def backward_output(output, target, hidden_activation, weight_to_output):
    """Special case for output layer (computes loss gradient)"""
    # Loss gradient (assuming MSE loss)
    loss_gradient = output - target
    
    # Through output activation
    error_before_activation = loss_gradient * sigmoid_derivative(output)
    
    # Gradients for output layer parameters
    gradient_weight = error_before_activation * hidden_activation
    gradient_bias = error_before_activation
    
    # Error to pass back to hidden layer
    error_to_hidden = error_before_activation * weight_to_output
    
    return error_before_activation, gradient_weight, gradient_bias, error_to_hidden

def compute_loss(output, target):
    """Compute MSE loss"""
    return 0.5 * (output - target) ** 2

# Main
if __name__ == "__main__":
    # Network: input → hidden1 → hidden2 → output
    
    # Network parameters
    input_value = 2.0
    target_output = 1.0
    learning_rate = 0.1
    
    # Layer 1 (input → hidden1)
    weight_input_to_hidden1 = 0.5
    bias_hidden1 = 0.1
    
    # Layer 2 (hidden1 → hidden2)
    weight_hidden1_to_hidden2 = -0.3
    bias_hidden2 = 0.2
    
    # Output layer (hidden2 → output)
    weight_hidden2_to_output = -0.8
    bias_output = 0.15
    
    print("=== Neural Network Training with Weight Updates ===")
    print("Network: input → hidden1 → hidden2 → output")
    print(f"Learning rate: {learning_rate}")
    print()
    
    # Train for a few iterations to see learning in action
    for iteration in range(5):
        print(f"=== Iteration {iteration + 1} ===")
        
        print("Forward Pass:")
        
        # Input → Hidden1
        pre_activation_h1, activation_h1 = forward(input_value, weight_input_to_hidden1, bias_hidden1)
        print(f"  Input → Hidden1: {input_value:.3f} → {activation_h1:.3f}")
        
        # Hidden1 → Hidden2
        pre_activation_h2, activation_h2 = forward(activation_h1, weight_hidden1_to_hidden2, bias_hidden2)
        print(f"  Hidden1 → Hidden2: {activation_h1:.3f} → {activation_h2:.3f}")
        
        # Hidden2 → Output
        pre_activation_output, output = forward(activation_h2, weight_hidden2_to_output, bias_output)
        print(f"  Hidden2 → Output: {activation_h2:.3f} → {output:.3f}")
        
        # Compute loss
        loss = compute_loss(output, target_output)
        print(f"  Loss: {loss:.4f}")
        print()
        
        print("Backward Pass:")
        
        # Output layer
        error_output, grad_weight_h2_to_output, grad_bias_output, error_to_hidden2 = backward_output(
            output, target_output, activation_h2, weight_hidden2_to_output
        )
        print(f"  Output gradients: weight={grad_weight_h2_to_output:.4f}, bias={grad_bias_output:.4f}")
        
        # Hidden2 layer
        error_h2, grad_weight_h1_to_h2, grad_bias_h2, error_to_hidden1 = backward(
            error_to_hidden2, activation_h2, activation_h1, weight_hidden1_to_hidden2
        )
        print(f"  Hidden2 gradients: weight={grad_weight_h1_to_h2:.4f}, bias={grad_bias_h2:.4f}")
        
        # Hidden1 layer
        error_h1, grad_weight_input_to_h1, grad_bias_h1, error_to_input = backward(
            error_to_hidden1, activation_h1, input_value, weight_input_to_hidden1
        )
        print(f"  Hidden1 gradients: weight={grad_weight_input_to_h1:.4f}, bias={grad_bias_h1:.4f}")
        print()
        
        print("Weight Updates:")
        print("  Before updates:")
        print(f"    Input→Hidden1: weight={weight_input_to_hidden1:.4f}, bias={bias_hidden1:.4f}")
        print(f"    Hidden1→Hidden2: weight={weight_hidden1_to_hidden2:.4f}, bias={bias_hidden2:.4f}")
        print(f"    Hidden2→Output: weight={weight_hidden2_to_output:.4f}, bias={bias_output:.4f}")
        
        # UPDATE WEIGHTS AND BIASES - This is the learning step!
        weight_input_to_hidden1 = weight_input_to_hidden1 - learning_rate * grad_weight_input_to_h1
        bias_hidden1 = bias_hidden1 - learning_rate * grad_bias_h1
        
        weight_hidden1_to_hidden2 = weight_hidden1_to_hidden2 - learning_rate * grad_weight_h1_to_h2
        bias_hidden2 = bias_hidden2 - learning_rate * grad_bias_h2
        
        weight_hidden2_to_output = weight_hidden2_to_output - learning_rate * grad_weight_h2_to_output
        bias_output = bias_output - learning_rate * grad_bias_output
        
        print("  After updates:")
        print(f"    Input→Hidden1: weight={weight_input_to_hidden1:.4f}, bias={bias_hidden1:.4f}")
        print(f"    Hidden1→Hidden2: weight={weight_hidden1_to_hidden2:.4f}, bias={bias_hidden2:.4f}")
        print(f"    Hidden2→Output: weight={weight_hidden2_to_output:.4f}, bias={bias_output:.4f}")
        
        print(f"  Predicted: {output:.4f}, Target: {target_output:.4f}")
        print("-" * 50)
    
    print("\n=== Training Complete ===")
   
