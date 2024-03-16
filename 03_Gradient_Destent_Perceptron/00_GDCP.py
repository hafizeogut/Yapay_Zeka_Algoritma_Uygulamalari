# Initialize weights to arbitrary values.
w0 = 1
w1 = 1
w2 = 1

# Learning rate
alpha = 0.1

# Given data
X1 = [1, 1, 3]
X2 = [2, 2, 4]
y = [1, 1, -1]

# First iteration of repeat loop
for _ in range(2):
    # Iterate through the data points
    for i in range(len(X1)):
        # Calculate the weighted sum
        z = w0 + w1 * X1[i] + w2 * X2[i]
        
        # Predict y
        if z > 0:
            y_pred = 1
        else:
            y_pred = -1
        
        # Update weights
        w0 = w0 - alpha * (y_pred - y[i])
        w1 = w1 - alpha * (y_pred - y[i]) * X1[i]
        w2 = w2 - alpha * (y_pred - y[i]) * X2[i]

# Print updated weights
print("Updated weights after two iterations:")
print("w0 =", w0)
print("w1 =", w1)
print("w2 =", w2)
