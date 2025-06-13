import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Create a toy dataset: Square footage vs Housing prices
np.random.seed(42)  # For reproducibility
X = np.array([500, 1000, 1500, 2000, 2500, 3000], dtype=np.float32)  # Square footage
y = np.array([150000, 200000, 250000, 300000, 350000, 400000], dtype=np.float32)  # Prices in USD

# Reshape X for the model input
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# 2. Define the model: A simple linear regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_dim=1)  # Linear layer with 1 input and 1 output (slope and intercept)
])

# 3. Compile the model with Mean Squared Error (MSE) loss and Adam optimizer
model.compile(optimizer='adam', loss='mse')

# 4. Train the model
history = model.fit(X, y, epochs=1000, verbose=0)

# 5. Plot the loss function over training epochs
plt.plot(history.history['loss'])
plt.title('Training Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# 6. Visualize the learned linear relationship
plt.scatter(X, y, color='blue', label='Data Points')
predicted_y = model.predict(X)
plt.plot(X, predicted_y, color='red', label='Fitted Line (Learned)')
plt.title('Housing Prices vs Square Footage')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.legend()
plt.show()

# 7. Make predictions on new data points
new_data = np.array([1200, 1800, 2200], dtype=np.float32).reshape(-1, 1)  # New square footage values
predictions = model.predict(new_data)

print(f"Predictions for new data points (Square Footage): {new_data.flatten()}")
print(f"Predicted Prices: {predictions.flatten()}")
