'''
Program to solve the XOR problem.
'''
import tensorflow as tf
import numpy as np

# Define XOR input and output data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Build a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=500, verbose=0)  # Training for 500 epochs

# Evaluate the model
predictions = model.predict(X)
print("Predictions:", np.round(predictions).flatten())

'''
Here's a well-structured TensorFlow implementation to solve the XOR problem using a simple neural network. The model consists of an input layer, one hidden layer with non-linearity (ReLU), and an output layer.

### Explanation:
1. **Define XOR Data**: Inputs (`X`) and expected outputs (`y`).
2. **Build the Model**:
   - Input layer: 2 neurons.
   - Hidden layer: 4 neurons with ReLU activation.
   - Output layer: 1 neuron with Sigmoid activation for binary classification.
3. **Compile the Model**: Using Adam optimizer and Binary Cross-Entropy loss.
4. **Train the Model**: Running for 500 epochs in silent mode (`verbose=0`).
5. **Evaluate & Predict**: Print rounded predictions for XOR inputs.

This model should learn to correctly classify XOR outputs. Let me know if you need modifications! 🚀

May I know why are training the model for 500 epochs? 
Great question! The choice of **500 epochs** is based on empirical observation and the nature of the XOR problem. Let me explain step by step.  

### 1️⃣ **Why Many Epochs?**
- The **XOR problem is non-linearly separable**, meaning a simple linear model cannot solve it.
- A **small neural network (like ours)** needs enough training iterations to properly learn the pattern.
- With **too few epochs**, the model might not converge to the correct solution.
- 500 epochs is generally **sufficient** for a small dataset like XOR.

### 2️⃣ **Why Not More or Fewer?**
| Epochs  | Result |
|----------|--------|
| **Too Few (<100)** | Model may not fully learn XOR patterns (low accuracy). |
| **Around 500** | Generally achieves ~100% accuracy. |
| **Too Many (>1000)** | Could lead to **overfitting**, wasting computation time. |

### 3️⃣ **How to Determine the Right Epochs?**
Instead of fixing 500, we can:
- **Use Early Stopping**: Stop training when accuracy stops improving.
- **Plot Loss Curve**: If loss stabilizes before 500, we can reduce the epochs.

### 4️⃣ **Alternative: Early Stopping in TensorFlow**
If you want the model to stop automatically when it converges, you can use:
```python
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model.fit(X, y, epochs=1000, callbacks=[early_stopping], verbose=0)
```
- `monitor='loss'`: Watches training loss.
- `patience=10`: Stops if no improvement for 10 consecutive epochs.

Would you like me to modify the code to include early stopping? 😊🚀
'''
