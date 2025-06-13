import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from tensorflow.keras.utils import to_categorical

# 1. Prepare the Data (Replace this with your dataset)
# Let's generate a synthetic dataset for multiclass classification
# Generate a synthetic multiclass classification dataset
X, y = make_classification(
    n_samples=1000,        # Number of samples
    n_features=20,         # Number of features
    n_classes=3,           # Number of classes
    n_clusters_per_class=1,  # Reducing clusters per class
    n_informative=5,       # Increase number of informative features
    random_state=42
)


# One-hot encode the labels for multi-class classification
y = to_categorical(y, num_classes=3)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (important for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Build the Model
model = Sequential()

# Input layer (input shape matches the number of features in the data)
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

# Hidden layer 1
model.add(Dense(32, activation='relu'))

# Hidden layer 2
model.add(Dense(16, activation='relu'))

# Output layer (for multiclass classification, use softmax activation)
model.add(Dense(3, activation='softmax'))  # 3 classes in the output

# 3. Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',  # Categorical cross-entropy loss
              metrics=['accuracy'])

# 4. Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 5. Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 6. Predicting the class for new data
# Example: Predicting the class for a new data point
new_data = np.random.randn(1, X_train.shape[1])  # Random example
new_data_scaled = scaler.transform(new_data)  # Scale the new data point
prediction = model.predict(new_data_scaled)
predicted_class = np.argmax(prediction)  # Convert prediction probabilities to the class with highest probability
print(f"Predicted class: {predicted_class}")
