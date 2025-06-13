# Write a program to implement deep learning Techniques for image segmentation. 

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
import numpy as np

# Define a simple U-Net model for image segmentation
def unet_model(input_size=(128, 128, 3)):
    inputs = Input(input_size)  # Input layer

    # Downsampling (encoder) path
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)

    # Upsampling (decoder) path
    up1 = UpSampling2D(size=(2, 2))(conv3)
    merge1 = concatenate([up1, conv2])
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    merge2 = concatenate([up2, conv1])
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge2)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

    # Output layer with sigmoid activation for binary segmentation
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Generate synthetic training data: 10 images of size 128x128 with 3 channels
    X_train = np.random.rand(10, 128, 128, 3)
    # Generate synthetic binary masks: 10 masks of size 128x128 with 1 channel
    y_train = np.random.randint(0, 2, (10, 128, 128, 1))

    # Initialize and compile the U-Net model
    model = unet_model()
    model.summary()  # Print model architecture

    # Train the model on synthetic data
    model.fit(X_train, y_train, epochs=3, batch_size=2)

    # Predict segmentation on a couple of synthetic images
    preds = model.predict(X_train[:2])
    print("Predicted segmentation shape:", preds.shape)
