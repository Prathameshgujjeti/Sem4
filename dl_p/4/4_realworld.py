import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import RandomFlip, RandomRotation, Conv2D, MaxPooling2D, Input, UpSampling2D, concatenate

# Load Oxford Pets dataset with segmentation masks
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
train_data = dataset['train'].map(self.preprocess).batch(16).prefetch(1)
test_data = dataset['test'].map(self.preprocess).batch(16).prefetch(1)

def preprocess(data):
    """Preprocess images and masks, resize and normalize"""
    image = tf.image.resize(data['image'], (128, 128))
    image = image / 255.0  # Normalize to [0,1]
    mask = tf.image.resize(data['segmentation_mask'], (128, 128), method='nearest')
    mask = tf.where(mask == 1, 1.0, 0.0)  # Convert to binary mask (pet vs background)
    return image, mask

# Data augmentation layer
data_augmentation = tf.keras.Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.2),
])

def unet_model():
    """U-Net architecture for image segmentation"""
    inputs = Input(shape=(128, 128, 3))
    
    # Contracting Path
    x = data_augmentation(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(x)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)
    
    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)
    
    # Bottleneck
    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    
    # Expanding Path
    u4 = UpSampling2D()(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(128, 3, activation='relu', padding='same')(u4)
    c4 = Conv2D(128, 3, activation='relu', padding='same')(c4)
    
    u5 = UpSampling2D()(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(64, 3, activation='relu', padding='same')(u5)
    c5 = Conv2D(64, 3, activation='relu', padding='same')(c5)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(c5)
    return tf.keras.Model(inputs, outputs)

model = unet_model()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])

# Train the model
history = model.fit(train_data, epochs=10, validation_data=test_data)

# Evaluate on test set
test_loss, test_acc, test_iou = model.evaluate(test_data)
print(f'Test Mean IoU: {test_iou:.2f}')