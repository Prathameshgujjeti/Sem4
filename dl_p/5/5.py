# Write a program to predict a caption for a sample image using LSTM
#
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
# Sample data: list of captions for images
captions = [
    "A cat sitting on a mat",
    "A dog playing with a ball",
    "A bird flying in the sky",
    "A fish swimming in water",
    "A child playing with toys"
]
# Corresponding images (dummy data, in practice use actual image data)
images = np.random.rand(len(captions), 64, 64, 3)  # Dummy images of size 64x64 with 3 channels
# Tokenize the captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size
# Convert captions to sequences
sequences = tokenizer.texts_to_sequences(captions)
# Pad sequences to ensure uniform length
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
## Prepare input and output data for LSTM (use captions only)
X = padded_sequences[:, :-1]  # Input: all words except last
y = padded_sequences[:, 1:]   # Output: all words except first

# Convert output to categorical (one-hot encoding)
y_categorical = to_categorical(y, num_classes=vocab_size)

# Define the LSTM model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=max_length-1))
model.add(LSTM(256, return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_categorical, epochs=10, batch_size=2)
# Predict captions for new images
def predict_caption(image):
    # Preprocess the image (dummy preprocessing here)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    # Predict the caption sequence
    predicted_sequence = model.predict(image)
    # Convert predicted sequence to words
    predicted_words = [tokenizer.index_word[np.argmax(word)] for word in predicted_sequence[0]]
    return ' '.join(predicted_words).strip()
# Test the prediction function with a sample image
sample_image = np.random.rand(64, 64, 3)  # Dummy sample image
predicted_caption = predict_caption(sample_image)
print("Predicted Caption:", predicted_caption)
