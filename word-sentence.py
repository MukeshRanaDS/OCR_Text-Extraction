import re
import os
import numpy as np
import pandas as pd
from config import *
from os import listdir
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

# DataSet Splitting
words_list = []

# Open the file and read line by line
with open(base_path, 'r') as labels:
    for label in labels:
        # Check if the line is empty or starts with "#"
        if label[0] == '#':
            continue
        # Check if the line contains "err"
        if 'err' in label:
            continue
        # If the line passes the above conditions, split it by whitespace and extract the transcription
        words_list.append(label.rstrip())

    # Print the list of valid transcriptions
words_list = words_list[:-1]


def data_split(split_size=0.95):
    split_idx = int(split_size * len(words_list))
    train_samples = words_list[:split_idx]
    validation_samples = words_list[split_idx:]

    assert len(words_list) == len(train_samples) + len(validation_samples)
    return train_samples, validation_samples


def get_image_paths_and_labels(samples):
    paths = []
    corrected_samples = []
    for (i, file_line) in enumerate(samples):
        line_split = file_line.strip()
        line_split = line_split.split(" ")

        # Each line split will have this format for the corresponding image:
        # part1/part1-part2/part1-part2-part3.png
        image_name = line_split[0]
        partI = image_name.split("-")[0]
        partII = image_name.split("-")[1]
        img_path = os.path.join(
            source_path, partI, partI + "-" + partII, image_name + ".png"
        )
        if os.path.getsize(img_path):
            paths.append(img_path)
            corrected_samples.append(file_line.split("\n")[0])

    return paths, corrected_samples


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def clean_labels(labels):
    cleaned_labels = []
    for label in labels:
        label = label.split(" ")[-1].strip()
        cleaned_labels.append(label)
    return cleaned_labels


def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image, paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image


def preprocess_image(image_path, img_size = (image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label


def process_images_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=AUTOTUNE
    )

    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)


def process_images_2(image_path):
    image = preprocess_image(image_path)
    # label = vectorize_label(label)
    return {"image": image}


def prepare_test_images(image_paths):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths)).map(
        process_images_2, num_parallel_calls=AUTOTUNE
    )
    # return dataset
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)


class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred


def build_model():
    input_img = keras.Input(shape=(image_width, image_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))

    # first conv block
    x = keras.layers.Conv2D(
        32, (3, 3), activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1"
    )(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = keras.layers.Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal",
        padding="same",
        name="Conv2"
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have two maxpool layers with pool size and strides 2
    # Hence downsampled feature maps are 4x smaller the number of filters in the last layer is 64,
    # Reshape accordingly before passing the output to the RNN part of the model.

    new_shape = ((image_width // 4), (image_height // 4) * 64)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)

    # RNN
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, dropout=0.25)
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
    )(x)
    # +2 is to account for the two special tokens introduced by the CTC loss.
    # The recommendation comes here: https://git.10/J0eXP.
    x = keras.layers.Dense(
        len(char_to_num.get_vocabulary()) + 2, activation="softmax", name="dense2"
    )(x)
    # Add CTC layer for calculating CTC Loss at each step.
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model.
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="handwriting_recognizer"
    )

    # optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model


def calculate_edit_distance(labels, predictions):
    # Get a single batch and convert its labels to sparse tensors.
    sparse_labels = tf.cast(tf.sparse.from_dense(labels), dtype=tf.int64)

    # Make predictions and convert them to sparse tensors.
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    predictions_decoded = keras.backend.ctc_decode(
        predictions, input_length=input_len, greedy=True
    )[0][0][:, :max_len]
    sparse_predictions = tf.cast(
        tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
    )

    # Compute individual edit distances and average them out.
    edit_distances = tf.edit_distance(
        sparse_predictions, sparse_labels, normalize=False
    )
    return tf.reduce_mean(edit_distances)


class EditDistanceCallback(keras.callbacks.Callback):
    def __init__(self, pred_model):
        super().__init__()
        self.prediction_model = pred_model

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []

        for i in range(len(validation_images)):
            labels = validation_labels[i]
            predictions = self.prediction_model.predict(validation_images[i])
            edit_distances.append(calculate_edit_distance(labels, predictions).numpy())
        print(f"Mean eidt distance for each {epoch + 1}: {np.mean(edit_distances): .4f}")


# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_len]

    # Iterate over the results and get back the text.
    output_text = []

    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_chars(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text



# Splitting the data into train, validation and test set
train_samples, validation_samples = data_split()
print(f"Total training samples: {len(train_samples)}")
print(f"Total validation samples: {len(validation_samples)}")

# Data Input PipeLine
train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
validation_img_paths, validation_labels = get_image_paths_and_labels(validation_samples)
print(train_img_paths[:10])
print(train_labels[:10])


print(extracted_image_path)
t_images = []
# Lopping all the Test Image
for f in listdir(extracted_image_path):
    t_images_path = os.path.join(extracted_image_path, f)
    t_images.append(t_images_path)
print(t_images[:10])

# Sequencing the Extracted Words
t_images.sort(key=natural_keys)
print(t_images[:10])
# Train Images
print(train_img_paths[0:10])
# Train Labels
print(train_labels[0: 10])


# find maximum length and the size of the vocabulary in the training data.
train_labels_cleaned = []
characters = set()
max_len = 0
for label in train_labels:
    label = label.split(" ")[-1].strip()
    for char in label:
        characters.add(char)
    max_len = max(max_len, len(label))
    train_labels_cleaned.append(label)

# Check Maximum len and Vocab size.
print("Maximum length: ", max_len)
print("Vocab size: ", len(characters))
# Check some label samples
print(f"train_labels_cleaned :\n {train_labels_cleaned[:10]}")

# Cleaning label of Validation and test set
validation_labels_cleaned = clean_labels(validation_labels)

# Print Cleaned Validation and Test labels
print(f"validation_labels_cleaned :\n {validation_labels_cleaned[:10]}")

ff = list(characters)
print(f"characters : {ff}")
# Missing Characters: ['}', ']', '$', '+', '>', '%', '_', '=', '[', ' ', '{', '@', '<', '&', '^', '|']

## Building Character Vocabulary
AUTOTUNE = tf.data.AUTOTUNE

# Mapping characters to integers
char_to_num = StringLookup(vocabulary=ff, mask_token=None)
# Mapping integers back to original characters
num_to_chars = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)


# Data Preparation
train_ds = prepare_dataset(train_img_paths, train_labels_cleaned)
validation_ds = prepare_dataset(validation_img_paths, validation_labels_cleaned)


# Visualize test images
inf_images = prepare_test_images(t_images)
np_config.enable_numpy_behavior()

for data in inf_images.take(1):
    images = data["image"]
    _, ax = plt.subplots(4, 4, figsize=(15, 8))

    for i in range(16):
        img = images[i]
        # print(img.shape)
        img = tf.image.flip_left_right(img)
        img = tf.transpose(img, perm=[1, 0, 2])
        img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
        img = img[:, :, 0]

        # Gather indices where Label!= padding token

        ax[i // 4, i % 4].imshow(img, cmap="gray")

    plt.show()


# Get the model
model = build_model()
print(model.summary())

# Evaluation Matrix
validation_images = []
validation_labels = []

for batch in validation_ds:
    validation_images.append(batch["image"])
    validation_labels.append(batch["label"])

# Training
model = build_model()
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
edit_distance_callback = EditDistanceCallback(prediction_model)


# Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=filepath,
    monitor='val_loss',
    save_best_only=False,
    save_weights_only=False,
    mode='auto',
    verbose=1
)

# Train the model with the ModelCheckpoint callback
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epochs,
    callbacks=[edit_distance_callback, checkpoint_callback],
)

# Convert the history dictionary to a pandas DataFrame
history_df = pd.DataFrame(history.history)

# Save the history DataFrame to an Excel file
history_df.to_excel(history_filepath, index=False)

# Prediction on images of Test set
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)

# Inference on New set of images
pred_test_text = []

# Let's check results on some test samples.
for batch in inf_images.take(1):
    batch_images = batch["image"]
    print(batch_images.shape)

    _, ax = plt.subplots(4, 4, figsize=(15, 8))

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)
    pred_test_text.append(pred_texts)

    for i in range(16):
        img = batch_images[i]
        img = tf.image.flip_left_right(img)
        img = tf.transpose(img, perm=[1, 0, 2])
        img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
        img = img[:, :, 0]

        title = f"Prediction: {pred_texts[i]}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")

    plt.show()

flat_list = [item for sublist in pred_test_text for item in sublist]
print(flat_list)

sentence = ' '.join(flat_list)
print(sentence)