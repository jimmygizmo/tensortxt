#! /usr/bin/env python
# git@github.com:jimmygizmo/tensortxt/tf-imdb-binary-sentiment.py
# Version 1.0.0

import matplotlib.pyplot as plt
from pathlib import Path
import os
import re
import shutil
import string
import tensorflow as tf
from keras import layers
from keras import losses
import pprint

# Program purpose: Train a binary classifier to perform sentiment analysis on an IMDB dataset.
# We will train a sentiment-analysis model to classify movie reviews as positive or negative.
# Stanford Large Movie Review data set: https://ai.stanford.edu/%7Eamaas/data/sentiment/
# Dataset: Text of 50,000 movie reviews from IMDB. 25,000 reviews for training and 25,000 reviews for testing.
# The training and testing sets are balanced, meaning they contain an equal number of positive and negative reviews.
# This is a Python program based off the Collab code for the following tutorial. It has been adapted to run in a
# standard Python environment and enhanced for clarity, learning and reusability.
# https://www.tensorflow.org/tutorials/keras/text_classification

WORKSPACE_DIRECTORY = "."
DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

pp = pprint.PrettyPrinter(indent=4)


def log(msg):
    print(f"\n[####]    {msg}")


def log_phase(msg):
    print(f"\n\n[####]    ----  {msg}  ----\n")


log_phase(f"PROJECT:  TEXT SENTIMENT ANALYSIS MODEL TRAINING & PREDICTION - IMDB MOVIE REVIEWS")
log(f"Tensorflow version: {tf.__version__}  -  Keras version: {tf.keras.__version__}")

log_phase(f"PHASE 1:  Download raw data. Inspect directory format and a sample text file.")

workspace_dir = Path(WORKSPACE_DIRECTORY)
log(f"workspace_dir: {workspace_dir}")

expected_dataset_dir = Path(workspace_dir) / "dataset-imdb"
log(f"expected_dataset_dir: {expected_dataset_dir}")

untar_dir_says_get_file = None  # This will be set via tf.keras.utils.get_file()

if expected_dataset_dir.exists() and expected_dataset_dir.is_dir():  # TODO: Seems like .exists() is redundant.
    log(f"* Raw dataset will not be downloaded. It appears you have already downloaded it.")
    untar_dir_says_get_file = expected_dataset_dir
else:
    log(f"Downloading raw dataset .tar.gz file from: {DATASET_URL}")
    # Generate a tf.data.Dataset object from text files in a directory.
    # https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file
    ignore_this_returned_path = tf.keras.utils.get_file(
        origin=DATASET_URL,
        untar=True,
        cache_dir=workspace_dir,
        cache_subdir=expected_dataset_dir
    )
    log(f"* * * * * * DEBUG * * * * * *: returned_dataset_dir: {untar_dir_says_get_file}")
    # NOTE: cache_dir will default to "~/.keras" if not specified.


dataset_dir = untar_dir_says_get_file

training_dir = Path(dataset_dir) / "train"
log(f"training_dir: {training_dir}")

testing_dir = Path(dataset_dir) / "test"
log(f"testing_dir: {testing_dir}")


# Cleanup of /unsup/ dir which is specific to the IMDB project only.
# It is in its own conditional block in this location for clarity and cleaner code.
if expected_dataset_dir.exists() and expected_dataset_dir.is_dir():
    dataset_dir = Path(untar_dir_says_get_file)
    log(f"Removing unsupported data from the raw dataset.")
    remove_dir = os.path.join(training_dir, "unsup")
    shutil.rmtree(remove_dir)


log(f"Lets have a quick look at the directory structures and one of the text files:")

log(f"dataset_dir listing:\n{pp.pformat(list(dataset_dir.iterdir()))}")

log(f"training_dir listing:\n{pp.pformat(list(training_dir.iterdir()))}")

log(f"testing_dir listing:\n{pp.pformat(list(testing_dir.iterdir()))}")

# TODO: This filename should not be hardcoded. At least move it to a CONFIG var.
# sample_file = os.path.join(training_dir, "pos/1181_9.txt")
sample_file = Path(training_dir) / "pos/1181_9.txt"
log(f"sample file: {sample_file}")

with open(sample_file) as f:
    print(f.read())

print("""
Expected directory structure:

main_directory/
...class_a/
......a_text_1.txt
......a_text_2.txt
...class_b/
......b_text_1.txt
......b_text_2.txt
""")

log_phase(f"PHASE 2:  Create datasets.")

log(f"Create RAW TRAINING DATASET from directory of text files.")

batch_size = 32
seed = 42

# https://www.tensorflow.org/api_docs/python/tf/keras/utils/text_dataset_from_directory
raw_training_dataset = tf.keras.utils.text_dataset_from_directory(
    training_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=seed
)

# TODO: Is this even giving more than one?
log(f"Sample of 5:")
for text_batch, label_batch in raw_training_dataset.take(1):
    for i in range(5):
        print("Review", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])

print("Label 0 corresponds to", raw_training_dataset.class_names[0])
print("Label 1 corresponds to", raw_training_dataset.class_names[1])

log(f"Create RAW VALIDATION DATASET from directory of text files")

# Uses data from the /train/ subdir because we are letting Keras manage the 0.2 splitting of train vs/ validation data.
# In other cases, the validation data might already be in a separate directory, but not in this case. This is the
# purpose of the validation_split option.
# TODO: Change to using a named argument for training_dir
raw_validation_dataset = tf.keras.utils.text_dataset_from_directory(
    training_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=seed
)

log(f"Create RAW TEST DATASET from directory of text files")

# TODO: Change to using a named argument for testing_dir
raw_test_dataset = tf.keras.utils.text_dataset_from_directory(
    testing_dir,
    batch_size=batch_size
)


log_phase(f"PHASE 3:  Prepare datasets. Custom standardization, tokenization, vectorization.")
# 1. Lower-case, strip <br /> tags and regex-style-escape punctuation. (standardization)
# 2. split up words (tokenization)
# 3. convert to numbers (vectorization)


# This custom_standardization() function appears to be a callback which will be applied to each item of input text
# individually.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, "[%s]" % re.escape(string.punctuation), "")


max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length
)

# Make a text-only dataset (without labels), then call adapt
training_text = raw_training_dataset.map(lambda x, y: x)
vectorize_layer.adapt(training_text)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


log(f"Example of the vectorized data:")

# retrieve a batch (of 32 reviews and labels) from the dataset
# ASSUMING next(iter()) will get us 32. It's up to the object's __next__ as to how many we get.
text_batch, label_batch = next(iter(raw_training_dataset))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_training_dataset.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

# Disabling these because the index changes on each run. Would need to pick the indices from the preceding output
# of the vectorized text.
# print("1787 ---> ", vectorize_layer.get_vocabulary()[1787])
# print("2601 ---> ", vectorize_layer.get_vocabulary()[2601])
print("Vocabulary size: {}".format(len(vectorize_layer.get_vocabulary())))

training_dataset = raw_training_dataset.map(vectorize_text)
validation_dataset = raw_validation_dataset.map(vectorize_text)
test_dataset = raw_test_dataset.map(vectorize_text)

# AUTOTUNE, .cache() and .prefetch() are all for maximizing performance.
AUTOTUNE = tf.data.AUTOTUNE
training_dataset = training_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

log(f"Creating the model/neural-network.")

embedding_dim = 16

model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)
])

log(f"Model summary:")
model.summary()

log(f"Compile the INITIAL MODEL.  Binary Sentiment Analysis.")
log(f"SPECS: losses.BinaryCrossentropy(from_logits=True), metrics=tf.metrics.BinaryAccuracy(threshold=0.0)")
model.compile(
    loss=losses.BinaryCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=tf.metrics.BinaryAccuracy(threshold=0.0)
)

log(f"Fit the INITIAL MODEL. Fitting the model is the primary and most intensive part of training.")
epochs = 10
history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=epochs
)

log(f"Evaluate/test the model.")
loss, accuracy = model.evaluate(test_dataset)

print("Loss: ", loss)
print("Accuracy: ", accuracy)


log_phase(f"PHASE 4:  Epoch history plots.")

history_dict = history.history
history_dict.keys()

acc = history_dict["binary_accuracy"]
val_acc = history_dict["val_binary_accuracy"]
loss = history_dict["loss"]
val_loss = history_dict["val_loss"]

epochs = range(1, len(acc) + 1)


log(f"PLOT: Training Loss, Validation Loss")

# "bo" is for "blue dot"
plt.plot(epochs, loss, "bo", label="Training loss")
# b is for "solid blue line"
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()


log(f"PLOT: Training Accuracy, Validation Accuracy")

plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")

plt.show()


log_phase(f"PHASE 5: Export and test phase. Test the exported model using RAW TEST DATASET.")

# TODO: Can this step be considered the 'export' or is the actual export a side-effect of compile, below?
#   Is there a physical binary somewhere?
log(f"tf.keras.Sequential groups a linear stack of layers into a tf.keras.Model.")
log(f"vectorize_layer passed in, model passed in, layers.Activation: sigmoid. Obj name: export_model.")
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation("sigmoid")
])


log(f"Compile the EXPORT MODEL.  Binary Sentiment Analysis.")
log(f"SPECS: losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy']")
# TODO: NOTE: logits setting is True for the compile of INITIAL MODEL and False for the compile of the EXPORT MODEL.
# TODO: LEARN ABOUT THIS LOGITS SETTING THOROUGHLY, DOCUMENT IT IN THIS PROJECT. IS THIS A GENERAL PATTERN?
export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False),
    optimizer="adam",
    metrics=["accuracy"]
)

log(f"EVALUATE: export_model.evaluate raw_test_dataset")
# Test it with `raw_test_dataset`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_dataset)
log(f"Accuracy:\n{accuracy}")


log_phase(f"PHASE 6:  Prediction. Running the exported model.")

examples = [
    "The movie was great!",
    "The movie was okay.",
    "The movie was terrible.",
    "I actually liked this film.",
    "Hated it, hated it, HATED IT!",
    "My kid makes better movies than this."
]

log(f"Predict. Show array of review text phrases and corresponding array of model-predicted scores.")
log(f"0 is most negative. 1 is most positive. Less than 0.5 is negative. 0.5 or greater is positive.")
result = export_model.predict(examples)

log(f"Array of examples:\n{pp.pformat(examples)}")
log(f"Array of corresponding model-predicted binary sentiment scores 0 -> 1  neg -> pos:\n{pp.pformat(result)}")

log_phase(f"PROJECT:  SENTIMENT ANALYSIS TENSORFLOW/KERAS DEMONSTRATION COMPLETE.  Exiting.")

