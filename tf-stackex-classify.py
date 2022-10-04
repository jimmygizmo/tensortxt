#! /usr/bin/env python
# git@github.com:jimmygizmo/tensortxt/tf-stackex-classify.py
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

# multi-class classification on Stack Overflow questions
# train a multi-class classifier to predict the tag of a programming question on Stack Overflow.

# Program purpose: Train a multi-class classifier to predict the tag of a programming question on Stack Overflow.
# We will train a xxxxxxxxxxx model to classify xxxxxx as xxxxxxxxxxxxxxx.
# xxxxxxxxx data set: https://ai.stanford.edu/%7Eamaas/data/sentiment/
# Dataset: A small subset of thousands of questions out of the 1.7 million post BigQuery version:
# https://console.cloud.google.com/marketplace/details/stack-exchange/stack-overflow?pli=1&project=atomonova01
# Occurrences of the language/category words have been replaced with "blank" in the data to increase the challenge
# since many questions do contain that string. The challenge would not be interesting if those were left in.
# This is a variation of the following tutorial, described at the end. This code was inspired by the original tutorial
# code which was intended for Collab. I am only making a standalone version of this challenge and not a Collab version.
# The code in this repository has evolved quite a bit from the original.
# https://www.tensorflow.org/tutorials/keras/text_classification

WORKSPACE_DIRECTORY = "."
DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"

pp = pprint.PrettyPrinter(indent=4)


def log(msg):
    print(f"\n[####]    {msg}")


def log_phase(msg):
    print(f"\n\n[####]    ----  {msg}  ----\n")


# TODO: IMPLEMENTING SAVING AND LOADING OF MODELS. FOLLOWING THIS:
# https://www.tensorflow.org/tutorials/keras/save_and_load
# https://www.tensorflow.org/tutorials/keras/save_and_load#save_checkpoints_during_training
# TODO: Doing a POC in a separate script then will adapt to this one. See: tf-save-load.py

log_phase(f"PROJECT:  xxxxxxxxx MODEL TRAINING & PREDICTION - xxxxxxx")
log(f"Tensorflow version: {tf.__version__}  -  Keras version: {tf.keras.__version__}")

log_phase(f"PHASE 1:  Download raw data. Inspect directory format and a sample text file.")

workspace_dir = Path(WORKSPACE_DIRECTORY)
log(f"workspace_dir: {workspace_dir}")

dataset_dir = Path(workspace_dir) / "aclImdb"
log(f"dataset_dir: {dataset_dir}")

training_dir = Path(dataset_dir) / "train"
log(f"training_dir: {training_dir}")

testing_dir = Path(dataset_dir) / "test"
log(f"testing_dir: {testing_dir}")

if dataset_dir.exists() and dataset_dir.is_dir():  # TODO: Seems like .exists() is redundant.
    log(f"* Raw dataset will not be downloaded. It appears you have already downloaded it.")
else:
    log(f"Downloading raw dataset .tar.gz file from: {DATASET_URL}")
    # Generate a tf.data.Dataset object from text files in a directory.
    # https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file
    returned_dataset_dir = tf.keras.utils.get_file("aclImdb",
                                                   DATASET_URL,
                                                   untar=True,
                                                   cache_dir='.',
                                                   cache_subdir=''
                                                   )
    # TODO: REALLY need to clarify what object is returned. We are treating it as a string a lot but
    #   later, similar ones are obviously much more.
    # NOT-A-BUG, but we got lucky. Here we might just be turning a dataset object into a Path object but we get
    # away with because in this one case, we don't use the dataset for anything else except downloading and
    # untarring? I think our path operations work on it possibly because the dataset object must have a .__string__
    # method (or equiv) so that Path() can coerce it into a Path, and then it works as a dir path, but we could no
    # longer use it's many dataset features (I suspect.) This all needs to be understood. The conundrum sort of
    # stems from how we had to move variables around in order to have the conditional download, and because we are
    # using Path() now and not just os.path/strings which in this case forced us to change to object, when otherwise
    # we could have used dataset.__string__ in effect, perhaps.
    # My point is .. I am trying out Path() and considering abndoning os.path except for high performance tight loops.
    # At this point I am not sure I like using Path() and it might cause more problems than it is worth.
    # My reaction about overriding the / operator for this is that I don't like it. I don't know if I will come to
    # view this idea as good. It seems like it is best to leave the division operator / alone. This is not solving
    # some critical problem and I don't really like the look of the cute (bad) syntax.
    # The jury is still out on pathlib. Maybe if we could do something other than the / override for the simple
    # concatenation tasks.


    dataset_dir = Path(returned_dataset_dir)
    # NOTE: cache_dir will default to "~/.keras" if not specified.
    log(f"Removing unsupported data from the raw dataset.")
    remove_dir = os.path.join(training_dir, "unsup")
    shutil.rmtree(remove_dir)

log(f"Lets have a quick look at the directory structures and one of the text files:")

log(f"dataset_dir listing:\n{pp.pformat(list(dataset_dir.iterdir()))}")

log(f"training_dir listing:\n{pp.pformat(list(training_dir.iterdir()))}")

log(f"testing_dir listing:\n{pp.pformat(list(testing_dir.iterdir()))}")


# sample_file = os.path.join(training_dir, "pos/1181_9.txt")
sample_file = Path(training_dir) / "pos/1181_9.txt"
log(f"sample file: {sample_file}")

with open(sample_file) as f:
    print(f.read())

print("""
Expected directory structure:

train/
...python/
......0.txt
......1.txt
...javascript/
......0.txt
......1.txt
...csharp/
......0.txt
......1.txt
...java/
......0.txt
......1.txt
""")

log_phase(f"PHASE 2:  Create datasets.")

log(f"Create RAW TRAINING DATASET from directory of text files.")

batch_size = 32
seed = 42

# https://www.tensorflow.org/api_docs/python/tf/keras/utils/text_dataset_from_directory
raw_training_dataset = tf.keras.utils.text_dataset_from_directory(
    "aclImdb/train",
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
raw_validation_dataset = tf.keras.utils.text_dataset_from_directory(
    training_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=seed
)

log(f"Create RAW TEST DATASET from directory of text files")

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

# NOTE regarding .Dense(XXXX) change when going from binary sentiment with 2 classes to 4 class tagging/classification.
# For binary sentiment analysis done in this project (tf-imdb-binary-sentiment.py): Dense(1)
# For the current classification project with 4 classes instead of just 2: Dense(4)
model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(4)
])

log(f"Model summary:")
model.summary()

# NOTE on compilation changes going from 2 class binary sentiment analysis to 4 class classification:
# When compiling the model, change the loss to tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True).
# This is the correct loss function to use for a multi-class classification problem, when the labels for each
# class are integers (in this case, they can be 0, 1, 2, or 3). In addition,
# change the metrics to metrics=['accuracy'], since this is a multi-class classification problem
# (tf.metrics.BinaryAccuracy is only used for binary classifiers).

log(f"Compile the model.")
model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=['accuracy']
)

log(f"Fit the model. Fitting the model is the primary and most intensive part of training.")
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
# NOTE about change from binary to 4 classes: When plotting accuracy over time,
# change binary_accuracy and val_binary_accuracy to accuracy and val_accuracy, respectively.

history_dict = history.history
history_dict.keys()

acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
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

# NOTE - Changes not exactly specified in the end of page instructions for this variant.
# We had to do this above, but obviously these changes are also needed down here in PHASE 5 for
# multi-category classification, vs. binary.
# Use: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# Use: metrics=["accuracy"]

log(f"Compile the EXPORT MODEL. Multi-Category Classification.")
# TODO: THIS IS EXPORT MODEL - NOT SURE WE WANT LOGITS = TRUE. MIGHT WANT LOGITS = FALSE FOR EXPORTS. I'm learning.
#   IT IS FALSE FOR THE BINARY EXPORT, WHILE TRUE FOR BINARY INITIAL. IF SAME PATTERN APPLIES, IT WILL BE FALSE HERE.
log(f"SPECS: losses.SparseCategoricalCrossentropy(from_logits=????????), metrics=['accuracy']")
# TODO: LEARN ABOUT THIS LOGITS SETTING THOROUGHLY, DOCUMENT IT IN THIS PROJECT.
export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
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

