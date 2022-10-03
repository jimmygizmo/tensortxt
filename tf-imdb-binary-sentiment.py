#! /usr/bin/env python

import matplotlib.pyplot as plt
from pathlib import Path
import os
import re
import shutil
import string
import tensorflow as tf
from keras import layers
from keras import losses

print(tf.__version__)

# Program purpose: Train a binary classifier to perform sentiment analysis on an IMDB dataset.


def log(msg):
    print(f"\n[########]    {msg}")


# TODO: IMPLEMENTING SAVING AND LOADING OF MODELS. FOLLOWING THIS:
# https://www.tensorflow.org/tutorials/keras/save_and_load
# https://www.tensorflow.org/tutorials/keras/save_and_load#save_checkpoints_during_training
# TODO: Doing a POC in a separate script then will adapt to this one. See: tf-save-load.py


url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
log(f"getting dataset .tar.gz file from: {url}")

# https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file
dataset = tf.keras.utils.get_file("aclImdb_v1",
                                  url,
                                  untar=True,
                                  cache_dir='.',
                                  cache_subdir=''
                                  )
# NOTE: cache_dir will default to "~/.keras". TODO: Clarify why this is called "cache" since at the moment it looks
#                                                   more like a "download dir". Clarify exact purpose of "cache.dir".

workspace_dir = os.path.dirname(dataset)
# There is an implication here that the object/variable "dataset" is either a string (path to dir of dataset structure)
# or has a .__string__() taking effect here in string context from os.path. which returns such.
# Also, This returns ".", the value that was passed to cache_dir. So "cache_dir" might have multiple purposes.

# TODO: Hypothesis, workspace_dir might always be the current working directory. Depends on tf.keras.utils.get_file()
# but this is likely. Would be good to positively clarify. Scripts might not always be invoked from the assumed
# CWD and/or workspace_dir. Some assumptions on relative locations should eventually be pinned down to standard and
# consistent locations for more robust code.
# UPDATE: Notice that cache_dir is specified as "." and this is also the value of workspace_dir.
log(f"workspace_dir: {workspace_dir}")

dataset_dir = os.path.join(workspace_dir, 'aclImdb')
log(f"dataset_dir: {dataset_dir}")

log(f"Just have a quick look at the dir structure and then one of the files.")
log(f"dataset_dir listing:\n\n{os.listdir(dataset_dir)}")

train_dir = os.path.join(dataset_dir, 'train')
log(f"train_dir listing:\n\n{os.listdir(train_dir)}")


sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
log(f"sample file: {sample_file}\n")

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

# This will error on the second try because no such dir will be found.
# Can do this only once after unpacking source data.
log(f"Removing unsupported data from the raw dataset")
# TODO: Add an existence check or use a lib that doesn't care. (pathlib?)
# Ideally we would download the data set once, remove this stuff once and then do training repeatedly. The original
# code was a one-time Collab from TF docs, and it is still being adapted to something more reusable and robust.
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)


log(f"RAW TRAINING\n")

batch_size = 32
seed = 42

# https://www.tensorflow.org/api_docs/python/tf/keras/utils/text_dataset_from_directory
raw_training_dataset = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

print(f"#####]    Sample of 5:\n")
for text_batch, label_batch in raw_training_dataset.take(1):
    for i in range(5):
        print("Review", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])

print("Label 0 corresponds to", raw_training_dataset.class_names[0])
print("Label 1 corresponds to", raw_training_dataset.class_names[1])

log(f"raw validation dataset from directory\n")

# Uses data from the /train/ subdir because we are letting Keras manage the 0.2 splitting of train vs/ validation data.
# In other cases, the validation data might already be in a separate directory, but not in this case. This is the
# purpose of the validation_split option.
raw_validation_dataset = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

log(f"raw test dataset from directory\n")

raw_test_dataset = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size)

# Prepare dataset: standardization, tokenization, vectorization (clean up punct., split up words, convert to numbers)
log(f"CUSTOM STANDARDIZATION\n")


# This custom_standardization() function appears to be a callback which will be applied to each item of input text
# individually.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = raw_training_dataset.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_training_dataset))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_training_dataset.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

print("1787 ---> ", vectorize_layer.get_vocabulary()[1787])
print("2601 ---> ", vectorize_layer.get_vocabulary()[2601])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

training_dataset = raw_training_dataset.map(vectorize_text)
val_ds = raw_validation_dataset.map(vectorize_text)
test_ds = raw_test_dataset.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

training_dataset = training_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.summary()

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

epochs = 10
history = model.fit(
    training_dataset,
    validation_data=val_ds,
    epochs=epochs)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)


log(f"HISTORY PLOTS\n")

history_dict = history.history
history_dict.keys()

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)


log(f"PLOT: Training Loss, Validation Loss\n")

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


log(f"PLOT: Training Accuracy, Validation Accuracy\n")

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()


log(f"TEST PHASE\n")


log(f"ACTION: Sequential groups a linear stack of layers into a tf.keras.Model. Object name: export_model.")
log(f"Sequential: vectorize_layer passed in, model passed in, layers.Activation: sigmoid")
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

log(f"export_model.compile - losses.BinaryCrossentropy-non-logits, optimizer: adam")
export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

log(f"export_model.evaluate raw_test_dataset")
# Test it with `raw_test_dataset`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_dataset)
log(f"Accuracy:\n{accuracy}")

examples = [
    "The movie was great!",
    "The movie was okay.",
    "The movie was terrible..."
]

log(f"predict")
result = export_model.predict(examples)

# In Collab, you would see the .predict() output automatically, but here we have to explicitly print it.
print(result)
print()
log(f"done")

