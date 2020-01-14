"""
Created on Mon Jan  6 02:43:09 2020

@author: nick
"""

import tensorflow as tf
import os
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 160
BATCH_SIZE = 40

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

INPUT_TENSOR_NAME = "inception_v3_input"

def keras_model_fn(hyperparameters):
    base_model = InceptionV3(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
    base_model.trainable = False
    model = tf.keras.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(units=256, activation='relu'))
    model.add(layers.Dense(units=5, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model


def train_input_fn(training_dir, hyperparameters):
    """Returns input function that would feed the model during training"""
    return _input(tf.estimator.ModeKeys.TRAIN,
                  batch_size=BATCH_SIZE,
                  data_dir=training_dir)


def eval_input_fn(training_dir, hyperparameters):
    return _input(tf.estimator.ModeKeys.EVAL,
                  batch_size=BATCH_SIZE,
                  data_dir=training_dir)


def serving_input_fn(hyperparameters):
    tensor = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, 3])
    inputs = {INPUT_TENSOR_NAME: tensor}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def _input(mode, batch_size, data_dir):
    assert os.path.exists(data_dir), ("Unable to find images resources for input, are you sure you downloaded them ?")

    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(data_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=batch_size)
    images, labels = generator.next()
    return {INPUT_TENSOR_NAME: images}, labels