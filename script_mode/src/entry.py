#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 02:43:09 2020

@author: nick
"""

import argparse
import os
import logging
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

IMG_SIZE = 160
BATCH_SIZE = 40
CLASSES = 14

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

INPUT_TENSOR_NAME = "inception_v3_input"


def keras_model_fn():
    base_model = MobileNetV2(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
    base_model.trainable = False
    model = Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
#     model.add(layers.Flatten())
#     model.add(layers.Dense(units=1024, activation='relu',        kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dense(units=512, activation='relu',        kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dense(units=CLASSES, kernel_initializer='glorot_normal', activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model


def train_input_fn():
    return _input(args.train, args.batch_size, 'train')


def validation_input_fn():
    return _input(args.validation, args.batch_size, 'validation')


def _input(data_dir, batch_size, mode):
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(data_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=batch_size)
    return generator


def save_model(model, output):
#     signature = tf.saved_model.signature_def_utils.predict_signature_def(
#         inputs={'image': model.input}, outputs={'scores': model.output})
    
#     builder = tf.saved_model.builder.SavedModelBuilder(output + '/1/')
#     builder.add_meta_graph_and_variables(
#         sess=K.get_session(),
#         tags=[tf.saved_model.tag_constants.SERVING],
#         signature_def_map={
#             tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
#                 signature
#         })
#     builder.save()
    
#     model.save(output+'/1', save_format='tf')
    
#     tf.keras.models.save_model(
#     model,
#     output+'/1',
#     save_format='tf'
#
    
    
#     tf.saved_model.save(model, output+'/1/')
    tf.saved_model.save(model, output + '/1/')
    #model.save(output + '/1/')
    print("Model successfully saved at: {}".format(output))
    return


def main(args):
    print("getting data")
    train_generator = train_input_fn()
    validation_generator = validation_input_fn()
    model = keras_model_fn()
    print(model.summary())

    history = model.fit_generator(
        train_generator,
        steps_per_epoch = math.ceil(train_generator.n / train_generator.batch_size),
        epochs=args.epochs,
        validation_data=validation_generator,
        validation_steps= math.ceil(validation_generator.n / validation_generator.batch_size)
    )
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,max(plt.ylim())])
    plt.title('Training and Validation Loss')
    
    if not os.path.exists(args.model_output_dir + '/1'):
        os.makedirs(args.model_output_dir + '/1')
    
    plt.savefig(args.model_output_dir + '/1/plot.jpg', dpi=300)

    save_model(model, args.model_output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_TRAIN'),
        help='The S3 directory where the training images are stored.')
    parser.add_argument(
        '--validation',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_VALIDATION'),
        help='The S3 directory where the validation images are stores')
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='The directory where the model will be stored.')
    parser.add_argument(
        '--model_output_dir',
        type=str,
        default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=2e-4,
        help='Weight decay for convolutions.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001)
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='The number of steps to use for training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for training.')

    args = parser.parse_args()
    main(args)

    # hyperparameters sent by the client are passed as command-line arguments to the script.
