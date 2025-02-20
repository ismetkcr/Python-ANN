# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 22:00:09 2024

@author: ismt
"""

import json
import math
import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import losses
import numpy as np
from keras import metrics
import matplotlib.pyplot as plt

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    'images',
    image_size=(224, 224),
    validation_split=0.3,
    subset='both',
    seed=123,
)

train_ds
val_ds

#Shuffle the train dataset to increase diversity of batches
train_ds = train_ds.shuffle(
    10 * 32 , reshuffle_each_iteration=True
    )


images = next(iter(train_ds.take(1)))[0]
print(images.shape)
keras_cv.visualization.plot_image_gallery(images, value_range=(0, 255))

model = keras_cv.models.ImageClassifier.from_preset(
    "efficientnetv2_b0_imagenet", num_classes=2
)
model.compile(
    loss="sparse_categorical_crossentropy",
    # optimizer=tf.optimizers.SGD(learning_rate=0.01),
    optimizer=tf.keras.optimizers.AdamW(),
    metrics=["accuracy"],
)


r = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
)


plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend();


plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend();

# Make a prediction on a single image
from PIL import Image

path = 'images/Cheetahs/ed51aa4321f10a21.jpg'
im = Image.open(path)
im

np_im = np.asarray(im)
resizing = keras_cv.layers.Resizing(
    224, 224, crop_to_aspect_ratio=True
)

np_im_rs = resizing(np_im)
np_im_rs
np_im_rs.shape

x = tf.reshape(np_im_rs, (1, 224, 224, 3))
out = model.predict(x)
print(out)
val_ds.class_names














