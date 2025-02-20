# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:08:50 2024

@author: ismt
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import optimizers
import keras_cv
import numpy as np
from keras_cv import bounding_box
import os
from keras_cv import visualization
import tqdm
import json
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


with open('cat_dog/output.coco.json') as f:
    j = json.load(f)

class_mapping = {c['id']: c['name'] for c in j['categories']}

image_paths = [f"cat_dog/{d['file_name']}" for d in j['images']]
#prefill same length as images
classes = [[] for _ in image_paths]
bounding_boxes = [[] for _ in image_paths]

for a in j['annotations']:
    idx = a['id'] - 1
    classes[idx].append(a['category_id'])
    bounding_boxes[idx].append(a['bbox'])

# Verify the entries of classes and bounding_boxes
print(classes)
print(bounding_boxes)

#we assume the images appear in the json in order! Note: you dont have to
#assume this - as an exercise, check that it's OK, and/or write a better version
#(Using the IDs to index the array of images instead of assuming their order)
"""
# Extract the ordered list of file names from the JSON
json_file_names = [img['file_name'] for img in j['images']]

# List all image files in the folder and filter to only include .png files, then sort them
folder_image_files = [f for f in os.listdir('cat_dog') if f.endswith('.png')]
# Check if the order of file names in the JSON matches the order of image files in the folder
if json_file_names == folder_image_files:
    print("The order of file names in JSON is consistent with the order of images in the folder.")
else:
    print("The order of file names in JSON is NOT consistent with the order of images in the folder.")
    print("JSON file names order:", json_file_names)
    print("Folder file names order:", folder_image_files)
"""
no_annotations = [len(x)==0 for x in classes]
np.sum(no_annotations)
len(image_paths)
#image_paths = [x for x, na in zip(image_paths, no_annotations) if not na]
#classes = [x for x, na in zip(classes, no_annotations) if not na]
#bounding_boxes = [x for x, na in zip(bounding_boxes, no_annotations) if not na]

bbox = tf.ragged.constant(bounding_boxes)
classes = tf.ragged.constant(classes)
image_paths = tf.ragged.constant(image_paths)

data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))
# Shuffle the dataset
data = data.shuffle(buffer_size=len(image_paths), seed=42)
len(data)
val_data = data.take(4)
train_data = data.skip(4)

def load_image(image_path):
    image=tf.io.read_file(image_path)
    image=tf.image.decode_jpeg(image, channels=3)
    return image

def load_dataset(image_path, classes, bbox):
    #read image
    image=load_image(image_path)
    bounding_boxes={
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox}
    return {
        "images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}

#hypreparameters
BATCH_SIZE = 4
augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xywh"),
        keras_cv.layers.RandomShear(
            x_factor=0.2, y_factor=0.2, bounding_box_format="xywh"),
        keras_cv.layers.JitteredResize(
            target_size=(640, 640), scale_factor=(0.75, 1.3),
            bounding_box_format="xywh")])

#Create train dataset
train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(BATCH_SIZE*4)
train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

#create validation dataset
resizing = keras_cv.layers.Resizing(
    640, 640, bounding_box_format="xywh", pad_to_aspect_ratio=True)

val_ds = val_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.shuffle(BATCH_SIZE*4)
val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
val_ds = val_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)

def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping)

visualize_dataset(
    train_ds, bounding_box_format="xywh", value_range=(0, 255), rows=1, cols=2)

visualize_dataset(
    val_ds, bounding_box_format="xywh", value_range=(0, 255), rows=1, cols=2)

def dict_to_tuple(inputs):
    return inputs["images"], bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=32)

train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

#global clipnorm helps to reduce exploding gradients
base_lr = 0.005
#including a global_clipnorm is extremely important in object detection tasks
optimizer=tf.keras.optimizers.SGD(
    learning_rate=base_lr, momentum=0.9, global_clipnorm=10.0)
#https://keras.io/api/keras_cv/models/ maybe choose smaller model..
model = keras_cv.models.RetinaNet.from_preset(
    "mobilenet_v3_small_imagenet",
    num_classes=len(class_mapping),
    bounding_box_format="xywh")

model.compile(
    classification_loss="focal",
    box_loss="smoothl1",
    optimizer=optimizer)

#remove take(20) for full training takes very long


model.fit(
    train_ds.take(12),
    validation_data=val_ds.take(4),
    epochs=1)

def visualize_detections(model, dataset, bounding_box_format):
    images, y_true = next(iter(dataset.take(1)))
    y_pred = model.predict(images)
    y_pred_boxes = y_pred["boxes"]
    y_pred_classes = y_pred["classes"]

    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred={"boxes": tf.convert_to_tensor(y_pred_boxes), "classes": tf.convert_to_tensor(y_pred_classes)},
        scale=4,
        rows=2,
        cols=2,
        show=True,
        font_scale=0.7,
        class_mapping=class_mapping,
    )


    

#set ıou and confidence threshold
model.prediction_decoder=keras_cv.layers.MultiClassNonMaxSuppression(
    bounding_box_format="xywh",
    from_logits=True,
    iou_threshold=0.1,
    confidence_threshold=0.1)

#construct dataaset with larger batches
visualization_ds = val_ds.unbatch()
visualization_ds = visualization_ds.ragged_batch(4)
visualization_ds = visualization_ds.shuffle(4)


visualize_detections(model, dataset=visualization_ds,
                     bounding_box_format="xywh")



