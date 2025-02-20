# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:38:29 2025

@author: ismt
"""

#mixed precision training we need to access GPU
from helper_functions import plot_loss_curves, compare_historys
##use tensorflow datasets to download data.

#get tensorflow datasets
import tensorflow_datasets as tfds
#list all availble datasets
datasets_list = tfds.list_builders() # get all avaible datasets
print("food101" in datasets_list)

#load in data 
(train_data,test_data), ds_info = tfds.load(name="food101",
                                            split=["train", "validation"],
                                            shuffle_files=True,
                                            as_supervised=True,
                                            with_info=True)


#explore and become one with data
#features of food101 from tfds
ds_info.features

#get class names
class_names = ds_info.features["label"].names
class_names[:10]

#to become one with data we want to find 
#class names, shape of input_data, data_dtype, how labels build?
#take one sample of train_data
train_one_sample = train_data.take(1)
train_one_sample

#output info about our training set
for image, label in train_one_sample:
    print(f"Image Shape: {image.shape} \n Image datatype: {image.dtype} \n Target_class tensor form: {label}, \n Classname : {class_names[label.numpy()]} ")

import tensorflow as tf
tf.reduce_min(image), tf.reduce_max(image)

##plot image from tfds
import matplotlib.pyplot as plt

plt.imshow(image)
plt.title(class_names[label.numpy()])
plt.axis(False)

#create preprocessing function to prepare our data.
#previously we used tf.keras.preprocessing.image_dataset_from_directory
#what we know about our data ?? is it normalized? batched?
#so in order to get ready data for neural network, often need preprocess functions
# uint8, comprised all different sizes images.. ant not scaled..
#what we know models like ? data float32, 

#make a function preprocess
def preprocess_img(image, label, img_shape=224):
    image = tf.image.resize(image, [img_shape, img_shape])
    #image = image/255.0 #not required for efficient model its had built-in
    return tf.cast(image, tf.float32), label

#preprocess single sample
preprocessed_img = preprocess_img(image, label)[0]

#batch and prepare datasets model..

#map preprocesing function to training data
train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
#shuffle train data and turn batches
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE) #we can use also len(train_data)

#test data
test_data = test_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)
train_data, test_data

#Hey, map this preprocessing function accross our training dataset, shuffle, batch and finaly prefetch
#create modelling callbacks
#modelcheckpoint to save models progress during trainin
checkpoint_path = "model_checkpoints/cp.ckpt"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      monitor="val_acc",
                                                      save_best_only=False,
                                                      save_weights_only=True,
                                                      verbose=0)



"""
TypeError: Input 'y' of 'Sub' Op has type float16 that does not match type float32 of argument 'x'
"""

#setup mixed precision trainin
from tensorflow.keras import mixed_precision
#mixed precision uses float32 and floa16
mixed_precision.set_global_policy("mixed_float16")

#create model
input_shape=(224, 224, 3)
base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
base_model.trainable = False

inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(len(class_names))(x)
#for mixed precision we need to sure outputs float32
outputs = tf.keras.layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x)
model=tf.keras.Model(inputs, outputs)


#labels not one encoded.. sparse
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

model.summary()

#check layer dtype policies are we using mixed precision??
for layer in model.layers:
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

for layer in model.layers[1].layers:
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)
    
#fit feature extraction model..
history_101_food_class_feature_extract = model.fit(train_data,
                                                   epochs=1,
                                                   steps_per_epoch=int(0.1*len(train_data)),
                                                   validation_data=test_data,
                                                   validation_steps=int(0.1 * len(test_data)),
                                                   callbacks=[model_checkpoint])

# #fine tuning
initial_epochs=1
fine_tune_epochs=1
for layer_number, layer in enumerate(model.layers):
    print(layer_number, layer.name, layer.trainable)
    
model_base_model = model.layers[1]
for layer_number, layer in enumerate(model_base_model.layers):
    print(layer_number, layer.name, layer.trainable)
    
for layer in model_base_model.layers[-5:]:
    layer.trainable=True
    
for layer_number, layer in enumerate(model_base_model.layers):
    print(layer_number, layer.name, layer.trainable)

#labels not one encoded.. sparse
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

new_history = model.fit(train_data,
                        epochs=fine_tune_epochs,
                        initial_epoch=0, #history_101_food_class_feature_extract.epoch[-1]
                        steps_per_epoch=int(0.1*len(train_data)),
                        validation_data=test_data,
                        validation_steps=int(0.1*len(test_data)))
