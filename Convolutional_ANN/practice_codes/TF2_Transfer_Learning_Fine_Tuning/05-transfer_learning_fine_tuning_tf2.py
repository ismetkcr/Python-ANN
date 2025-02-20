# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 22:31:54 2025

@author: ismt
"""

#previous we covered featue extraction now fine tuning. (train more layer)

from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, walk_through_dir

unzip_data("10_food_classes_10_percent.zip")
walk_through_dir("10_food_classes_10_percent")

#create train and test dir paths

train_dir = "10_food_classes_10_percent/train"
test_dir = "10_food_classes_10_percent/test"

import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE=32
train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                            image_size=IMG_SIZE,
                                                                            label_mode="categorical",
                                                                            batch_size=BATCH_SIZE)

test_data = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                image_size=IMG_SIZE,
                                                                label_mode="categorical",
                                                                batch_size=BATCH_SIZE)

#check class names

train_data_10_percent.class_names
#see example of batch data

for images, labels in train_data_10_percent.take(1):
    print(images, labels)


#base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
#1create model with functional api
base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)

#2 freeze the base model
base_model.trainable=False

#3 create inputs to our model
inputs = tf.keras.layers.Input(shape=(224,224,3), name="input_layer")

#4 If using Resnet we need to normalize input
#x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)

#5 Pass the inputs to base model
x = base_model(inputs)
print(f"Shape after passing inputs through base model: {x.shape}")

#6. Average pool the outputs of the base model(aggregate all the most important information with less computations)
x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
print(f"Shape after passig GlobalPool2D: {x.shape}")

#7 Create output activation layer
outputs = tf.keras.layers.Dense(10, activation='softmax', name="output_layer")(x)

#8 Combine inputs and outputs
model_0 = tf.keras.Model(inputs, outputs)

#9 Compile Model

model_0.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


history_0 = model_0.fit(train_data_10_percent,
                        epochs=5,
                        steps_per_epoch=len(train_data_10_percent),
                        validation_data=test_data,
                        validation_steps=int(0.25 * len(test_data)),
                        )

model_0.evaluate(test_data)

#check layers on our base model
for layer_number, layer in enumerate(base_model.layers):
    print(layer_number, layer.name)

#summary of base model
base_model.summary()

model_0.summary()

#check loss curves
plot_loss_curves(history_0)


#getting feature vector from trained model
#we have tensor after goes through base model of shape 7, 7, 1280
#then when it passes to globalaverage pooling None, 1280

#let use similar shaped tensor 1, 4, 4, 3 and pass globalavaragepool2D
input_shape = (1, 4, 4, 3)
tf.random.set_seed(42)
input_tensor = tf.random.normal(input_shape)
print(f"random input tensor: \n {input_tensor}\n")
#pass random tensor globalavaragepool

global_average_pooled_tensor = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
print(f"2D global average pooled random tensor:\n {global_average_pooled_tensor}\n")

#check shaepe
print(f"Shape of input tensor: {input_tensor.shape}\n")
print(f"Shape of pooled tensor: {global_average_pooled_tensor.shape}\n")

#replicate globalaveragepool2d layer
tf.reduce_mean(input_tensor, axis=[1, 2])

#model_1 - use feature extraction with 1 percent of the training data and use data augmentation.
#model_2 - use feature_extraction with 10 percent of the trainind data and use data augöentation
#use same test dataset when evaluation step
#model_3 - use fine tuning on 10 percent of training data with data augmentation
#model_4 - use fine tuning on 100 percent of the training data with data augmentation..

#unzip data..
#unzip_data("10_food_classes_1_percent.zip")
walk_through_dir("10_food_classes_1_percent")

#create training and test dir
train_dir_1_percent = "10_food_classes_1_percent/train"
test_dir = "10_food_classes_1_percent/test"

walk_through_dir(train_dir_1_percent)

#setup data loaders
#getting and preprocess data for model_1
IMG_SIZE = (224, 224)
train_data_1_percent = tf.keras.preprocessing.image_dataset_from_directory(train_dir_1_percent,
                                                                           label_mode="categorical",
                                                                           image_size=IMG_SIZE,
                                                                           batch_size=32)

test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode="categorical",
                                                                image_size=IMG_SIZE,
                                                                batch_size=32)

#adding data augmentation right into model
#tf.keras.layers.experimental.preprocessing()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing

data_augmentation = keras.Sequential([
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomRotation(0.2),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomWidth(0.2),
    #preprocessing.Rescale(1./255), #efficient rescale built in
    
    ], name="data_augmentation")


#augmented_img = data_augmentation(img, training=True)
#visualize augmentation layer
#view random image and compare it augmented version
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random

target_class = random.choice(train_data_1_percent.class_names)
target_dir = "10_food_classes_1_percent/train/" + target_class
random_image = random.choice(os.listdir(target_dir))
random_image_path = target_dir + "/" + random_image

img = mpimg.imread(random_image_path)
print(img)
plt.imshow(img)
plt.title(f"original_random image from class {target_class}")
plt.axis(False)

#now lets plot augmented random image
augmented_img = data_augmentation(img)
plt.figure()
plt.imshow(augmented_img/255.0)
plt.title(f"Augmented random image from class {target_class}")
plt.axis(False)

#feature extraction transfer learning with 1 percent of data with data augmentation

#setup input shape and base model
input_shape = (224, 224, 3)
base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
base_model.trainable = False

inputs = tf.keras.layers.Input(shape=input_shape,
                               name="input_layer")

#add sequential model as layer
x = data_augmentation(inputs)

#give base model the inputs
x = base_model(x, training=False)

#pool the output features
x = tf.keras.layers.GlobalAveragePooling2D(name="average_pooling_layer")(x)

#put dense layer on as the output
outputs = tf.keras.layers.Dense(10, activation='softmax', name="output_layer")(x)

model_1 = tf.keras.Model(inputs, outputs)


#compile model
model_1.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

#fit the model
history_1_percent_of_model_1 = model_1.fit(train_data_1_percent,
                                           epochs=5,
                                           steps_per_epoch=len(train_data_1_percent),
                                           validation_data=test_data,
                                           validation_steps=int(0.25 * len(test_data)))

results_1_percent_data_aug = model_1.evaluate(test_data)
results_1_percent_data_aug

#how loss curves with 1 percent of data with data augmentation
plot_loss_curves(history_1_percent_of_model_1)

#model 2.. #feature extraction transfer learning 10 percent data with data augmentation..

import tensorflow as tf

train_dir_10_percent = "10_food_classes_10_percent/train"
test_dir = "10_food_classes_10_percent/test"
train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(train_dir_10_percent,
                                                                            label_mode="categorical",
                                                                            image_size=IMG_SIZE,
                                                                            batch_size=32)

test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode="categorical",
                                                                image_size=IMG_SIZE,
                                                                batch_size=32)


#create model 2 with data augmentation

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential

data_augmentation = Sequential([
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomWidth(0.2),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomRotation(0.2)
    ], name="data_augmentation")


input_shape = (224, 224, 3)

#create frozen base model also called backbone
base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
base_model.trainable=False

#create inputs and outputs 
inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="global_pool_layer")(x)
outputs = layers.Dense(10, activation="softmax", name="output_layer")(x)
model_2 = tf.keras.Model(inputs, outputs)

model_2.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

#Create model check point ant fit
#set check point path
checkpoint_path = "ten_percent_model_checkpoint/checkpoint.ckpt"

#create callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         save_best_only=False,
                                                         save_freq="epoch",
                                                         verbose=1)

#fit model 2 pass with checkpoint callback
initial_epochs = 5
history_10_percent_data_aug = model_2.fit(train_data_10_percent,
                                          epochs=initial_epochs,
                                          validation_data=test_data,
                                          validation_steps=int(0.25 * len(test_data)),
                                          callbacks=checkpoint_callback)

#how to use saved weightts
#Load saved model weights and evaluate
results_10_percent_data_aug = model_2.evaluate(test_data)
plot_loss_curves(history_10_percent_data_aug)


model_2.load_weights(checkpoint_path)

#evaluate with loaded weights
loaded_weights_model_results = model_2.evaluate(test_data)

import numpy as np
np.isclose(np.array(results_10_percent_data_aug), np.array(loaded_weights_model_results))

#model - 3 Fine-tuning 10 percent data
#Fine tuning usually works best after training feature extraction model for a few epochs
#with large amount of custom data

#layers in our loaded model
model_2.layers

for layer in model_2.layers:
    print(layer, layer.trainable)

for i, layer in enumerate(model_2.layers[2].layers):
    print(layer, layer.name, layer.trainable)

#how many trainable variables
print(len(model_2.layers[2].trainable_variables))

#to begin fine tunnig lets start by setting last 10 layers of our base model
base_model.trainable=True
for layer in base_model.layers[:-10]:
    layer.trainable=False

#we have to recompile our models 
model_2.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                metrics=["accuracy"])

#check traiable layerrs
for layer_number, layer in enumerate(model_2.layers[2].layers):
    print(layer_number, layer.name, layer.trainable)
    
print(len(model_2.trainable_variables))

#5 epochs for fine tune
fine_tune_epochs = initial_epochs + 5

#refit the model same as model 2 (with more trainable)
history_fine_10_percent_data_aug = model_2.fit(train_data_10_percent,
                                               epochs=fine_tune_epochs,
                                               validation_data=test_data,
                                               validation_steps=int(0.25 * len(test_data)),
                                               initial_epoch=history_10_percent_data_aug.epoch[-1],
                                               callbacks=checkpoint_callback) #start training from previous last epoch


#evaluate fine tuned model 
results_fine_tune_10_percent = model_2.evaluate(test_data)

#comparing model results..
plot_loss_curves(history_fine_10_percent_data_aug)

def compare_histories(original_history, new_history, initial_epochs=5):
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]
    
    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]
    
    #combine original history 
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]
    
    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]
    
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label="Training accuracy")
    plt.plot(total_val_acc, label="Validation accuracy")
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label="Start Fine Tuning")
    plt.legend(loc="lower right")
    plt.title("Training and Val Accuracy")
    
    #plt.figure()
    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label="Training loss")
    plt.plot(total_val_loss, label="Validation loss")
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label="Start Fine Tuning")
    plt.legend(loc="upper right")
    plt.title("Training and Val Loss")
    
compare_histories(history_10_percent_data_aug,
                  history_fine_10_percent_data_aug,
                  initial_epochs=5)
    
    
#Model 4 more data same model 3
#unzip all data
#unzip_data("10_food_classes_all_data.zip")

train_dir_all_data = "10_food_classes_all_data/train"
test_dir = "10_food_classes_all_data/test"

walk_through_dir("10_food_classes_all_data")

#setup data 
import tensorflow as tf
IMG_SIZE = (224, 224)
train_data_10_classes_all = tf.keras.preprocessing.image_dataset_from_directory(train_dir_all_data,
                                                                                image_size=IMG_SIZE,
                                                                                label_mode="categorical",
                                                                                batch_size=32)

test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                image_size=IMG_SIZE,
                                                                label_mode="categorical",
                                                                batch_size=32)




#to train fine tune model_4 we need to revert model 2 back to its feature extraction weights use checkpoint
model_2.evaluate(test_data)
model_2.load_weights(checkpoint_path)
model_2.evaluate(test_data) #evaluate after checkpoint

for layer_number, layer in enumerate(model_2.layers):
    print(layer_number, layer.name, layer.trainable)
    
#drill base model

for layer_number, layer in enumerate(model_2.layers[2].layers):
    print(layer_number, layer.name, layer.trainable)
    
    

model_2.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                metrics=["accuracy"])

fine_tune_epochs = initial_epochs + 5
history_fine_10_classes_full = model_2.fit(train_data_10_classes_all,
                                           epochs=fine_tune_epochs,
                                           validation_data = test_data,
                                           validation_steps=int(0.25*len(test_data)),
                                           initial_epoch=history_10_percent_data_aug.epoch[-1],
                                           callbacks=checkpoint_callback)

