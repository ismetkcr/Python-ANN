# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 22:15:54 2024

@author: ismt
"""

#become one with data.
#preprocess data
#create model
#fit model
#evaluate model
#adjust different hyperparameters 
#repeat until satisfy

import zipfile

#unzip data

zip_ref = zipfile.ZipFile("10_food_classes_all_data.zip", "r")
zip_ref.extractall()
zip_ref.close()

import os 
#walk through 10 classes of food image data
for dirpath, dirnames, filenames in os.walk("10_food_classes_all_data"):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    
    
#setup train and test directories
train_dir = "10_food_classes_all_data/train/"
test_dir = "10_food_classes_all_data/test/"

#get sub directories (class names)
#visualizeee

import pathlib
import numpy as np
data_dir = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(class_names)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

def view_random_image(target_dir, target_class):
    target_folder = target_dir + target_class
    
    random_image = random.sample(os.listdir(target_folder), 1)
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    print(f"Image shape: {img.shape}")
    return img


img = view_random_image(target_dir=train_dir,
                        target_class=random.choice(class_names))

#preprocess data.. (prepare it for the model)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
#rescale
train_datagen = ImageDataGenerator(rescale=1/255.0)
test_datagen = ImageDataGenerator(rescale=1/255.0)

#load data in from directories and turn batches
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(224,224),
                                               batch_size=32,
                                               class_mode="categorical")

test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(224,224),
                                             batch_size=32,
                                             class_mode="categorical")

##create multiclass cNN model start with the baseline

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation

model_8 = Sequential([
    Conv2D(10, 3, input_shape=(224, 224,3)),
    Activation(activation="relu"),
    Conv2D(10,3, activation="relu"),
    MaxPool2D(),
    Conv2D(10,3, activation="relu"),
    Conv2D(10,3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(10, activation="softmax")
    ])


#compile
model_8.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

#Fit the model

history_8 = model_8.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))



#evaluate model..

model_8.evaluate(test_data)

#check loss curves
#plot validation and training curves seperately
def plot_loss_curves(history):
    """
    returns seperate loss curves for train and valid

    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy=history.history["accuracy"]
    val_accuracy=history.history["val_accuracy"]
    
    epochs=range(len(history.history["loss"])) #how many epochs
    #plot loss
    plt.figure()
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()
    
    #plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label="training_acc")
    plt.plot(epochs, val_accuracy, label="val_acc")
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.legend()
    plt.show()
    
    
plot_loss_curves(history_8)

#adjust model hyperparams to reduce overfiting
#due to its performance on train data, learns something but cant generalize unseen data
#try to fixt overfit 
#get more data, 
#simplify model
#data augmentation

model_9 = Sequential([
    Conv2D(10,3, activation="relu", input_shape=(224,224,3)),
    MaxPool2D(),
    Conv2D(10,3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(10, activation="softmax")
    ])

model_9.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


history_9 = model_9.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))


plot_loss_curves(history_9)
#not much improve..
#try data augmentation.

train_datagen_augmented = ImageDataGenerator(rescale=1/255.0,
                                             rotation_range=0.2,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=True
                                             )

train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode="categorical")

#create another model fit it with augmented train data of 10 classes

model_10 = tf.keras.models.clone_model(model_8)
model_10.compile(loss="categorical_crossentropy",
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=["accuracy"])

history_10 = model_10.fit(train_data_augmented,
                          epochs=5,
                          steps_per_epoch=len(train_data_augmented),
                          validation_data=test_data,
                          validation_steps=len(test_data))

plot_loss_curves(history_10)
class_names

#run lost of experiments to improve results increate layers/hiddenlayers..
#adjust lr 
#different methods of data augmentation
#train for longer maybe 10 epochs
#try transfer learning.

#make prediction with our trained model 10

#create function to import image and resize 224,224
def load_and_prep_image(filename, img_shape=224):
    #read img
    img = tf.io.read_file(filename)
    #decode img
    img = tf.image.decode_image(img)
    #resize img
    img = tf.image.resize(img, size=[img_shape, img_shape])
    #rescale image get all values 0 and one
    img = img/255.
    
    return img


#work for multiclass images
def pred_and_plot(model, filename, class_names=class_names):
    img = load_and_prep_image(filename)
    pred = model.predict(tf.expand_dims(img, axis=0))
    print(len(pred[0]))
    
    if len(pred[0]) > 1:
        pred_class = class_names[tf.argmax(pred[0])]
    else:
        pred_class = class_names[int(tf.round(pred[0]))]
        
    
    
    
    plt.imshow(img)
    plt.title(f"Prediction:{pred_class}")
    plt.axis(False)
    plt.show()
    

pred_and_plot(model=model_10,
              filename="03-pizza-dad.jpeg",
              class_names=class_names)



#save a model
model_10.save("saved_trained_model_10.h5")

loaded_model_10 = tf.keras.models.load_model("saved_trained_model_10.h5")
loaded_model_10.evaluate(test_data)
model_10.evaluate(test_data)







