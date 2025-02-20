# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:25:36 2024

@author: ismt
"""
#https://poloclub.github.io/cnn-explainer/ #kaynak
#get the data.

import zipfile

#Unzip file
zip_ref = zipfile.ZipFile("pizza_steak.zip")
zip_ref.extractall()
zip_ref.close()

#now we have data as file, its splitted already as test and train data.. 
#data --> subset of food_101 only contains 2 class: pizza, steak

#inspect the data .. become one with data.

#A very crucial step at the begin.. become one with data..
#visualize visualize visualize !!!!

import os

#walk pizza steak directory

for dirpath, dirnames, filenames in os.walk("pizza_steak"):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

#250 images in test, 750 images in train data..


#another way to find how many images
num_steak_images_train = len(os.listdir("pizza_steak/train/steak"))
print(num_steak_images_train)

#visualize images 
import pathlib
import numpy as np

data_dir = pathlib.Path("pizza_steak/train")
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
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

img = view_random_image(target_dir="pizza_steak/train/",
                        target_class="pizza")

import tensorflow as tf
tf.constant(img)

img.shape #width, height, color channels

#get all the pixel values between 0 & 1
img/255 #normalization actually (img-min)/(max-min) min= 0, max= 255


#### LOAD IMAGES -- PREPROCESS IMAGES - BUILD CNN - COMPİLE AND FIT CNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#setseed
tf.random.set_seed(42)

#preprocess normalize
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
#set up paths
train_dir = "C:/Users/ismt/Desktop/python-ann/tensorflow_2/tf2_convolutionalnetworks/pizza_steak/train"
test_dir = "C:/Users/ismt/Desktop/python-ann/tensorflow_2/tf2_convolutionalnetworks/pizza_steak/test"
#import data form dicts and turn it to batches

train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode='binary',
                                               seed=42)
valid_data = valid_datagen.flow_from_directory(directory=test_dir,
                                               batch_size=32,
                                               target_size=(224,224),
                                               class_mode="binary",
                                               seed=42)

#build CNN - tiny VGG
model_1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=10,
                           kernel_size=3,
                           activation="relu",
                           input_shape=(224,224,3)),
    tf.keras.layers.Conv2D(10, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2,
                              padding="valid"),
    tf.keras.layers.Conv2D(10,3, activation="relu"),
    tf.keras.layers.Conv2D(10,3, activation="relu"),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation="sigmoid")
        
    ])

#compile CNN
model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_1 = model_1.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))


##using the same model as before regression model..

tf.random.set_seed(42)

model_2 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
    
    ])

model_2.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_2 = model_2.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))

#improve regression model for classification_image

tf.random.set_seed(42)

model_3 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(224,224,3)),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
    ])

model_3.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_3 = model_3.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))

"""
1 Become one with data visualize
2 preprocess data normalize
3 create model
4 fit the model
5 evaluate model
6 adjust parameters improve model
7 evaluate model
"""
#1 - visualize data..

plt.figure()
plt.subplot(1, 2, 1)
steak_img = view_random_image("pizza_steak/train/", "steak")
pizza_img = view_random_image("pizza_steak/train/", "pizza")

#2 - preprocess data

train_dir = "pizza_steak/train/"
test_dir = "pizza_steak/test/"

#turn data to batches
#create train and test data generators and rescale data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

#load in our image data from directories and turn batches

train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               batch_size=32)

test_data = test_datagen.flow_from_directory(directory=test_dir,
                                             target_size=(224, 224),
                                             class_mode="binary",
                                             batch_size=32)

images, labels = train_data.next() #get the next batch of images
len(images)
len(labels)

#get first 2 images
images[:2], images[0].shape

#3 create model

#create cnn model (start baseline: relatively simple model)


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential

#create baseline model 
model_4 = Sequential([
    Conv2D(filters=10, kernel_size=3, strides=1, padding="valid",
           activation="relu", input_shape=(224, 224, 3)),
    Conv2D(10, 3, activation="relu"),
    Conv2D(10, 3, activation="relu"),
    Flatten(),
    Dense(1, activation="sigmoid")
    ])


#compile the model..
model_4.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_4.summary()


#check data first
len(train_data), len(test_data)

history_4 = model_4.fit(train_data, epochs=5, steps_per_epoch=len(train_data),
                        validation_data=test_data, validation_steps=len(test_data))

#evaluate model
import pandas as pd
import matplotlib.pyplot as plt
#plot training curves
pd.DataFrame(history_4.history).plot()

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
    

plot_loss_curves(history_4)

#reduce overfit with max pool..
model_5 = Sequential([
    Conv2D(10, 3, activation="relu", input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=2),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(pool_size=2),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation="sigmoid")
    ])

model_5.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

history_5 = model_5.fit(train_data, epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))


#reduce overfit with data augmentation..
#ımage generator with data augmentation

train_datagen_augmented = ImageDataGenerator(rescale=1/255,
                                             rotation_range=0.2,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             horizontal_flip=True)

#image generator without augmentation
train_datagen = ImageDataGenerator(rescale=1/255.)

test_datagen = ImageDataGenerator(rescale=1/255.)

#visualize augmented data..
print("Augmented Training data")
train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                   target_size=(224,224),
                                                                   batch_size=32,
                                                                   class_mode="binary",
                                                                   shuffle=False)
print("non augmented training data")
#create non augmented train data batches
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(224, 244),
                                               batch_size=32,
                                               class_mode="binary",
                                               shuffle=False)

IMG_SIZE = (224, 224)
#create non augmented test data_batches
print("non augmented test data")
test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size= IMG_SIZE,
                                             batch_size=32,
                                             class_mode="binary",
                                             )


#note: data augmentation is usually only performed on training data..

#visualize some augmented data..

images, labels = train_data.next()
augmented_images, augmented_labels = train_data_augmented.next()

import random
random_number = random.randint(0, 32) #batch size iz 32
plt.imshow(images[random_number])
plt.imshow(augmented_images[random_number])


#train network wiht augmented data..
#create model  same as model 5
model_6 = Sequential([
    Conv2D(10, 3, activation='relu'),
    MaxPool2D(pool_size=2),
    Conv2D(10, 3, activation='relu'),
    MaxPool2D(),
    Conv2D(10, 3, activation='relu'),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation="sigmoid")
    ])


model_6.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=['accuracy'])

history_6 = model_6.fit(train_data_augmented,
                        epochs=5,
                        steps_per_epoch=len(train_data_augmented),
                        validation_data=test_data,
                        validation_steps=len(test_data))



#check models loss curves
plot_loss_curves(history_6)

#discover powwer of shuffle..

#shuflee train data and augmented training data and train same model..

train_data_augmented_shuffled = train_datagen_augmented.flow_from_directory(train_dir,
                                                                            target_size=(224,224,),
                                                                            class_mode='binary',
                                                                            batch_size=32,
                                                                            shuffle=True
                                                                            )


#create same model as 5 and 6 
model_7 = Sequential([
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Conv2D(10,3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation="sigmoid")
    ])

model_7.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

history_7 = model_7.fit(train_data_augmented_shuffled,
            epochs=5,
            steps_per_epoch=len(train_data_augmented_shuffled),
            validation_data=test_data,
            validation_steps=len(test_data))

plot_loss_curves(history_7)

#improving results. still ..? increase layers?, increase number of filters en each conv layer
#train for longer
#Find ideal lr
#use transfer learning.. ??? we will see this.

#classes we are working with
import matplotlib.image as mpimg
steak = mpimg.imread("03-steak.jpeg")
plt.imshow(steak)
plt.axis(False)

#check shape of image
steak.shape
tf.expand_dims(steak, axis=0).shape

pred = model_7.predict(tf.expand_dims(steak, axis=0))
#when we train nn we want to make predictiom with own custom data..
#its important we need to preprocess custom data same format as trained data..

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

steak = load_and_prep_image("03-steak.jpeg")
steak.shape

pred = model_7.predict(tf.expand_dims(steak, axis=0)) #returns prediction probabilty.

#looks like our custom image is being put through model..
class_names

pred_class = class_names[int(tf.round(pred))]
pred_class

def pred_and_plot(model, filename, class_names=class_names):
    img = load_and_prep_image(filename)
    pred = model.predict(tf.expand_dims(img, axis=0))
    pred_class = class_names[int(tf.round(pred))]
    
    plt.imshow(img)
    plt.title(f"Prediction:{pred_class}")
    plt.axis(False)
    
pred_and_plot(model_7, "03-steak.jpeg")
#model works, try with another image..
pred_and_plot(model_7, "03-pizza-dad.jpeg")

   
