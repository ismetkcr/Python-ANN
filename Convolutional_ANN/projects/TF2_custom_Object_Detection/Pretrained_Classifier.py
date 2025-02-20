# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 21:16:30 2024

@author: ismt
"""
import json
import keras_cv
# import keras
import keras_core as keras
import numpy as np



model = keras_cv.models.ImageClassifier.from_preset(
  "efficientnetv2_b0_imagenet_classifier"
)
#free to chose which model we need from here
#https://keras.io/api/keras_cv/models/

#url = 'https://i.redd.it/twpfhcw58xgb1.jpg'
#url = 'https://i.redd.it/aq53vqeaz5nb1.jpg'
url = 'https://wallup.net/wp-content/uploads/2014/10/cars/Ferrari_Enzo_Sport_Car_HD.jpg'
filepath = keras.utils.get_file(origin=url)
image = keras.utils.load_img(filepath)
image = np.array(image)
h, w, c = image.shape
image_batch = image.reshape((1, h, w, c))
keras_cv.visualization.plot_image_gallery(
  image_batch, rows=1, cols=1, value_range=(0, 255), show=True, scale=4
)

probs = model.predict(image_batch)
probs.shape

top_classes = (-probs[0]).argsort()
top_classes[:5]

url = 'https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/kerascv/imagenet_label_names.json'
label_names_filepath = keras.utils.get_file(origin=url)

with open(label_names_filepath) as f:
  label_names = json.load(f)
  
label_names

for c in top_classes[:5]:
    print(label_names[c])
    