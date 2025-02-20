# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:25:46 2024

@author: ismt
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD

vgg = tf.keras.applications.VGG16(input_shape=[100,100,3], include_top=False, weights='imagenet')

x = Flatten()(vgg.output)
x = Dense(4, activation='sigmoid')(x)
model = Model(vgg.input, x)
#generate white boxes on black background color
def image_generator(batch_size=64):
    #generate image and targets
    while True:
        #each epoch will have 50 batches
        for _ in range(50):
            X = np.zeros((batch_size, 100, 100, 3))
            Y = np.zeros((batch_size, 4))
            
            for i in range(batch_size):
                #make the boxes and store their location in target
                row0 = np.random.randint(90)
                col0 = np.random.randint(90)
                row1 = np.random.randint(row0, 100)
                col1 = np.random.randint(col0, 100)
                X[i, row0:row1, col0:col1,:] = 1
                Y[i, 0] = row0/100
                Y[i, 1] = col0/100
                Y[i, 2] = (row1 - row0)/100.
                Y[i, 3] = (col1 - col0)/100.
            yield X, Y

model.compile(loss = 'mse', optimizer=Adam(learning_rate=0.001)) #also we can use here
#binary_cross_entropy loss, we can behave outpts are predictions. it not false.
model.fit(image_generator(),
          steps_per_epoch=20,
          epochs=2)

from matplotlib.patches import Rectangle
#make predictions
def make_prediction():
    #generate random image
    x = np.zeros((100, 100, 3))
    row0 = np.random.randint(90)
    col0 = np.random.randint(90)
    row1 = np.random.randint(row0, 100)
    col1 = np.random.randint(col0, 100)
    x[row0:row1, col0:col1,:] = 1
    print(row0, col0, row1, col1)
    #predict
    X = np.expand_dims(x,0)
    p = model.predict(X)[0]
    #draw the box
    fig, ax = plt.subplots(1)
    ax.imshow(x)
    rect = Rectangle(
        (p[1]*100, p[0]*100),
        p[3]*100, p[2]*100, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()

#PART 2 starts here we try to localize charmander on black background    
from tensorflow.keras.preprocessing import image
ch = image.load_img('charmander-tight.png')
plt.imshow(ch)
plt.show()
print(np.array(ch).shape)

from imageio import imread
ch = imread('charmander-tight.png')
plt.imshow(ch)
print(ch.shape) #four channel is transparancy only for png files
#whats in the 4th channel?
plt.imshow(ch[:, :, 3])
plt.show()
#Data generator
POKE_DIM = 200
ch = np.array(ch)
CH_H, CH_W, _ = ch.shape
def pokemon_generator(batch_size=64):
    #generate image and targets
    while True:
        #each epoch will have 50 batches
        for _ in range(50):
            X = np.zeros((batch_size, POKE_DIM, POKE_DIM, 3))
            Y = np.zeros((batch_size,4))
            for i in range(batch_size):
                #choose location and store in target
                row0 = np.random.randint(POKE_DIM - CH_H)
                col0 = np.random.randint(POKE_DIM - CH_W)
                row1 = row0 + CH_H
                col1 = col0 + CH_W
                X[i, row0:row1, col0:col1,:] = ch[:, :, :3]
                Y[i,0] = row0 / POKE_DIM
                Y[i,1] = col0 / POKE_DIM
                Y[i,2] = (row1-row0)/POKE_DIM
                Y[i,3] = (col1-col0)/POKE_DIM
            yield X / 255., Y
                
def make_model():
    vgg = tf.keras.applications.VGG16(
        input_shape=[POKE_DIM, POKE_DIM, 3],
        include_top = False,
        weights = 'imagenet')
    x = Flatten()(vgg.output)
    x = Dense(4, activation='sigmoid')(x)
    model = Model(vgg.input, x)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001))
    return model

model = make_model()
model.fit(
    
    pokemon_generator(),
    steps_per_epoch=20,
    epochs=2,
)
#Make predictions
def pokemon_prediction():
    #generate random image
    x = np.zeros((POKE_DIM, POKE_DIM, 3))
    row0 = np.random.randint(POKE_DIM - CH_H)
    col0 = np.random.randint(POKE_DIM - CH_W)
    row1 = row0 + CH_H
    col1 = col0 + CH_W
    x[row0:row1, col0:col1, :] = ch[:, :, :3]
    print("true", row0, col0, row1, col1)
    
    #predict
    X = np.expand_dims(x,0) / 255.
    p = model.predict(X)[0]
    #calculate target / loss
    y = np.zeros(4)
    y[0] = row0/POKE_DIM
    y[1] = col0/POKE_DIM
    y[2] = (row1-row0)/POKE_DIM
    y[3] = (col1-col0)/POKE_DIM
    
    #draw the box 
    row0 = int(p[0]*POKE_DIM)
    col0 = int(p[1]*POKE_DIM)
    row1 = int(p[2] * POKE_DIM + row0)
    col1 = int(p[3] * POKE_DIM + col0)
    print("pred:", row0, col0, row1, col1)
    print("loss:", -np.mean(y * np.log(p) + (1-y)*np.log(1-p)))
    
    fig, ax = plt.subplots(1)
    ax.imshow(x.astype(np.uint8))
    rect = Rectangle(
        (p[1]*POKE_DIM, p[0]*POKE_DIM),
        p[3]*POKE_DIM, p[2]*POKE_DIM, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()

#PART 3 make arbitrary size charmander..
from skimage.transform import resize
def pokemon_generator_with_resize(batch_size=64):
  # generate image and targets
  while True:
    # Each epoch will have 50 batches. Why? No reason
    for _ in range(50):
      X = np.zeros((batch_size, POKE_DIM, POKE_DIM, 3))
      Y = np.zeros((batch_size, 4))
      
      for i in range(batch_size):
        # resize charmander - make it bigger or smaller
        scale = 0.5 + np.random.random() # [0.5, 1.5]
        new_height = int(CH_H * scale)
        new_width = int(CH_W * scale)
        obj = resize(
            ch,
            (new_height, new_width),
            preserve_range=True).astype(np.uint8) # keep it from 0..255
        
        # choose location and store in target
        row0 = np.random.randint(POKE_DIM - new_height)
        col0 = np.random.randint(POKE_DIM - new_width)
        row1 = row0 + new_height
        col1 = col0 + new_width
        X[i,row0:row1,col0:col1,:] = obj[:,:,:3]
        Y[i,0] = row0/POKE_DIM
        Y[i,1] = col0/POKE_DIM
        
        # later: make the pokemon different sizes
        Y[i,2] = (row1 - row0)/POKE_DIM
        Y[i,3] = (col1 - col0)/POKE_DIM
      
      yield X/255., Y
def make_model2():
    vgg = tf.keras.applications.VGG16(
        input_shape = [POKE_DIM, POKE_DIM, 3],
        include_top = False,
        weights = 'imagenet')
    x = Flatten()(vgg.output)
    x = Dense(4, activation='sigmoid')(x)
    model = Model(vgg.input, x)
    model.compile(loss='binary_crossentropy',
                 optimizer=Adam(learning_rate=0.001))
    return model

model = make_model2()
model.fit(pokemon_generator_with_resize(),steps_per_epoch=25, epochs=2)
#make predictions with resize
# Make predictions with resize
def pokemon_prediction_with_resize():
  # resize charmander - make it bigger or smaller
  scale = 0.5 + np.random.random()
  new_height = int(CH_H * scale)
  new_width = int(CH_W * scale)
  obj = resize(
      ch,
      (new_height, new_width),
      preserve_range=True).astype(np.uint8) # keep it from 0..255
  
  # Generate a random image
  x = np.zeros((POKE_DIM, POKE_DIM, 3))
  row0 = np.random.randint(POKE_DIM - new_height)
  col0 = np.random.randint(POKE_DIM - new_width)
  row1 = row0 + new_height
  col1 = col0 + new_width
  x[row0:row1,col0:col1,:] = obj[:,:,:3]
  print("true:", row0, col0, row1, col1)
  
  # Predict
  X = np.expand_dims(x, 0) / 255.
  p = model.predict(X)[0]
  
  # Draw the box
  row0 = int(p[0]*POKE_DIM)
  col0 = int(p[1]*POKE_DIM)
  row1 = int(row0 + p[2]*POKE_DIM)
  col1 = int(col0 + p[3]*POKE_DIM)
  print("pred:", row0, col0, row1, col1)
  
  
  fig, ax = plt.subplots(1)
  ax.imshow(x.astype(np.uint8))
  rect = Rectangle(
      (p[1]*POKE_DIM, p[0]*POKE_DIM),
      p[3]*POKE_DIM, p[2]*POKE_DIM,linewidth=1,edgecolor='r',facecolor='none')
  ax.add_patch(rect)
  plt.show()
  
#STEP 4 flip charmander
def pokemon_generator_flip(batch_size=64):
  # generate image and targets
  while True:
    # Each epoch will have 50 batches. Why? No reason
    for _ in range(50):
      X = np.zeros((batch_size, POKE_DIM, POKE_DIM, 3))
      Y = np.zeros((batch_size, 4))
      
      for i in range(batch_size):
        # make the circles and store their location in target
        row0 = np.random.randint(POKE_DIM - CH_H)
        col0 = np.random.randint(POKE_DIM - CH_W)
        row1 = row0 + CH_H
        col1 = col0 + CH_W
        
        # maybe flip
        if np.random.random() < 0.5:
          obj = np.fliplr(ch)
        else:
          obj = ch
        
        X[i,row0:row1,col0:col1,:] = obj[:,:,:3]
        Y[i,0] = row0/POKE_DIM
        Y[i,1] = col0/POKE_DIM
        
        # later: make the pokemon different sizes
        Y[i,2] = (row1 - row0)/POKE_DIM
        Y[i,3] = (col1 - col0)/POKE_DIM
      
      yield X / 255., Y
    
model = make_model()
model.fit(
    pokemon_generator_flip(),
    steps_per_epoch=25,
    epochs=2,
)    
                
# Make predictions
def pokemon_prediction_flip():
  # Generate a random image
  x = np.zeros((POKE_DIM, POKE_DIM, 3))
  row0 = np.random.randint(POKE_DIM - CH_H)
  col0 = np.random.randint(POKE_DIM - CH_W)
  row1 = row0 + CH_H
  col1 = col0 + CH_W
  
  # maybe flip
  if np.random.random() < 0.5:
    obj = np.fliplr(ch)
  else:
    obj = ch

  x[row0:row1,col0:col1,:] = obj[:,:,:3]
  print("true:", row0, col0, row1, col1)
  
  # Predict
  X = np.expand_dims(x, 0) / 255.
  p = model.predict(X)[0]
  
  # Draw the box
  row0 = int(p[0]*POKE_DIM)
  col0 = int(p[1]*POKE_DIM)
  row1 = int(row0 + p[2]*POKE_DIM)
  col1 = int(col0 + p[3]*POKE_DIM)
  print("pred:", row0, col0, row1, col1)
  
  
  fig, ax = plt.subplots(1)
  ax.imshow(x.astype(np.uint8))
  rect = Rectangle(
      (p[1]*POKE_DIM, p[0]*POKE_DIM),
      p[3]*POKE_DIM, p[2]*POKE_DIM,linewidth=1,edgecolor='r',facecolor='none')
  ax.add_patch(rect)
  plt.show()

#STEP 5 localization on backgrounded image..
from glob import glob
backgrounds = []

background_files = glob('backgrounds/*.jpg')
for f in background_files:
  # Note: they may not all be the same size
  bg = np.array(image.load_img(f))
  backgrounds.append(bg)
plt.imshow(backgrounds[3])            

def pokemon_generator_bg(batch_size=64):
  # generate image and targets
  while True:
    # Each epoch will have 50 batches. Why? No reason
    for _ in range(50):
      X = np.zeros((batch_size, POKE_DIM, POKE_DIM, 3))
      Y = np.zeros((batch_size, 4))
      
      for i in range(batch_size):
        # select a random background
        bg_idx = np.random.choice(len(backgrounds))
        bg = backgrounds[bg_idx]
        bg_h, bg_w, _ = bg.shape
        rnd_h = np.random.randint(bg_h - POKE_DIM)
        rnd_w = np.random.randint(bg_w - POKE_DIM)
        X[i] = bg[rnd_h:rnd_h+POKE_DIM,rnd_w:rnd_w+POKE_DIM].copy()
        #X is 200x200 slice from the full background
        #and this will serve as bg for charmander
        # resize charmander - make it bigger or smaller
        scale = 0.5 + np.random.random()
        new_height = int(CH_H * scale)
        new_width = int(CH_W * scale)
        obj = resize(
            ch,
            (new_height, new_width),
            preserve_range=True).astype(np.uint8) # keep it from 0..255
        
        # maybe flip
        if np.random.random() < 0.5:
          obj = np.fliplr(obj)
        
        # choose a random location to store the object
        row0 = np.random.randint(POKE_DIM - new_height)
        col0 = np.random.randint(POKE_DIM - new_width)
        row1 = row0 + new_height
        col1 = col0 + new_width
        
        # can't 'just' assign obj to a slice of X
        # since the transparent parts will be black (0)
        mask = (obj[:,:,3] == 0) # find where the pokemon is 0
        bg_slice = X[i,row0:row1,col0:col1,:] # where we want to place `obj`
        bg_slice = np.expand_dims(mask, -1) * bg_slice # (h,w,1) x (h,w,3)
        bg_slice += obj[:,:,:3] # "add" the pokemon to the slice
        X[i,row0:row1,col0:col1,:] = bg_slice # put the slice back
        
        # make targets
        Y[i,0] = row0/POKE_DIM
        Y[i,1] = col0/POKE_DIM
        
        # later: make the pokemon different sizes
        Y[i,2] = (row1 - row0)/POKE_DIM
        Y[i,3] = (col1 - col0)/POKE_DIM
      
      yield X / 255., Y
                
                
xx = None
yy = None
for x, y in pokemon_generator_bg():
  xx, yy = x, y
  break

plt.imshow(xx[5])

model = make_model2()
model.fit(
    pokemon_generator_bg(),
    steps_per_epoch=20,
    epochs=5,
)

# Make predictions
# Make predictions
def pokemon_prediction_bg():
  # select a random background
  bg_idx = np.random.choice(len(backgrounds))
  bg = backgrounds[bg_idx]
  bg_h, bg_w, _ = bg.shape
  rnd_h = np.random.randint(bg_h - POKE_DIM)
  rnd_w = np.random.randint(bg_w - POKE_DIM)
  x = bg[rnd_h:rnd_h+POKE_DIM,rnd_w:rnd_w+POKE_DIM].copy()
        
  # resize charmander - make it bigger or smaller
  scale = 0.5 + np.random.random()
  new_height = int(CH_H * scale)
  new_width = int(CH_W * scale)
  obj = resize(
      ch,
      (new_height, new_width),
      preserve_range=True).astype(np.uint8) # keep it from 0..255
        
  # maybe flip
  if np.random.random() < 0.5:
    obj = np.fliplr(obj)
        
  # choose a random location to store the object
  row0 = np.random.randint(POKE_DIM - new_height)
  col0 = np.random.randint(POKE_DIM - new_width)
  row1 = row0 + new_height
  col1 = col0 + new_width
        
  # can't 'just' assign obj to a slice of X
  # since the transparent parts will be black (0)
  mask = (obj[:,:,3] == 0) # find where the pokemon is 0
  bg_slice = x[row0:row1,col0:col1,:] # where we want to place `obj`
  bg_slice = np.expand_dims(mask, -1) * bg_slice # (h,w,1) x (h,w,3)
  bg_slice += obj[:,:,:3] # "add" the pokemon to the slice
  x[row0:row1,col0:col1,:] = bg_slice # put the slice back
  print("true:", row0, col0, row1, col1)
  
  # Predict
  X = np.expand_dims(x, 0) / 255.
  p = model.predict(X)[0]
  
  # Draw the box
  row0 = int(p[0]*POKE_DIM)
  col0 = int(p[1]*POKE_DIM)
  row1 = int(row0 + p[2]*POKE_DIM)
  col1 = int(col0 + p[3]*POKE_DIM)
  print("pred:", row0, col0, row1, col1)
  
  
  fig, ax = plt.subplots(1)
  ax.imshow(x.astype(np.uint8))
  rect = Rectangle(
      (p[1]*POKE_DIM, p[0]*POKE_DIM),
      p[3]*POKE_DIM, p[2]*POKE_DIM,linewidth=1,edgecolor='r',facecolor='none')
  ax.add_patch(rect)
  plt.show()







