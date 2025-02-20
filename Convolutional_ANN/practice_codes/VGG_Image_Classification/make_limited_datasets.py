# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:32:05 2024

@author: ismt
"""
import os

def mkdir(p):
    if not os.path.exists(p):  # Hata düzeltildi: exists
        os.mkdir(p)

def link(src, dst):
    os.symlink(src, dst, target_is_directory=True)
    
mkdir('C:/Users/ismt/Desktop/Python-ANN/ConvolutionalNetwork/VGG_exp/fruits-360-small')

# classes = ['Apple Golden 1',
#            'Avocado',
#            'Lemon',
#            'Mango',
#            'Kiwi',
#            'Banana',
#            'Strawberry',
#            'Raspberry']

classes = ['Banana',
           'Strawberry',
           'Raspberry']

train_path_from = os.path.abspath('C:/Users/ismt/Desktop/Python-ANN/ConvolutionalNetwork/VGG_exp/fruits-360/Training')
valid_path_from = os.path.abspath('C:/Users/ismt/Desktop/Python-ANN/ConvolutionalNetwork/VGG_exp/fruits-360/Validation')

train_path_to = os.path.abspath('C:/Users/ismt/Desktop/Python-ANN/ConvolutionalNetwork/VGG_exp/fruits-360-small/Training')
valid_path_to = os.path.abspath('C:/Users/ismt/Desktop/Python-ANN/ConvolutionalNetwork/VGG_exp/fruits-360-small/Validation')

mkdir(train_path_to)
mkdir(valid_path_to)


for c in classes:
  link(train_path_from + '/' + c, train_path_to + '/' + c)
  link(valid_path_from + '/' + c, valid_path_to + '/' + c)
