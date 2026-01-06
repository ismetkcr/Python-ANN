# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 13:28:51 2024

@author: ismt
"""

import win32gui, win32ui, win32con
import numpy as np
import cv2 as cv
from time import time
import cv2
import matplotlib.pyplot as plt
from get_window_2 import WindowCapture


import cv2 as cv



def draw_rectangle_grid(image, top_left, bottom_right, rows, cols, spacing):
    """Generates a grid of rectangles and draws them on the given image."""
    rectangles = []
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]

    for row in range(rows):
        for col in range(cols):
            x1 = top_left[0] + col * (width + spacing)
            y1 = top_left[1] + row * (height + spacing)
            x2 = x1 + width
            y2 = y1 + height

            # Append the rectangle coordinates to the list
            rectangles.append((x1, y1, x2, y2))

            # Draw the rectangle on the image
            cv.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

    return rectangles


# Global variables for mouse coordinates
mouse_x, mouse_y = 0, 0

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y



def calculate_hp_fill_percentage(hp_bar_array):
    # Mavi renk için tolerans
    blue_mask = (hp_bar_array[:, :, 0] < 10) & (hp_bar_array[:, :, 1] < 10) & (hp_bar_array[:, :, 2] > 200)
    # Siyah renk için tolerans
    black_mask = (hp_bar_array[:, :, 0] < 10) & (hp_bar_array[:, :, 1] < 10) & (hp_bar_array[:, :, 2] < 10)
    
    # Mavi (dolu) piksel sayısı
    blue_pixel_count = np.sum(blue_mask)
    # Siyah (boş) piksel sayısı
    black_pixel_count = np.sum(black_mask)
    
    # Toplam doluluk oranı
    total_pixels = blue_pixel_count + black_pixel_count
    fill_percentage = (blue_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0

    return fill_percentage

def calculate_mp_fill_percentage(mp_bar_array):
    """
    MP bar doluluk oranını yüzde olarak hesaplar.
    Kırmızı pikseller dolu, siyah pikseller boş olarak kabul edilir.
    """
    # Kırmızı renk için tolerans
    red_mask = (mp_bar_array[:, :, 0] > 200) & (mp_bar_array[:, :, 1] < 50) & (mp_bar_array[:, :, 2] < 50)
    # Siyah renk için tolerans
    black_mask = (mp_bar_array[:, :, 0] < 10) & (mp_bar_array[:, :, 1] < 10) & (mp_bar_array[:, :, 2] < 10)
    
    # Kırmızı (dolu) piksel sayısı
    red_pixel_count = np.sum(red_mask)
    # Siyah (boş) piksel sayısı
    black_pixel_count = np.sum(black_mask)
    
    # Toplam doluluk oranı
    total_pixels = red_pixel_count + black_pixel_count
    fill_percentage = (red_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0

    return fill_percentage


def normalize_to_0_100(value, min_val=0, max_val=100):
    """
    Verilen bir değeri [0, 100] aralığına normalize eder.
    """
    normalized_value = (value - min_val) / (max_val - min_val) * 100
    return normalized_value


# Initialize the window capture
wincap = WindowCapture('Knight Online Client')
#wincap = WindowCapture('Warfare OnLine Client')

loop_time = time()

# Set up the mouse callback for the Computer Vision window
cv.namedWindow('Computer Vision')
cv.setMouseCallback('Computer Vision', mouse_callback)

while True:
    # Get an updated screenshot
    screenshot = wincap.get_screenshot()

    # Display the mouse coordinates on the screenshot
    mouse_text = f'Mouse Position: ({mouse_x}, {mouse_y})'
    cv.putText(screenshot, mouse_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw Boxes for inventory
    inventory_top_left = (660, 422)
    inventory_bottom_right = (700, 463)
    draw_rectangle_grid(screenshot, inventory_top_left, inventory_bottom_right, rows=4, cols=7, spacing=9)

    # Draw box for durability
    durability_top_left = (850, 356)
    durability_bottom_right = (890, 396)
    #cv.rectangle(screenshot, durability_top_left, durability_bottom_right, color=(0, 0, 255), thickness=2)
    
    # Draw boxes for support actions
    support_top_left = (14, 108)
    support_bottom_right = (46, 138)
    draw_rectangle_grid(screenshot, support_top_left, support_bottom_right, rows=1, cols=7, spacing=0)

    # Coordinates for monster HP bar
    monster_hp_bar_top_left = (402, 38)
    monster_hp_bar_bottom_right = (602, 56)
    cv.rectangle(screenshot, monster_hp_bar_top_left, monster_hp_bar_bottom_right, color=(0, 0, 255), thickness=2)
    
    # Coordinates for character HP bar
    char_hp_bar_top_left = (26, 33)
    char_hp_bar_bottom_right = (220, 47)
    cv.rectangle(screenshot, char_hp_bar_top_left, char_hp_bar_bottom_right, color=(0, 0, 255), thickness=2)
    
    # Coordinates for character MP bar
    char_mp_bar_top_left = (24, 51)
    char_mp_bar_bottom_right = (223, 65)
    cv.rectangle(screenshot, char_mp_bar_top_left, char_mp_bar_bottom_right, color=(0, 0, 255), thickness=2)
    
    #coordinates for monster name 
    monster_name_top_left = (448,14)
    monster_name_bottom_right = (566, 29)
    cv.rectangle(screenshot, monster_name_top_left, monster_name_bottom_right, color=(0, 0, 255), thickness=2)

    # Crop monster_hp bar
    cropped_monster_hp_bar = screenshot[monster_hp_bar_top_left[1]:monster_hp_bar_bottom_right[1], 
                              monster_hp_bar_top_left[0]:monster_hp_bar_bottom_right[0]]
    #gray_cropped_monster_hp_bar = cv2.cvtColor(cropped_monster_hp_bar, cv2.COLOR_BGR2GRAY)
    monster_hp_bar_array = cropped_monster_hp_bar
    monster_hp_fill_percentage = calculate_hp_fill_percentage(monster_hp_bar_array)
    #boşta iken 100 gösteriyor..
    # if monster_hp_fill_percentage == 100:
    #     print("monster tutulmadı..")
    # else:        
    normalized_monster_hp = normalize_to_0_100(monster_hp_fill_percentage, min_val=20.59, max_val=77.62)
    print(f"Monster HP Doluluk Oranı: {normalized_monster_hp:.2f}%")


    
    
    # Crop the HP bar area and convert to grayscale
    cropped_char_hp_bar = screenshot[char_hp_bar_top_left[1]:char_hp_bar_bottom_right[1], 
                              char_hp_bar_top_left[0]:char_hp_bar_bottom_right[0]]
    #gray_cropped_char_hp_bar = cv2.cvtColor(cropped_char_hp_bar, cv2.COLOR_BGR2GRAY)
    char_hp_bar_array = cropped_char_hp_bar
    hp_fill_percentage = calculate_hp_fill_percentage(char_hp_bar_array)
    normalized_hp = normalize_to_0_100(hp_fill_percentage, min_val=35.14, max_val=66.69)
    print(f"HP Doluluk Oranı: {normalized_hp:.2f}%")
    
    
    cropped_char_mp_bar = screenshot[char_mp_bar_top_left[1]:char_mp_bar_bottom_right[1], 
                              char_mp_bar_top_left[0]:char_mp_bar_bottom_right[0]]
    #gray_cropped_char_mp_bar = cv2.cvtColor(cropped_char_mp_bar, cv2.COLOR_BGR2GRAY)
    char_mp_bar_array = cropped_char_mp_bar
    mp_fill_percentage = calculate_mp_fill_percentage(char_mp_bar_array)
    normalized_mp = normalize_to_0_100(mp_fill_percentage, min_val=1, max_val=73.87)
    print(f"MP Doluluk Oranı: {normalized_mp:.2f}%")


    # Display the screenshot with mouse coordinates in a computer vision window
    cv.imshow('Computer Vision', screenshot)

    # Debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()
    
       # Add this inside the loop where you process the bar arrays
    if cv.waitKey(1) == ord('s'):  # Press 's' to display the monster HP bar
        plt.figure(figsize=(6, 3))
        plt.title("Monster HP Bar")
        plt.imshow(monster_hp_bar_array, cmap='gray')
        plt.colorbar(label='Intensity')
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.show()
    
    if cv.waitKey(1) == ord('h'):  # Press 'h' to display the character HP bar
        plt.figure(figsize=(6, 3))
        plt.title("Character HP Bar")
        plt.imshow(char_hp_bar_array, cmap='gray')
        plt.colorbar(label='Intensity')
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.show()
    
    if cv.waitKey(1) == ord('m'):  # Press 'm' to display the character MP bar
        plt.figure(figsize=(6, 3))
        plt.title("Character MP Bar")
        plt.imshow(char_mp_bar_array, cmap='gray')
        plt.colorbar(label='Intensity')
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.show()
    
    # Press 'q' with the output window focused to exit
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break