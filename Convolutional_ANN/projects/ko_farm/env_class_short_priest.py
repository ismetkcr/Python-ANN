# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:40:22 2024

@author: ismt
"""

import time
import cv2
import numpy as np
from game_controller import GameController
from get_window_2 import WindowCapture
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from collections import deque

class KnightOnline:
    def __init__(self):
        # Initialize components
        self.controller = GameController('Knight OnLine Client')
        self.wincap = WindowCapture('Knight OnLine Client')
        #self.inventory_model = load_model('inventory_model.h5')
        self.support_model = load_model('support_model.h5')
        #self.durability_model = load_model('durability_model.h5')
        self.actions = 8
        self.action_space = [i for i in range(self.actions)]
        
        # Initialize mouse coordinates
        self.mouse_x, self.mouse_y = 0, 0
        
        # FPS calculation
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.fps = 0
        self.fps_update_interval = 1.0
        self.fps_history = deque(maxlen=10)
        self.loop_time = time.time()
        
        # Screenshot caching
        self.screenshot_cache = None
        self.screenshot_cache_time = 0
        self.screenshot_cache_duration = 0.03
        
        # Coordinates for monster HP bar
        self.monster_hp_bar_top_left = (391, 33)
        self.monster_hp_bar_bottom_right = (584, 50)
        
        # Coordinates for character HP bar
        self.char_hp_bar_top_left = (20, 33)
        self.char_hp_bar_bottom_right = (215, 44)
        
        # Coordinates for character MP bar
        self.char_mp_bar_top_left = (21, 50)
        self.char_mp_bar_bottom_right = (215, 63)
        
        
        self.monster_name_top_left = (455,6)
        self.monster_name_bottom_right = (521, 26)
        
        self.member_1_hp_top_left = (784, 114)
        self.member_1_hp_bottom_right=(895, 122)
        
        #draw box for member 2
        self.member_2_hp_top_left = (784,164)
        self.member_2_hp_bottom_right = (895, 174)
        
        #draw box for member 3
        self.member_3_hp_top_left = (784,216)
        self.member_3_hp_bottom_right = (895, 226)
        
        
        
        
        # Initialize member support timers and statuses
        self.member_1_last_support_time = 0
        self.member_2_last_support_time = 0
        self.member_3_last_support_time = 0
        self.member_1_status = False
        self.member_2_status = False
        self.member_3_status = False
        
        # Execute support actions 4 and 5 on startup
        self.execute_action_priest(6)
        self.execute_action_priest(4)
        self.execute_action_priest(5)
        
 
        # Set up the mouse callback
        cv2.namedWindow('Computer Vision')
        cv2.setMouseCallback('Computer Vision', self.mouse_callback)
    
   
        
   

    
    

    def calculate_hp_fill_percentage(self, hp_bar_array):
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

    def calculate_mp_fill_percentage(self, mp_bar_array):
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

    def normalize_to_0_100(self, value, min_val=36.49, max_val=78.68):
        normalized_value = (value - min_val) / (max_val - min_val) * 100
        return max(0, min(100, normalized_value))

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y

    def update_fps(self):
        self.fps_counter += 1
        current_time = time.time()
        elapsed_time = current_time - self.fps_start_time
    
        if elapsed_time >= self.fps_update_interval:
            self.fps = self.fps_counter / elapsed_time
            self.fps_history.append(self.fps)
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            print(f"Current FPS: {self.fps:.2f} | Average FPS: {avg_fps:.2f}")
            self.fps_start_time = current_time
            self.fps_counter = 0
        return self.fps

    def get_screenshot(self):
        current_time = time.time()
        if (self.screenshot_cache is None or 
            current_time - self.screenshot_cache_time > self.screenshot_cache_duration):
            self.screenshot_cache = self.wincap.get_screenshot()
            self.screenshot_cache_time = current_time
        return self.screenshot_cache.copy()
    
    
                       
    def heal_member(self, member_number):
       coordinates = (840,118), (839, 168), (839, 222)
       
       crd = coordinates[member_number-1]
       self.controller.click_(crd[0], crd[1], clicks=3)
       self.controller.press_("6")
       time.sleep(1.25)
      
    def support_member(self, member_number):
       coordinates = (840,118), (839, 168), (839, 222)
       crd = coordinates[member_number-1]
       self.controller.click_(crd[0], crd[1], clicks=3)
       self.controller.press_("7")
       time.sleep(1.25)
       self.controller.click_(crd[0], crd[1], clicks=3)
       self.controller.press_("8")
       time.sleep(1.25)
    
    def execute_action_priest(self, action):
        if action == 0:
            self.controller.press_("1")  # hp pot
            
        elif action == 1:
            self.controller.press_("2")  # mp pot
            print("MP POT")
        
        elif action == 2:
            self.heal_member(2)
            print("heal 2.")
        
        elif action == 3:
            self.heal_member(3)
            print("heal 3")
          
        elif action == 4:
            self.support_member(2)
            print("sp m2")
            # Update support time and status for member 2
            self.member_2_last_support_time = time.time()
            self.member_2_status = True
        
        elif action == 5:
            self.support_member(3)
            print("sp m3")
            # Update support time and status for member 3
            self.member_3_last_support_time = time.time()
            self.member_3_status = True
        
        elif action==6:
            self.support_member(1)
            print("sp m1")
            self.member_1_last_support_time = time.time()
            self.member_1_status = True
        elif action==7:
            
            pass



       
    
    def step(self, action):
        if action not in self.action_space:
            raise ValueError("Invalid Action")
        
        self.execute_action_priest(action)
    
        # Pass the screenshot to get_states
        screenshot = self.get_screenshot()


    
        
        cv2.rectangle(screenshot, self.char_hp_bar_top_left, self.char_hp_bar_bottom_right, color=(0, 0, 255), thickness=2)
        cv2.rectangle(screenshot, self.char_mp_bar_top_left, self.char_mp_bar_bottom_right, color=(0, 0, 255), thickness=2)
        cv2.rectangle(screenshot, self.member_1_hp_top_left, self.member_1_hp_bottom_right, color=(0, 0, 255), thickness=2)
        cv2.rectangle(screenshot, self.member_2_hp_top_left, self.member_2_hp_bottom_right, color=(0, 0, 255), thickness=2)
        cv2.rectangle(screenshot, self.member_3_hp_top_left, self.member_3_hp_bottom_right, color=(0, 0, 255), thickness=2)

        

        return screenshot
    
    def get_states(self, screenshot):
        """
        Oyun durumunu monitör eden fonksiyon.
        Monster HP, Character HP ve MP değerlerini hesaplar.
        
        Args:
            screenshot: Oyun ekranının anlık görüntüsü
            
        Returns:
            dict: Monster, HP ve MP durumlarını içeren sözlük
        """
        
        
      
        
        
        
        
        
        char_hp_bar = screenshot[
            self.char_hp_bar_top_left[1]:self.char_hp_bar_bottom_right[1],
            self.char_hp_bar_top_left[0]:self.char_hp_bar_bottom_right[0]
        ]
        hp_percentage = self.calculate_hp_fill_percentage(char_hp_bar)
        normalized_hp = self.normalize_to_0_100(hp_percentage, min_val=50, max_val=69.4)
        
        # Character MP işlemleri
        char_mp_bar = screenshot[
            self.char_mp_bar_top_left[1]:self.char_mp_bar_bottom_right[1],
            self.char_mp_bar_top_left[0]:self.char_mp_bar_bottom_right[0]
        ]
        mp_percentage = self.calculate_mp_fill_percentage(char_mp_bar)
        normalized_mp = self.normalize_to_0_100(mp_percentage, min_val=0.6, max_val=57.4)
        
        
        
        
        
        
        #Member 2 hp reading
        member_2_hp_bar = screenshot[
            self.member_2_hp_top_left[1]:self.member_2_hp_bottom_right[1],
            self.member_2_hp_top_left[0]:self.member_2_hp_bottom_right[0]
            ]
        member_2_hp_percentage = self.calculate_hp_fill_percentage(member_2_hp_bar)
        print(member_2_hp_percentage)
        member_2_hp_norm = self.normalize_to_0_100(member_2_hp_percentage, min_val=39.6, max_val=75.8)
        
        #Member 3 hp reading
        member_3_hp_bar = screenshot[
            self.member_3_hp_top_left[1]:self.member_3_hp_bottom_right[1],
            self.member_3_hp_top_left[0]:self.member_3_hp_bottom_right[0]
            ]
        member_3_hp_percentage = self.calculate_hp_fill_percentage(member_3_hp_bar)
        print(member_3_hp_percentage)
        member_3_hp_norm = self.normalize_to_0_100(member_3_hp_percentage, min_val=39.6, max_val=75.8)
        
        #burada member 2 ve member 3 için sup skills zamanı hesaplat ve status olarak dondur
        
        current_time = time.time()
        self.member_1_status = (current_time - self.member_1_last_support_time) < 601
        # Update member 2 support status based on timer
        self.member_2_status = (current_time - self.member_2_last_support_time) < 601  # 600 seconds = 10 minutes
        # Update member 3 support status based on timer
        self.member_3_status = (current_time - self.member_3_last_support_time) < 601
        
        # Tüm değerleri bir sözlük içinde döndür
        return {
            "character_hp": hp_percentage,
            "character_mp": mp_percentage,
            "member_2_hp": member_2_hp_norm,
            "member_3_hp": member_3_hp_norm,
            "member_support_status": self.member_1_status,
            "member_2_support_status": self.member_2_status,
            "member_3_support_status": self.member_3_status
            
        }
    
    
    
    def action_decider(self, game_status):
      
        #check character hp first
        #check member_2 hp
        #check member_3hp
        #check member status and return member 2 for 2 member 3 for 3 and
        
       
        """
        Determines the action to take based on the current game status.
        
        Args:
            game_status (dict): A dictionary containing game state information such as
                                monster_flag, monster_hp, character_hp, and character_mp.
        
        Returns:
            int: The action to be executed.
        """
        
         
        # Emergency HP pot if HP is critically low (below 20%)
        if game_status['character_hp'] < 20:
            return 0  # hp_pot
        
        if game_status['character_mp'] < 50:
            return 1 # mp pot
        # Critical HP check - Heal if HP is below 40%
        if game_status['member_2_hp'] < 70:
            return 2  #(heal member 2)
       
        
        # MP management - Use pot if MP is below 40%
        if game_status['member_3_hp'] < 70:
            return 3  #(heal number 3)
        
        if not game_status['member_2_support_status']:
            #self.member_2_last_support_time = time.time()  # Reset timer
            return 4  # Action code for member 2 support
        
        # Check if member 3 needs support due to timer
        if not game_status['member_3_support_status']:
            #self.member_3_last_support_time = time.time()  # Reset timer
            return 5 # Action code for member 3 support
        
        # Check if member 3 needs support due to timer
        if not game_status['member_support_status']:
            #self.member_3_last_support_time = time.time()  # Reset timer
            return 6 # Action code for member 3 support
        
        return 7
        
        
        
           
           

time.sleep(3)            
        

# Main game instance
game = KnightOnline()
def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"
if __name__ == '__main__':
    action = 0
    while True:
        print("currentactionis", action)
        screenshot = game.step(action)
        
        # Tüm durum monitörleme işlemlerini tek bir fonksiyon çağrısıyla yap
        game_status = game.get_states(screenshot)
        action = game.action_decider(game_status)
        
        # Calculate remaining support times
        current_time = time.time()
        m1_remaining = max(0, 600 - (current_time - game.member_1_last_support_time))
        m2_remaining = max(0, 600 - (current_time - game.member_2_last_support_time))
        m3_remaining = max(0, 600 - (current_time - game.member_3_last_support_time))
        # Convert remaining times to min:sec format
        
        m1_remain_formatted = format_time(m1_remaining)
        m2_remain_formatted = format_time(m2_remaining)
        m3_remain_formatted = format_time(m3_remaining)
       
        
        # Basit text gösterimi (sağ alt köşe)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8  # Punto büyütüldü
        
        # FPS güncellemesi
        game.update_fps()
        fps_text = f"FPS: {game.fps:.1f}"
        
        # FPS'yi sağ üst köşeye ekleme
        cv2.putText(
            screenshot,
            fps_text,
            (screenshot.shape[1] - 150, 75),  # Sağ üst köşe koordinatları
            font,
            font_scale,
            (0, 255, 0),  # Yeşil renk
            2
        )
        
        # Diğer metinleri sağ alta ekleme
        texts = [
            
            f"Char HP: {game_status['character_hp']:.1f}%",
            f"Char MP: {game_status['character_mp']:.1f}%",
            f"member_2_hp: {game_status['member_2_hp']:.1f}",
            f"member_3_hp: {game_status['member_3_hp']:.1f}",
            f"member_1_status : {game_status['member_support_status']}",
            f"member_2_status : {game_status['member_2_support_status']}",
            f"member_3_status : {game_status['member_3_support_status']}",
            f"member_1_remain : {m1_remain_formatted}",
            f"member_2_remain : {m2_remain_formatted}",
            f"member_3_remain: {m3_remain_formatted}",
        ]
        
        # Sağ alt metinler
        y_offset = screenshot.shape[0] - 300  # Alt kenardan başlama noktası
        for text in texts:
            text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
            x_pos = screenshot.shape[1] - text_size[0] - 20  # Sağ kenardan 20 pixel içeride
            cv2.putText(screenshot, text, (x_pos, y_offset), font, font_scale, (0, 255, 0), 2)  # Yeşil renk (0,255,0)
            y_offset += 30  # Satır aralığı artırıldı
        
        # Display the screenshot
        cv2.imshow('Computer Vision', screenshot)
        
        # Handle key events
        key = cv2.waitKey(1)
        if key == ord('q'):  # Quit
            cv2.destroyAllWindows()
            break