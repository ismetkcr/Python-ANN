# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:31:48 2025

@author: ismt
"""
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
        self.inventory_model = load_model('inventory_model.h5')
        self.support_model = load_model('support_model.h5')
        self.durability_model = load_model('durability_model.h5')
        self.actions = 8
        self.action_space = [i for i in range(self.actions)]
        self.consecutive_false_count = 0
        
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
        
        #draw box for member 2
        self.member_2_hp_top_left = (895,224)
        self.member_2_hp_bottom_right = (1008, 233)
        
        #draw box for member 3
        self.member_3_hp_top_left = (895,276)
        self.member_3_hp_bottom_right = (1008, 285)
        
        self.ironclaw_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_pvp/pngs/ironclaw.png"
        # Paths and settings
        
        self.inventory_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/clean_inventory/pngs/inventory.png"
        self.dead_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/clean_inventory/pngs/dead.png"
        self.enchanter_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_main/pngs/enchanter.png"
        
        self.ironclaw_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/ironclaw.png"
        self.troll_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/troll.png"
        self.troll_warrior_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/troll_warrior.png"
        self.troll_warrior_2_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/troll_warrior_2.png"

        self.eslant_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/eslant.png"
        self.guard_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/guard.png"
        self.guard_2_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/guard_2.png"
        self.women_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/women.png"
        self.haunga_warrior_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/haunga_warrior.png"
        self.haunga_warrior_2_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/haunga_warrior_2.png"
        self.haunga_warrior_3_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/haunga_warrior_3.png"
        self.haunga_warrior_4_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/haunga_warrior_4.png"

        self.dtk_1_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/dtk_1.png"
        self.dtk_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/dtk.png"
        self.ancient_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/ancient.png"
        self.troll_2_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/troll_2.png"
        
        self.max_steps = 20 # max step for inventory search
        self.has_predicted = False
        self.locked_slots = set([0, 1, 2, 3, 4])
        self.center_x, self.center_y = self.wincap.w // 2, self.wincap.h // 2
        self.target_size = (21,21)
        # Grid coordinates for bank inventory
        self.bank_inventory_top_left = (645, 398)
        self.bank_inventory_bottom_right = (683, 437)
        self.bank_inventory_rects = self.draw_rectangle_grid(
            self.bank_inventory_top_left, 
            self.bank_inventory_bottom_right, 
            rows=4, 
            cols=7, 
            spacing=12
        )
        # Grid coordinates for inventory
        self.inventory_top_left = (660, 422)
        self.inventory_bottom_right = (700, 463)
        self.inventory_rects = self.draw_rectangle_grid(
            self.inventory_top_left, 
            self.inventory_bottom_right, 
            rows=4, 
            cols=7, 
            spacing=10
        )
        
        #grid coordinates for durability ((848, 356), (892, 396))
        self.durability_top_left = (848, 356)
        self.durability_bottom_right = (892, 396)
        self.durability_rects = self.draw_rectangle_grid(
            self.durability_top_left, 
            self.durability_bottom_right, 
            rows=1, 
            cols=1, 
            spacing=0)
        
        self.episode_step_counter = 0
        
        self.durab_step_counter = 0
        self.bank_step_counter = 0
        
        self.durability_check_step = 250
        self.bank_check_step = 500

        
        # Initialize member support timers and statuses
        self.member_2_last_support_time = 0
        self.member_3_last_support_time = 0
        self.member_2_status = False
        self.member_3_status = False
        
        #Inıtialize wolf timer and wolf status
        self.wolf_last_time = 0
        self.wolf_status = False
        
        #!!111! #self....!!! !!!
        
       
 
        # Set up the mouse callback
        cv2.namedWindow('Computer Vision')
        cv2.setMouseCallback('Computer Vision', self.mouse_callback)
    
    
    
    
    def draw_rectangle_grid(self, top_left, bottom_right, rows, cols, spacing):
        """Generates a grid of rectangles based on top-left and bottom-right coordinates."""
        rectangles = []
        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]
        for row in range(rows):
            for col in range(cols):
                x1 = top_left[0] + col * (width + spacing)
                y1 = top_left[1] + row * (height + spacing)
                x2 = x1 + width
                y2 = y1 + height
                rectangles.append((x1, y1, x2, y2))
                
        return rectangles   
    
    def preprocess_rectangles(self, screenshot, rectangles, target_size=(21, 21)):
        """Batch preprocess rectangle regions from the image."""
        """this returns specified rectangles with gray scale.."""
        preprocessed_images = []
    
        for rect_coords in rectangles:
            x1, y1, x2, y2 = rect_coords  # Unpack all four coordinates correctly
    
            roi = screenshot[y1:y2, x1:x2]
            roi_resized = cv2.resize(roi, target_size)
            
            # if len(roi_resized.shape) == 3:
            #     roi_resized = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    
            roi_normalized = roi_resized / 255.0
            roi_normalized = np.expand_dims(roi_normalized, axis=-1)
            preprocessed_images.append(roi_normalized)
    
        return np.array(preprocessed_images)
        
    
    
    
        
        
    
    def check_durability(self):
        time.sleep(1)
        
        # Ensure inventory is open
        found = False
        while not found:
            self.controller.click_(800, 748)
            location = self.controller.find_template(self.inventory_path, confidence=0.7)
            if location:
                found = True
                print("Inventory opened")
                time.sleep(1)
                break

        screenshot_durab = self.wincap.get_screenshot()
        screenshot_durab = cv2.cvtColor(screenshot_durab, cv2.COLOR_BGR2RGB)
        
        processed_images = self.preprocess_rectangles(screenshot_durab, self.durability_rects)
        predictions = self.durability_model.predict((processed_images), verbose=0)
        prediction = np.argmax(predictions, axis=1)
        
        # Check prediction
        if prediction == 0:
            print("Durability issue detected!")
            self.controller.press_("f2")
            self.controller.press_('1')
            self.controller.press_("f1")
        
        self.controller.click_(800, 748)
        # Update the last known prediction
        self.last_durability_prediction = prediction
        return prediction

    def step_check_durability(self):
        # Check every 20 steps
        if self.durab_step_counter % self.durability_check_step == 0:
            prediction = self.check_durability()
            print("DURABILITY CHECKED!!!!!!!!!!!!!!!!")
        else:
            prediction = self.last_durability_prediction  # Use last known prediction

        # Increment the counter
        return prediction
    
    
    
    
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
    
    
                       
    
      
    
      
    
    def execute_action_warrior(self, action):
        if action == 0:
            self.controller.press_("1")  # hp pot
            
          
        elif action == 1: # attack
            self.controller.press_("r")
            time.sleep(0.1)
            self.controller.press_("2")
            self.controller.press_("r")
            time.sleep(0.1)
            self.controller.press_("2")
            
        elif action == 2:# monster find
            self.controller.press_("z")
            time.sleep(0.15)
            print("monster_finding")
           
        
        
        elif action == 3:
            pass
        
        
                    
            

            
        
        
            



       
    
    def step(self, action):
        if action not in self.action_space:
            raise ValueError("Invalid Action")
        
        self.step_check_durability()
        self.execute_action_warrior(action)
        
        screenshot = self.get_screenshot()
        states = self.get_states(screenshot)  # Initial state check
        
        # Update consecutive_false_count
        if states["monster_flag"]:
            self.consecutive_false_count = 0
        else:
            self.consecutive_false_count += 1
    
        # Trigger tuning if needed
        if self.consecutive_false_count >= 3:
            self.tune_screen()
            self.consecutive_false_count = 0  # Reset counter
    
            # Post-tuning check
            post_tune_screenshot = self.get_screenshot()
            post_tune_states = self.get_states(post_tune_screenshot)
            if not post_tune_states["monster_flag"]:
                self.controller.press_("s", hold_duration=3)
    
        # Rest of your step logic
        cv2.rectangle(screenshot, self.char_hp_bar_top_left, self.char_hp_bar_bottom_right, (0, 0, 255), 2)
        cv2.rectangle(screenshot, self.monster_hp_bar_top_left, self.monster_hp_bar_bottom_right, (0, 0, 255), 2)
        cv2.rectangle(screenshot, self.monster_name_top_left, self.monster_name_bottom_right, (0, 0, 255), 2)
        
        self.episode_step_counter += 1
        self.durab_step_counter += 1
        
        print(f"Episode Step Counter: {self.episode_step_counter}, "
              f"Durab Step Counter: {self.durab_step_counter}")
        
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
        normalized_hp = self.normalize_to_0_100(hp_percentage)
        
        # Monster HP işlemleri
        monster_hp_bar = screenshot[
            self.monster_hp_bar_top_left[1]:self.monster_hp_bar_bottom_right[1],
            self.monster_hp_bar_top_left[0]:self.monster_hp_bar_bottom_right[0]
        ]
        monster_hp_percentage = self.calculate_hp_fill_percentage(monster_hp_bar)
        normalized_monster_hp = self.normalize_to_0_100(monster_hp_percentage, min_val=20.59, max_val=77.62)
        
        #Monster name işlemleri
        monster_flag_bar = screenshot[
            self.monster_name_top_left[1]:self.monster_name_bottom_right[1],
            self.monster_name_top_left[0]:self.monster_name_bottom_right[0]
            ]
        
        loc = self.controller.find_template_non_gray(self.troll_path, confidence=0.65)
        loc_2 = self.controller.find_template_non_gray(self.troll_2_path, confidence=0.65)
        loc_3 = self.controller.find_template_non_gray(self.troll_warrior_2_path, confidence=0.65)
        loc_4 = self.controller.find_template_non_gray(self.troll_warrior_path, confidence=0.65)
        monster_flag=False
        if loc is not None or loc_2 is not None:
            monster_flag=True
        if loc_3 is not None or loc_4 is not None:
            #self.controller.press_("s", hold_duration=3)
            monster_flag = False
        current_time = time.time()
        # Update member 2 support status based on timer
        
        
        # Tüm değerleri bir sözlük içinde döndür
        return {
            "character_hp": normalized_hp,
            "monster_hp": normalized_monster_hp,
            "monster_flag": monster_flag
            }
    
    
    def tune_screen(self):
        for _ in range(10):
            game.controller.press_("z")
            loc = game.controller.find_template_non_gray(game.troll_path, confidence=0.65)
            if loc is None:
                game.controller.drag_(500, 400, direction='right', move_distance=50, num_moves=2, button='right', delay=0.05)
                time.sleep(0.5)
            else:
                break
    
    
    
    def action_decider_warrior(self, game_status):
      
       
        
         
        # Emergency HP pot if HP is critically low (below 20%)
        if game_status['character_hp'] < 20:
            return 0  # hp_pot
        
        # Monster engagement logic
        if game_status['monster_flag']:  # If monster is present
            if game_status['monster_hp'] == 0:
                # Monster is dead, find new 
                time.sleep(1)
                self.monster_dead_flag = True
                return 2  # monster finding
            else:
                # Monster is alive, attack
                self.monster_dead_flag = False
                return 1  # attack
        else:
            # No monster present, need to find one
            return 2  # monster finding
        
        
        
        
        
        
        
           
           

time.sleep(3)            
        

# Main game instance
game = KnightOnline()
def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"
if __name__ == '__main__':
    action = 6
    #tune screen for troll ,
    for _ in range(10):
        game.controller.press_("z")
        loc = game.controller.find_template_non_gray(game.troll_path, confidence=0.65)
        if loc is None:
            game.controller.drag_(500, 400, direction='right', move_distance=50, num_moves=2, button='right', delay=0.05)
            time.sleep(0.5)
        else:
            break
    
    while True:
        print("currentactionis", action)
        screenshot = game.step(action)
        
        # Tüm durum monitörleme işlemlerini tek bir fonksiyon çağrısıyla yap
        game_status = game.get_states(screenshot)
        action = game.action_decider_warrior(game_status)
        
       
        
        
        
       
        
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
            f"monster_flag: {game_status['monster_flag']}",
            f"monster_hp: {game_status['monster_hp']}"
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