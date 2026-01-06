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
        self.path_flag = False
        self.slot_flag = False
        self.inn_letter_2_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/clean_inventory/pngs/inn_letter_2.png"
        self.inn_letter_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/clean_inventory/pngs/inn_letter.png"
        self.inn_use_storage_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/clean_inventory/pngs/inn_use_storage.png"
        self.inn_confirm_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/clean_inventory/pngs/inn_confirm.png"
        self.dtc_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/clean_inventory/pngs/dtc.png"
        self.dtc_2_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/clean_inventory/pngs/dtc_2.png"
        self.town_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/clean_inventory/pngs/town.png"
        self.inventory_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/clean_inventory/pngs/inventory.png"
        self.dead_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/clean_inventory/pngs/dead.png"
        self.enchanter_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_main/pngs/enchanter.png"
        self.dtc_3_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_main/pngs/dtc_3.png"
        self.dtc_4_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_main/pngs/dtc_4.png"
        self.emc_gate_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/emc_gate.png"
        self.kalluga_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/kalluga.png"
        self.slave_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/slave.png"
        self.lich_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/lich.png"
        self.warp_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/warp_emcgate.png"
        self.press_ok_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/press_ok.png"
        self.osi_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/osi.png"
        self.inn_emc_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/inn_emc.png"
        self.appraiser_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/appraiser.png"
        self.ironclaw_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/ironclaw.png"
        self.troll_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/troll.png"
        self.troll_warrior_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/troll_warrior.png"
        self.eslant_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/eslant.png"
        self.guard_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/guard.png"
        self.guard_2_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/guard_2.png"
        self.women_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/women.png"
        self.haunga_warrior_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/haunga_warrior.png"
        self.haunga_warrior_2_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/haunga_warrior_2.png"
        self.haunga_warrior_3_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/haunga_warrior_3.png"
        self.haunga_warrior_4_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/haunga_warrior_4.png"
        self.troll_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/troll.png"
        self.troll_warrior_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/troll_warrior.png"
        self.troll_warrior_2_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_last/pngs/troll_warrior_2.png"
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
        
        self.execute_action_asas(1)
        self.execute_action_asas(2)
        self.execute_action_asas(3)
        
 
        # Set up the mouse callback
        cv2.namedWindow('Computer Vision')
        cv2.setMouseCallback('Computer Vision', self.mouse_callback)
    
    
    def town_(self):
        time.sleep(1)
        #buraya emin olmak için bir template eklenebilir...
        location = self.controller.find_template(self.dead_path, confidence=0.7)
        if location:
            time.sleep(0.5)
            self.controller.click_(location[0], location[1])
        found = False        
        while not found:
            self.controller.press_('h', hold_duration=0.1)
            time.sleep(1)
            location = self.controller.find_template(self.town_path, confidence=0.7)
            if location:
                found = True                
                self.controller.click_(903, 334, clicks=2)
                self.controller.press_('h', hold_duration=0.1)
                break
        time.sleep(1)
        
    
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
        
    
    
    def find_bank_location_emc(self):
        """Attempts to locate the bank position and clicks a set number of times."""
        
        updated_location = self.controller.find_template(self.osi_path, confidence=0.70)
        if updated_location is None:
            print("osi location not found. Dragging screen to continue searching...")
            self.controller.drag_(500, 400, direction='left', move_distance=125, num_moves=4, button='right', delay=0.25)

        osi_location = updated_location
        print(f"osi location found at: {osi_location}. Performing initial clicks.")

        
        
        for step in range(26):
            updated_location = self.controller.find_template(self.osi_path, confidence=0.7)
            if updated_location is None:
                print("osi not found, drag screen and continue searching..")
                self.controller.drag_(500, 400, direction='left', move_distance=100, num_moves=3, button='right', delay=0.25)
            elif updated_location != osi_location:
                self.controller.click_(updated_location[0], updated_location[1])
            else:
                print("osi loc unchanged.. press a and w")
                self.controller.press_('a', hold_duration=1.5)
                self.controller.press_('w', hold_duration=0.5)
                
            osi_location=updated_location
            print(f"Step: {step} completed.")

        
        time.sleep(6)
        self.controller.press_("w", hold_duration=1.5)
        
        bank_location = self.controller.find_template(self.inn_emc_path, confidence=0.7)
        for step in range(10):
            updated_location = self.controller.find_template(self.inn_emc_path, confidence=0.7)
            
            if updated_location is None:
                print("Bank location not found. Dragging screen to continue searching...")
                self.controller.drag_(500, 400, direction='right', move_distance=100, num_moves=1, button='right', delay=0.25)
            elif updated_location != bank_location:
                self.controller.click_(updated_location[0], updated_location[1])
            else:
                print("Bank location unchanged. Pressing 'd' and 'w'.")
                self.controller.press_('d', hold_duration=3)
                self.controller.press_('w', hold_duration=1)
            
            bank_location = updated_location
            print(f"Step: {step} completed.")
        time.sleep(7)
        self.controller.press_("d", hold_duration=0.5)
        self.controller.press_("w", hold_duration=0.5)
        self.controller.press_("b", hold_duration=0.25)
        self.controller.press_("w", hold_duration=1)
        self.controller.press_("b", hold_duration=0.1)
        return bank_location

    def search_near_center(self):
       """Attempts to find the bank location from screen center."""
       found = False
       # ilk 10 adım sol-aşaüı yönde deneme
       for k in range(10):
           offset_x = self.center_x - k * 20
           offset_y = self.center_y + 100
           print(f"Clicking at ({offset_x}, {offset_y}) in left-down direction")
           self.controller.click_(offset_x, offset_y, button='right')
           #check use_storage loc
           inn_use_storage_loc = self.controller.find_template(self.inn_use_storage_path, confidence=0.7)
           if inn_use_storage_loc:
               self.controller.click_(inn_use_storage_loc[0] + 10, inn_use_storage_loc[1])
               print("Inn hostess clicked successfully in left-down direction.")
               found = True
               break
   
       if not found:
           print("Returning to center.")
           self.controller.click_(self.center_x, self.center_y, button='right')
           time.sleep(0.1)
   
           for k in range(10):
               offset_x = self.center_x + k * 25
               offset_y = self.center_y + 100
               print(f"Clicking at ({offset_x}, {offset_y}) in right-down direction")
               self.controller.click_(offset_x, offset_y, button='right')
   
               inn_use_storage_loc = self.controller.find_template(self.inn_use_storage_path, confidence=0.7)
               if inn_use_storage_loc:
                   self.controller.click_(inn_use_storage_loc[0] + 10, inn_use_storage_loc[1])
                   print("Inn hostess clicked successfully in right-down direction.")
                   found = True
                   break
               
       return found
       
    def check_visiting_bank(self):
        time.sleep(0.5)
        
        # Ensure inventory is open
        found = False
        while not found:
            self.controller.click_(800, 748)
            location = self.controller.find_template(self.inventory_path, confidence=0.7)
            if location:
                found = True
                #print("Inventory opened")
                time.sleep(1)
                break

        screenshot_visit = self.wincap.get_screenshot()
        screenshot_visit = cv2.cvtColor(screenshot_visit, cv2.COLOR_BGR2RGB)
        
        processed_images = self.preprocess_rectangles(screenshot_visit, self.inventory_rects)
        print("shape is ", processed_images.shape)
        predictions = self.inventory_model.predict(np.squeeze(processed_images), verbose=0)
        predictions = np.argmax(predictions, axis=1)
        print(predictions)
        
        # Check the number of empty slots
        number_of_zeros = sum(1 for i in predictions if i == 0)
        print("Number of empty slots:", number_of_zeros)
        
        if number_of_zeros <= 5:
            bank_flag = None
            print("Empty slot value is too low, go to bank and clear inventory")
        else:
            bank_flag = True
            print("Empty slot value is fine, no need to go to bank")
        
        self.controller.click_(800, 748)  # Close inventory
        print("Inventory closed")
        self.controller.move_(512, 374)
        
        # Update last bank check results
        self.last_bank_check = (bank_flag, predictions)
        return bank_flag, predictions

    def step_check_bank(self):
        # Perform check every 50 steps
        if self.bank_step_counter % self.bank_check_step == 0:
            bank_flag, predictions = self.check_visiting_bank()
            print("BANK VISIT CHECKED!!!!!!!!!!!!!!!")
        else:
            bank_flag, predictions = self.last_bank_check  # Use last known results

         
        return bank_flag, predictions
    
    def process_inventory_selective(self):
        time.sleep(1)
        """Process inventory with selective slot processing."""
        screenshot = self.wincap.get_screenshot()
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        
        processed_images = self.preprocess_rectangles(screenshot, self.bank_inventory_rects)
        predictions = self.inventory_model.predict(np.squeeze(processed_images), verbose=0)
        predictions = np.argmax(predictions, axis=1)
        print(predictions)
        
        # Process eligible slots
        for idx, (rect_coords, pred) in enumerate(zip(self.bank_inventory_rects, predictions)):
            # Skip locked slots, slots with prediction 1 in first 4 slots, and slots with prediction 2
            if (idx in self.locked_slots) or (pred == 0):
                continue
                
            if pred != 0:  # Process all non-0 predictions except for locked slots
                x1, y1, x2, y2 = rect_coords  # Unpack all four coordinates correctly
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                self.controller.click_(center_x, center_y, button='right')
                confirm_loc = self.controller.find_template(self.inn_confirm_path, confidence=0.7)
                
                if confirm_loc is not None:
                    time.sleep(0.25)
                    self.controller.press_('enter')
                time.sleep(0.25)
        
        self.town_()
            
        return predictions
    
    
    
     
    def go_slot_ironclaw(self):
        found = False
        # Attempts to locate the bank position and clicks a set number of times.
        time.sleep(0.1)
        updated_location = self.controller.find_template(self.dtc_path, confidence=0.75)
        
        if updated_location is None:
            print("DTC location not found. Dragging screen to continue searching...")
            self.controller.drag_(500, 400, direction='right', move_distance=150, num_moves=4, button='right', delay=0.25)
    
        dtc_location = updated_location
        print(f"DTC location found at: {dtc_location}. Performing initial clicks.")
    
        for step in range(12):
            updated_location = self.controller.find_template(self.dtc_path, confidence=0.75)
            
            if updated_location is None:
                print("DTC location not found. Dragging screen to continue searching...")
                self.controller.drag_(500, 400, direction='left', move_distance=100, num_moves=2, button='right', delay=0.5)
            elif updated_location != dtc_location:
                self.controller.click_(updated_location[0] , updated_location[1])
            else:
                print("DTC location unchanged. Pressing 'd' and 'w'.")
                self.controller.press_('d', hold_duration=2)
                self.controller.press_('w', hold_duration=2)
            
            dtc_location = updated_location
            print(f"Step: {step} completed.")
    
        # Reaching the royal area
        time.sleep(8)
       
        
    
        for _ in range(15):
            time.sleep(0.1)
            location = self.controller.find_template(self.inn_letter_2_path, confidence=0.7)
            
            if location is None:
                print("Continue search...")
                self.controller.drag_(500, 400, direction='left', move_distance=100, num_moves=3, button='right', delay=0.1)
            else:
                print("Turned face to inn hostess!")
                break
    
        self.controller.click_(512, 389, clicks=3)
        self.controller.press_('d', hold_duration=1.75)
        self.controller.click_(975, 380, clicks=8)
        #self.controller.click_(865, 600, clicks=4)
        #self.controller.click_(865, 400, clicks=3)
    
        for step in range(12):
            inn_location = self.controller.find_template(self.inn_letter_path, confidence=0.7)
            
            if inn_location is None:
                print("Inn location not found. Dragging screen to continue searching...")
                self.controller.drag_(500, 400, direction='left', move_distance=200, num_moves=3, button='right', delay=0.25)
            else:
                # Inn location found; perform 3 'w' presses and track location updates
                self.controller.press_('w', hold_duration=0.5)
                first_update = self.controller.find_template(self.inn_letter_path, confidence=0.75)
                
                self.controller.press_('w', hold_duration=0.5)
                second_update = self.controller.find_template(self.inn_letter_path, confidence=0.75)
                                
                
                # Compare the last two locations
                if second_update != first_update:
                    print("Inn location changed after final 'w' press. Continuing with process.")
                else:
                    print("Inn location unchanged after final 'w' press, stopping function as found=False.")
                    return False
                break
        else:
            # If 15 attempts are made without finding inn_location
            print("Failed to find inn location after 10 attempts. Exiting with found=False.")
            return False
    
        
    
        for _ in range(3):  # Max step 3
            time.sleep(0.5)
            self.controller.press_('z')
            time.sleep(1)
            commander_flag = self.controller.find_template(self.dtc_4_path, confidence=0.7)
            
            if commander_flag is None:
                self.controller.press_('d', hold_duration=3.5)
                self.controller.press_('w', hold_duration=3)
                self.controller.click_(880, 260, clicks=5)
            elif commander_flag:
                print("Near the DTC slot. now i need to check for item appraiser disney and click to it")
                self.controller.press_('r', hold_duration=3)
                self.controller.press_('s')
                self.controller.drag_(500, 400, direction='right', move_distance=200, num_moves=3, button='right', delay=0.25)               
                break
        
        for _ in range(5):
            appraiser_flag = self.controller.find_template(self.appraiser_path, confidence=0.7)
            time.sleep(1)
            if appraiser_flag is None:
                self.controller.press_('w', hold_duration=3)
            elif appraiser_flag:
                self.controller.click_(appraiser_flag[0]+75, appraiser_flag[1])
                time.sleep(1)
                
        time.sleep(20)
        self.controller.press_('d', hold_duration=2)
        self.controller.press_('w', hold_duration=3)
        self.controller.press_('a', hold_duration=0.5)
        self.controller.press_('w', hold_duration=3)
        self.controller.press_('a', hold_duration=0.75)
        self.controller.press_('w', hold_duration=2)
        self.controller.press_('z', hold_duration=1)
        time.sleep(1)
        ironclaw_loc = self.controller.find_template(self.ironclaw_path, confidence=0.7)
        self.controller.press_('z', hold_duration=1)
        time.sleep(1)
        ironclaw_loc = self.controller.find_template(self.ironclaw_path, confidence=0.7)
        if ironclaw_loc:
            print("ironclaw_loc arrived")
            found=True
            return found
        else:
            print("ironclaw_loc slot cant find, go to start")
            found=False
            return found
        
    def search_kalluga(self):
        found = False
        warp_loc = self.controller.find_template(self.warp_path, confidence=0.7)
        if warp_loc is None:
            found = False
            return found
        for k in range(10):
            
            print("clicking in left direction")
            self.controller.click_(warp_loc[0], warp_loc[1]+10, button='left')
            self.controller.click_(warp_loc[0], warp_loc[1]+10, button='right')
            #check kalluga loc
            kalluga_loc = self.controller.find_template(self.kalluga_path, confidence=0.75)
            time.sleep(1)
            if kalluga_loc:
                time.sleep(0.5)
                print("kalluaga gate opened")
                time.sleep(0.25)
                self.controller.click_(593, 266, clicks=2)                


                found = True
                time.sleep(0.5)
                
                break
            if not found:
                found=False
                kalluga_loc = None
                return found
        
    
    def go_slot_troll(self):
        self.controller.town_()
        time.sleep(2)
        found = False
        time.sleep(0.1)
        for _ in range(10):
            emc_gate_location = self.controller.find_template(self.emc_gate_path, confidence=0.7)
            if emc_gate_location is None:
                print("searching emc gate")
                self.controller.drag_(500, 400, direction='right', move_distance=250, num_moves=4, button='right', delay=0.05)
            else:
                updated_emc_gate_location = emc_gate_location
                print("emc gate found")
                self.controller.click_(updated_emc_gate_location[0] + 75, updated_emc_gate_location[1] + 75,
                                       clicks=8)
                
                print("near emc gate")
                break
        
        time.sleep(1)
        self.search_kalluga()
        time.sleep(0.5)
        #self.controller.click_(593, 266, clicks=2)                
        time.sleep(2)
        print("at kalluga")
        self.controller.press_("b", hold_duration=0.2)
        time.sleep(0.5)
        slave_loc = self.controller.find_template(self.slave_path, confidence=0.6)
        slave_loc_2 = self.controller.find_template(self.haunga_warrior_4_path, confidence=0.6)
        time.sleep(1)
        if slave_loc or slave_loc_2:
            print("slave found..")
            pass
        else:
            print("slave not found returning to town..")
            found = False
            return found
        
        time.sleep(0.5)
        
        for _ in range(10):
            time.sleep(0.25)
            lamb_location = self.controller.find_template(self.haunga_warrior_3_path, confidence=0.65)
            lamb_location_2 = self.controller.find_template(self.haunga_warrior_path, confidence=0.65)
            if lamb_location is None and lamb_location_2 is None:
                print("searching lamp")
                self.controller.drag_(500, 400, direction='left', move_distance=100, num_moves=2, button='right', delay=0.05)
            else:
                updated_lamb_location = lamb_location if lamb_location is not None else lamb_location_2
                print("lamb found")
                self.controller.click_(updated_lamb_location[0]+10, updated_lamb_location[1]+10,
                                       clicks=2)
                
                break
        if lamb_location is None and lamb_location_2 is None:
            found = False
            return found
        
        time.sleep(3.52)
        self.controller.press_("s")
        self.controller.press_("d", hold_duration=1.58)
        self.controller.press_("w", hold_duration=15.03)
        self.controller.press_("w", hold_duration=19.15)
        self.controller.press_("w", hold_duration=7.15)
        self.controller.press_("a", hold_duration=1)
        self.controller.press_("w", hold_duration=2)


       # exit()
        # self.controller.press_("a", hold_duration=0.32)
        # self.controller.press_("w", hold_duration=6)
        # self.controller.press_("s")
        time.sleep(0.5)
        
        
        self.controller.press_("z")
        time.sleep(0.5)
        troll_loc = self.controller.find_template(self.troll_path, confidence=0.65)
        troll_loc_2 = self.controller.find_template(self.troll_2_path, confidence=0.65)
        

        
        
        if troll_loc:
            print("slot 3 arrived")
            found=True
            return found
        else:
            print("troll slot cant find, go to start")
            found=False
            return found
        
        
    
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
    
    
                       
    
      
    def support_member(self, member_number):
       coordinates = (839, 168), (839, 222)
       crd = coordinates[0] if member_number == 2 else coordinates[1]
       self.controller.click_(crd[0], crd[1], clicks=3)
       self.controller.press_("0")
       time.sleep(1.25)
      
    
    def execute_action_asas(self, action):
        if action == 0:
            self.controller.press_("1")  # hp pot
            
        elif action == 1:
            self.support_member(2)
            print("sp m2")
            # Update support time and status for member 3
            self.member_2_last_support_time = time.time()
            self.member_2_status = True
        
        elif action == 2:
            self.support_member(3)
            print("sp m3")
            # Update support time and status for member 3
            self.member_3_last_support_time = time.time()
            self.member_3_status = True
        
        elif action == 3:
            time.sleep(0.5)
            self.controller.press_("9")
            self.controller.press_("9")

            time.sleep(2)
            self.wolf_last_time=time.time()
            self.wolf_status=True
            print("wolf 3")
          
        elif action == 4:
            self.controller.press_("r")
            self.controller.press_("2")
            self.controller.press_("r")
            self.controller.press_("3")
            self.controller.press_("r")
            self.controller.press_("4")
            self.controller.press_("r")
            self.controller.press_("5")
            self.controller.press_("r")
            self.controller.press_("2")
            self.controller.press_("r")

           
            
        
        elif action == 5:
            self.controller.press_("z")
            time.sleep(0.15)
            print("monster_finding")
            pass
        
        
            
        elif action == 6:
            pass
        
        elif action ==7:
            self.controller.press_("1")
            time.sleep(1)
            self.controller.press_("1")
            self.controller.town_()
            self.path_flag, _ = self.check_visiting_bank()
            self.slot_flag = False
            print(f"Initial path_flag status: {self.path_flag}")
                
            # Bank location and storage search
            while not self.path_flag:
                time.sleep(1)
                last_known_bank_location = self.find_bank_location_emc() #!!! banka loc burada
                storage_found = self.search_near_center()
                
                if storage_found:
                    self.path_flag = True  # Bank inventory opened
                    self.process_inventory_selective()
                    print("Bank found and inventory processed successfully")
                    break
                else:
                    #print("Inn hostess not found. Retrying bank location search...")
                    self.town_()  # Return to town before retrying
                    
            # Slot search after bank processes
            while self.path_flag and not self.slot_flag:
                self.town_()
                time.sleep(1)
                

                #slot_found = self.go_slot_lich() # slot burada
                slot_found = self.go_slot_troll() #!!!! slot burada !!!!

                if slot_found:
                    self.slot_flag = True
                    
            

            
        
        
            
    def tune_screen(self):
        for _ in range(10):
            game.controller.press_("z")
            loc = game.controller.find_template_non_gray(game.troll_path, confidence=0.65)
            if loc is None:
                game.controller.drag_(500, 400, direction='right', move_distance=40, num_moves=2, button='right', delay=0.05)
                time.sleep(0.5)
            else:
                break


       
    
    def step(self, action):
        if action not in self.action_space:
            raise ValueError("Invalid Action")
        
        
        self.step_check_durability()
        self.bank_flag = self.step_check_bank()
        self.execute_action_asas(action)
    
        # Pass the screenshot to get_states
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

    
        
        cv2.rectangle(screenshot, self.char_hp_bar_top_left, self.char_hp_bar_bottom_right, color=(0, 0, 255), thickness=2)
        # Draw monitoring rectangles
        cv2.rectangle(screenshot, self.monster_hp_bar_top_left, self.monster_hp_bar_bottom_right, color=(0, 0, 255), thickness=2)
        cv2.rectangle(screenshot, self.monster_name_top_left, self.monster_name_bottom_right, color=(0, 0, 255), thickness=2)
        
        self.episode_step_counter += 1
        
        self.durab_step_counter += 1
        self.bank_step_counter += 1
        
        print(f"Episode Step Counter: {self.episode_step_counter}, "
              f"Durab Step Counter: {self.durab_step_counter}, "
              f"Bank Step Counter: {self.bank_step_counter}")

        

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
        self.member_2_status = (current_time - self.member_2_last_support_time) < 601  # 600 seconds = 10 minutes
        # Update member 3 support status based on timer
        self.member_3_status = (current_time - self.member_3_last_support_time) < 601
        
        self.wolf_status = (current_time - self.wolf_last_time) < 121
        
        # Tüm değerleri bir sözlük içinde döndür
        return {
            "character_hp": normalized_hp,
            "monster_hp": normalized_monster_hp,
            "monster_flag": monster_flag,
            "member_2_support_status": self.member_2_status,
            "member_3_support_status": self.member_3_status,
            "wolf_status" : self.wolf_status
            
        }
    
    
    
    def action_decider_asas(self, game_status):
      
       
        
         
        # Emergency HP pot if HP is critically low (below 20%)
        if game_status['character_hp'] < 20:
            return 0  # hp_pot
        
        
        if not game_status['member_2_support_status']:
            #self.member_2_last_support_time = time.time()  # Reset timer
            return 1  # Action code for member 2 support
        
        # Check if member 3 needs support due to timer
        if not game_status['member_3_support_status']:
            #self.member_3_last_support_time = time.time()  # Reset timer
            return 2 # Action code for member 3 support
        
        # Check if member 3 needs support due to timer
        if not game_status['wolf_status']:
            #self.wolf_last_time = time.time()  # Reset timer
            return 3 # Action code for member 3 support
        
        
        if self.bank_flag == False:
            #go bank and empty inventory, go back to slot again
            return 7
        
        # Monster engagement logic
        if game_status['monster_flag']:  # If monster is present
            if game_status['monster_hp'] == 0:
                # Monster is dead, find new 
                time.sleep(1)
                self.monster_dead_flag = True
                return 5  # monster finding
            else:
                # Monster is alive, attack
                self.monster_dead_flag = False
                return 4  # attack
        else:
            # No monster present, need to find one
            return 5  # monster finding
        
        
        
        
        
        
           
           

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
        action = game.action_decider_asas(game_status)
        
        # Calculate remaining support times
        current_time = time.time()
        m2_remaining = max(0, 601 - (current_time - game.member_2_last_support_time))
        m3_remaining = max(0, 601 - (current_time - game.member_3_last_support_time))
        # Convert remaining times to min:sec format
        wolf_remaining=max(0, 121) - (current_time - game.wolf_last_time)
        
        
        m2_remain_formatted = format_time(m2_remaining)
        m3_remain_formatted = format_time(m3_remaining)
        wolf_remain_formatted = format_time(wolf_remaining)
       
        
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
            f"member_2_status : {game_status['member_2_support_status']}",
            f"member_3_status : {game_status['member_3_support_status']}",
            f"member_2_remain : {m2_remain_formatted}",
            f"member_3_remain: {m3_remain_formatted}",
            f"wolf_status: {game_status['wolf_status']}",
            f"wolf_remain: {wolf_remain_formatted}",
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