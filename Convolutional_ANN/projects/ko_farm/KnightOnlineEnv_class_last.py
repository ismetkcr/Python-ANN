import time
import cv2
import numpy as np
from game_controller import GameController
from get_window_2 import WindowCapture
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import math
from collections import deque


class KnightOnline:
    def __init__(self):
        # Initialize components
        self.controller = GameController('Knight OnLine Client')
        self.wincap = WindowCapture('Knight OnLine Client')
        self.inventory_model = load_model('inventory_model.h5')
        self.support_model = load_model('support_model.h5')
        self.durability_model = load_model('durability_model.h5')
        self.actions = 10
        self.action_space = [i for i in range(self.actions)]
        self.durability_check_step = 250
        self.support_check_step = 5
        self.bank_check_step = 500
        
        # FPS calculation
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.fps = 0
        self.fps_update_interval = 1.0  # Update FPS every second
        self.fps_history = deque(maxlen=10)  # Store last 10 FPS values for averaging
        
        
        # self.previous_fill_percentage = None
        # self.fill_percentage = None
        # self.reward = 0
        
        # self.zero_reward_actions = [5, 6, 9]
        
        self.episode_step_counter = 0
        self.support_step_counter = 0
        self.durab_step_counter = 0
        self.bank_step_counter = 0
        self.total_score = 0
        self.last_support_predictions = None
        self.last_durability_prediction = None  # Store the last known durability prediction
        self.last_bank_check = (None, None)  # Store the last bank check result as (bank_flag, predictions)

        # Performance optimization settings
        self.screenshot_cache = None
        self.screenshot_cache_time = 0
        self.screenshot_cache_duration = 0.03  # Cache screenshots for 30ms
        
        
        
        self.loop_time = time.time()
        
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
        
        
        
        #grid coordinates for support skills check        
        self.support_top_left = (14, 108)
        self.support_bottom_right = (46, 138)
        self.support_rects = self.draw_rectangle_grid(self.support_top_left,
                                                      self.support_bottom_right,
                                                      rows=1, cols=7, spacing=0)
        
        # Coordinates for monster HP bar
        self.monster_hp_bar_top_left = (402, 38)
        self.monster_hp_bar_bottom_right = (602, 56)
        
        # Coordinates for character HP bar
        self.char_hp_bar_top_left = (24, 33)
        self.char_hp_bar_bottom_right = (222, 49)
        
        # Coordinates for character MP bar
        self.char_mp_bar_top_left = (24, 51)
        self.char_mp_bar_bottom_right = (223, 65)
        
        #Coordinates for monster name 
        self.monster_name_top_left = (448,10)
        self.monster_name_bottom_right = (566, 29)
        

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
        
    def find_bank_location_eslant(self):
        """Attempts to locate the bank position and clicks a set number of times."""
        updated_location = self.controller.find_template(self.inn_letter_path, confidence=0.75)
        if updated_location is None:
            print("Bank location not found. Dragging screen to continue searching...")
            self.controller.drag_(500, 400, direction='left', move_distance=100, num_moves=4, button='left', delay=0.25)

        bank_location = updated_location
        print(f"Bank location found at: {bank_location}. Performing initial clicks.")

        for _ in range(3):
            if bank_location:
                print(f"Clicking at bank location: {bank_location}")
                self.controller.click_(bank_location[0] + 100, bank_location[1] + 100)
                time.sleep(1)

        for step in range(20):
            updated_location = self.controller.find_template(self.inn_letter_path, confidence=0.70)
            
            if updated_location is None:
                print("Bank location not found. Dragging screen to continue searching...")
                self.controller.drag_(500, 400, direction='left', move_distance=100, num_moves=3, button='right', delay=0.25)
            elif updated_location != bank_location:
                self.controller.click_(updated_location[0] + 30, updated_location[1] + 10)
            else:
                print("Bank location unchanged. Pressing 'd' and 'w'.")
                self.controller.press_('d', hold_duration=2)
                self.controller.press_('w', hold_duration=3)
            
            bank_location = updated_location
            print(f"Step: {step} completed.")
            
        time.sleep(5)
        self.controller.press_('b')
        return bank_location
    
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
            offset_y = self.center_y + k * 20
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
                offset_x = self.center_x + k * 20
                offset_y = self.center_y + k * 20
                print(f"Clicking at ({offset_x}, {offset_y}) in right-down direction")
                self.controller.click_(offset_x, offset_y, button='right')

                inn_use_storage_loc = self.controller.find_template(self.inn_use_storage_path, confidence=0.7)
                if inn_use_storage_loc:
                    self.controller.click_(inn_use_storage_loc[0] + 10, inn_use_storage_loc[1])
                    print("Inn hostess clicked successfully in right-down direction.")
                    found = True
                    break
                
        return found
    
    
    
    def search_laiba(self):
        pass
    
    
    def search_eslant(self):
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
            time.sleep(0.5)
            eslant_loc = self.controller.find_template(self.eslant_path, confidence=0.7)
            time.sleep(0.5)
            if eslant_loc:
                print("eslant_loc_ffound")
                time.sleep(0.5)
                self.controller.click_(eslant_loc[0], eslant_loc[1]-15, clicks=2)
                time.sleep(0.5)
                found = True
                break
            if not found:
                found = False
                eslant_loc = None
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
            time.sleep(0.5)
            kalluga_loc = self.controller.find_template(self.kalluga_path, confidence=0.7)
            
            time.sleep(0.5)
            if kalluga_loc:
                time.sleep(0.5)
                print("kalluaga gate opened")
                self.controller.click_(kalluga_loc[0], kalluga_loc[1]-15, clicks=2)
                time.sleep(0.5)

                found = True
                
                
                break
            if not found:
                found = False
                kalluga_loc = None
                return found
                    
        
   

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
                    self.controller.press_('enter')
        
        self.town_()
            
        return predictions
    
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
        print("at kalluga")
        self.controller.press_("b", hold_duration=0.2)
        time.sleep(0.5)
        slave_loc = self.controller.find_template(self.slave_path, confidence=0.7)
        if slave_loc:
            print("slave found..")
            pass
        else:
            print("slave not found returning to town..")
            found = False
            return found
        
        
        for _ in range(50):
            time.sleep(0.25)
            lamb_location = self.controller.find_template(self.haunga_warrior_3_path, confidence=0.65)
            if lamb_location is None:
                print("searching lamp")
                self.controller.drag_(500, 400, direction='right', move_distance=15, num_moves=2, button='right', delay=0.05)
            else:
                updated_lamb_location = lamb_location
                print("lamb found")
                self.controller.click_(updated_lamb_location[0]+10, updated_lamb_location[1]+10,
                                       clicks=2)
                
                break
        if lamb_location is None:
            found = False
            return found
        time.sleep(0.5)
        
        self.controller.press_('a', hold_duration=2.96)
        self.controller.press_('w', hold_duration=8.03)
        self.controller.press_('d', hold_duration=1)
        self.controller.press_('w', hold_duration=26.36)
        self.controller.press_('a', hold_duration=1)
        self.controller.press_('w', hold_duration=2.75)
        self.controller.press_('d', hold_duration=0.8)
        self.controller.press_('w', hold_duration=8.62)
        self.controller.press_('a', hold_duration=0.05)
        self.controller.press_('w', hold_duration=20.33)
            
            
                
        
        
        self.controller.press_('z', hold_duration=1)
        time.sleep(0.5)
        troll_loc = self.controller.find_template(self.troll_path, confidence=0.65)
        
    
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
    
    def check_support_actions(self):
        values_to_check = {
            0: '7', #buff
            1: '8', #ac
            2: '9', #kitap
            3: '0', #el
            4: None  # özel durum: 4 eksikse press_ts çalışacak
        }
    
        screenshot = self.wincap.get_screenshot()
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        processed_images = self.preprocess_rectangles(screenshot, self.support_rects)
        predictions = self.support_model.predict(processed_images, verbose=0)
        predictions = np.argmax(predictions, axis=1)
        print("Support predictions:", predictions)
    
        # Check for missing values
        missing_values = [value for value in values_to_check if value not in predictions]
    
        return predictions, missing_values
    
    def press_ts(self):
        time.sleep(0.1)
        self.controller.press_("f2")
        time.sleep(0.1)
        self.controller.press_('2')
        time.sleep(0.1)
        self.controller.click_(899, 303, clicks=2)
        time.sleep(0.1)
        self.controller.click_(430, 441)
        time.sleep(0.1)
        self.controller.press_("f1")
    
    def step_check_support_actions(self):
        # İlk missing value kontrolü için bir flag ekleyelim
        if not hasattr(self, 'support_missing_value_first_detected'):
            self.support_missing_value_first_detected = None
    
        # Check every 5 steps
        if self.support_step_counter % self.support_check_step == 0:
            predictions, missing_values = self.check_support_actions()
    
            if missing_values and self.monster_dead_flag == True:
                # Eğer 4 eksikse press_ts çalıştır
                if 4 in missing_values:
                    print("Missing value 4 detected. Running press_ts function.")
                    self.press_ts()
                else:
                    # İlk defa missing value tespit ettiyse kaydet ama bir şey yapma
                    if self.support_missing_value_first_detected is None:
                        self.support_missing_value_first_detected = self.support_step_counter
                        print("Missing values first detected. Will wait 5 steps before taking action.")
    
                    # 5 adım sonra tekrar kontrol
                    elif (self.support_step_counter - self.support_missing_value_first_detected) >= 0:
                        values_to_check = {
                            0: '7', #buff
                            1: '8', #ac
                            2: '9', #kitap
                            3: '0' #el
                        }
    
                        for value, output in values_to_check.items():
                            if value in missing_values:
                                self.controller.click_(512, 414)
                                self.controller.press_(output)
                                if value == 0:
                                    self.controller.press_("1")
                                time.sleep(1.25)
    
                        # Resetle
                        self.support_missing_value_first_detected = None
                        print("Taking support action for missing values.")
            else:
                # Missing value yoksa resetle
                self.support_missing_value_first_detected = None
    
            self.last_support_predictions = predictions
            print("SUPPORT ACTIONS CHECKED!!!!!!!!!!!!!!!!!")
        else:
            predictions = self.last_support_predictions  # Use last known predictions
    
        return predictions
    
    
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

        
    
    def reset(self): # burada gidilecek slot belirlenmeli ve ona göre girilmeli lich? ironclaw?
    #ek olarak gidilecek inn_hosstess nerede belirtilmeli emc ? eslant?
            
        
        print("Resetting in 5 seconds...")
        for i in range(5, 0, -1):
            print(i)
            time.sleep(1)
        print('ENV RESETTING...')
        
        self.episode_step_counter = 0
        self.support_step_counter = 0
        self.durab_step_counter = 0
        self.bank_step_counter = 0
        self.total_score = 0
        self.last_support_predictions = None
        self.last_durability_prediction = None  # S
        self.last_bank_check = (None, None)  # Store the last bank check result as (bank_flag, predictions)
        
        
        # dead_location = self.controller.find_template(self.dead_path, confidence = 0.7)
        # if dead_location:
        #     time.sleep(2)
        #     self.controller.click_(dead_location[0], dead_location[1]) # check et. 
        
        self.town_()
        time.sleep(1)
        self.press_ts() #!! 
        # Initial bank visit check
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
            self.controller.press_('7', hold_duration=1)
            time.sleep(1)
            self.controller.press_('8', hold_duration=1)
            time.sleep(1)

            #slot_found = self.go_slot_lich() # slot burada
            slot_found = self.go_slot_troll() #!!!! slot burada !!!!

            if slot_found:
                self.slot_flag = True
                #self.check_support_actions()
                #self.check_durability()

                print(" slot found")
                obs = self.get_screenshot()
                print("resetten dönen obs calıstı..")
                return obs
                
                
        
    def check_done(self):
        done = False
        
        location = self.controller.find_template(self.press_ok_path, confidence=0.7)
        bank_flag, predictions = self.step_check_bank()
        if location:
            print("öldüm :(")
            time.sleep(1)
            confirm_loc = self.controller.find_template(self.dead_path, confidence=0.7)
            time.sleep(1)
            self.controller.click_(confirm_loc[0], confirm_loc[1])
            done=True
            return done
        elif bank_flag is None:
            print("inventory is full, go bank to empty")
            done=True
            return done
            

    
    def step(self, action):
        if action not in self.action_space:
            raise ValueError("Invalid Action")
        
        
        self.step_check_durability()
        self.step_check_support_actions()
        done = self.check_done()
        self.execute_action(action)
    
        # Pass the screenshot to get_states
        
        self.episode_step_counter += 1
        self.support_step_counter += 1
        self.durab_step_counter += 1
        self.bank_step_counter += 1
        
        print(f"Episode Step Counter: {self.episode_step_counter}, "
              f"Support Step Counter: {self.support_step_counter}, "
              f"Durab Step Counter: {self.durab_step_counter}, "
              f"Bank Step Counter: {self.bank_step_counter}")


        screenshot = self.get_screenshot()

        # Draw monitoring rectangles
        cv2.rectangle(screenshot, self.monster_hp_bar_top_left, self.monster_hp_bar_bottom_right, color=(0, 0, 255), thickness=2)
        cv2.rectangle(screenshot, self.char_hp_bar_top_left, self.char_hp_bar_bottom_right, color=(0, 0, 255), thickness=2)
        cv2.rectangle(screenshot, self.char_mp_bar_top_left, self.char_mp_bar_bottom_right, color=(0, 0, 255), thickness=2)
        cv2.rectangle(screenshot, self.monster_name_top_left, self.monster_name_bottom_right, color=(0, 0, 255), thickness=2)

        return screenshot, done
    
    def get_states(self, screenshot): #burada slot belirtilmeli lich? ironclaw?
        """
        Oyun durumunu monitör eden fonksiyon.
        Monster HP, Character HP ve MP değerlerini hesaplar.
        
        Args:
            screenshot: Oyun ekranının anlık görüntüsü
            
        Returns:
            dict: Monster, HP ve MP durumlarını içeren sözlük
        """
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
        
        #loc = self.controller.find_monster_template(monster_flag_bar, self.ironclaw_path, confidence=0.65)
        #loc = self.controller.find_monster_template(monster_flag_bar, self.troll_path, confidence=0.65)
        #loc = self.controller.find_monster_template(monster_flag_bar, self.lich_path, confidence=0.65)
        #bu ayarlar troll slotu için.. diger slotlar için farklı ayarlar yapılacak
        #tw_loc = self.controller.find_template(self.troll_warrior_path, confidence=0.7)
        loc = self.controller.find_template(self.dtk_path, confidence=0.7) #dtk slot
        ancient_loc = self.controller.find_template(self.ancient_path, confidence=0.7) #dtk slot
        if ancient_loc:
            self.controller.press_("s", hold_duration=4)
        if loc is None:
        #if loc is None or tw_loc is not None: #!!!!!! this is for only troll..
            monster_flag = False
        else:
            monster_flag = True
        
        # Character HP işlemleri
        char_hp_bar = screenshot[
            self.char_hp_bar_top_left[1]:self.char_hp_bar_bottom_right[1],
            self.char_hp_bar_top_left[0]:self.char_hp_bar_bottom_right[0]
        ]
        hp_percentage = self.calculate_hp_fill_percentage(char_hp_bar)
        normalized_hp = self.normalize_to_0_100(hp_percentage, min_val=29.16, max_val=63.94)
        
        # Character MP işlemleri
        char_mp_bar = screenshot[
            self.char_mp_bar_top_left[1]:self.char_mp_bar_bottom_right[1],
            self.char_mp_bar_top_left[0]:self.char_mp_bar_bottom_right[0]
        ]
        mp_percentage = self.calculate_mp_fill_percentage(char_mp_bar)
        normalized_mp = self.normalize_to_0_100(mp_percentage, min_val=35.20, max_val=76.37)
        
        
        
        # Tüm değerleri bir sözlük içinde döndür
        return {
            "monster_flag": monster_flag,
            "monster_hp": normalized_monster_hp,            
            "character_hp": normalized_hp,
            "character_mp": normalized_mp
        }
                       
    
    
    def execute_action(self, action):
        
        if action == 0:
            self.controller.press_("1") # hp pot
            self.controller.press_("5")
            print("HP POT + heal")
        elif action == 1:
            self.controller.press_("2") # mp pot
            print("MP POT")
        
        elif action == 2:
            self.controller.press_("z")
            self.controller.press_("4")
            time.sleep(0.25)
            # self.controller.press_("6") 
            # time.sleep(0.25)


            print("MONSTER FINDING.")
        
        elif action == 3:
            self.controller.press_("4") # malice
            print("MALICE")
          
        elif action == 4:
            self.controller.press_("5") # heal
            print("HEAL")
        
        elif action == 5:
            pass
            print("pass")
        elif action == 6:
            self.controller.press_("r", hold_duration=0.1)
            self.controller.press_("3", hold_duration=0.1)
            self.controller.press_("6", hold_duration=0.1)

            
            #self.controller.press_("6")
            print("ATTACK")
    
    
            
    def calculate_hp_fill_percentage(self, hp_bar_array):
        # Mavi renk için tolerans
        blue_mask = (hp_bar_array[:, :, 0] < 10) & (hp_bar_array[:, :, 1] < 10) & (hp_bar_array[:, :, 2] > 200)
        # Kırmızı renk için tolerans
        red_mask = (hp_bar_array[:, :, 0] > 200) & (hp_bar_array[:, :, 1] < 10) & (hp_bar_array[:, :, 2] < 10)

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
        blue_mask = (mp_bar_array[:, :, 0] < 10) & (mp_bar_array[:, :, 1] < 10) & (mp_bar_array[:, :, 2] > 200)

        # Siyah renk için tolerans
        black_mask = (mp_bar_array[:, :, 0] < 10) & (mp_bar_array[:, :, 1] < 10) & (mp_bar_array[:, :, 2] < 10)
        
        # Kırmızı (dolu) piksel sayısı
        red_pixel_count = np.sum(blue_mask)
        # Siyah (boş) piksel sayısı
        black_pixel_count = np.sum(black_mask)
        
        # Toplam doluluk oranı
        total_pixels = red_pixel_count + black_pixel_count
        fill_percentage = (red_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0

        return fill_percentage

    def normalize_to_0_100(self, value, min_val=0, max_val=100):
        normalized_value = (value - min_val) / (max_val - min_val) * 100
        return max(0, min(100, normalized_value))
    
    def action_decider(self, game_status):
        """0 : hp_pot, 1 : mp_pot, 2 : monster finding,3 - malice, 4-heal 5-pass 6- attack"""
        #check hp first
        # The most important value is hp.. if hp below %40 return 5-heal
        #if hp  below % 20 return  0 - hp pot
        #if hp is okey second important thing is mana(mp).. 
        #if mp below %40 return 1 mp pot...
        #if this 2 condition is okey, then press z to find monster (if this condition is met 
        #game statues, monster flag shold return 1 then we can ready to throw malice so return 
        # 3(to throw malice) after throwing malice we can ready to attact so return 6)
        ##know while we attacking monster hp fill dcrease if its goes zero now we should return to find new monster
        #so press z  means monster finding so return 2, after monster finding game_status fonster flag should return
        #true and keep same things (attack, kill monster(means that monster hp 0) and find new monster
        #while doing this check for hp and mp conditions also..)
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
        # Critical HP check - Heal if HP is below 40%
        if game_status['character_hp'] < 35:
            return 4  # heal
           
        # MP management - Use pot if MP is below 40%
        if game_status['character_mp'] < 20:
            return 1  # mp_pot
        
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
                return 6  # attack
        else:
            # No monster present, need to find one
            return 2  # monster finding
        
   
    
        
            

step = 1
# repeat = 3
    
if __name__ == '__main__':
    game = KnightOnline()
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)
#     print('Starting...')
   
    
    while True:
        if step<=1:
            game.go_slot_haunga_warrior()
            #game.step(1)
            step+=1
        screenshot = game.get_screenshot()
       
        # #Draw monitoring rectangles
        cv2.rectangle(screenshot, game.monster_hp_bar_top_left, game.monster_hp_bar_bottom_right, color=(0, 0, 255), thickness=2)
        cv2.rectangle(screenshot, game.char_hp_bar_top_left, game.char_hp_bar_bottom_right, color=(0, 0, 255), thickness=2)
        cv2.rectangle(screenshot, game.char_mp_bar_top_left, game.char_mp_bar_bottom_right, color=(0, 0, 255), thickness=2)
        
        # # Process monster HP
        # monster_hp_bar = screenshot[
        #     game.monster_hp_bar_top_left[1]:game.monster_hp_bar_bottom_right[1],
        #     game.monster_hp_bar_top_left[0]:game.monster_hp_bar_bottom_right[0]
        # ]
        # monster_hp_percentage = game.calculate_hp_fill_percentage(monster_hp_bar)
        # if monster_hp_percentage == 100:
        #     monster_flag = False
        #     print("monster tutulmadı..")
        # else:
        #     monster_flag = True
        #     normalized_monster_hp = game.normalize_to_0_100(monster_hp_percentage, min_val=23.27, max_val=81.18)
        #     print(f"Monster HP Doluluk Oranı: {normalized_monster_hp:.2f}%")
        
        # # Process character HP
        # char_hp_bar = screenshot[
        #     game.char_hp_bar_top_left[1]:game.char_hp_bar_bottom_right[1],
        #     game.char_hp_bar_top_left[0]:game.char_hp_bar_bottom_right[0]
        # ]
        # hp_percentage = game.calculate_hp_fill_percentage(char_hp_bar)
        # normalized_hp = game.normalize_to_0_100(hp_percentage, min_val=29.16, max_val=63.94)
        # print(f"HP Doluluk Oranı: {normalized_hp:.2f}%")
        
        # # Process character MP
        # char_mp_bar = screenshot[
        #     game.char_mp_bar_top_left[1]:game.char_mp_bar_bottom_right[1],
        #     game.char_mp_bar_top_left[0]:game.char_mp_bar_bottom_right[0]
        # ]
        # mp_percentage = game.calculate_mp_fill_percentage(char_mp_bar)
        # normalized_mp = game.normalize_to_0_100(mp_percentage, min_val=35.20, max_val=76.36)
        # print(f"MP Doluluk Oranı: {normalized_mp:.2f}%")
        
        
        
        
        
        # Display the screenshot
        cv2.imshow('Computer Vision', screenshot)
    
        if cv2.waitKey(1) == ord('q'):
            print("Exiting search loop.")
            cv2.destroyAllWindows()
            break



