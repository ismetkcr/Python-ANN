import time
import cv2
import numpy as np
from game_controller import GameController
from get_window_2 import WindowCapture
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import math
from collections import deque
from env_wrappers import RepeatActionAndMaxFrameKnightOnline, StackFramesKnightOnline


class KnightOnline:
    def __init__(self):
        # Initialize components
        self.controller = GameController('Knight Online Client')
        self.wincap = WindowCapture('Knight Online Client')
        self.inventory_model = load_model('inventory_model.h5')
        self.support_model = load_model('support_model.h5')
        self.durability_model = load_model('durability_model.h5')
        self.actions = 10
        self.action_space = [i for i in range(self.actions)]
        self.durability_check_step = 250
        self.support_check_step = 25
        self.bank_check_step = 500
        # FPS calculation
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.fps = 0
        self.fps_update_interval = 1.0  # Update FPS every second
        self.fps_history = deque(maxlen=10)  # Store last 10 FPS values for averaging
        
        
        self.previous_fill_percentage = None
        self.fill_percentage = None
        self.reward = 0
        
        self.zero_reward_actions = [5, 6, 9]
        
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

        self.max_steps = 20 # max step for inventory search
        self.has_predicted = False
        self.locked_slots = set([0, 1, 2, 3])
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
        
        # #draw box for monster
        # self.monster_top_left = (513, 15)
        # self.monster_bottom_right = (593, 27)
        # self.monster_rect = [(self.monster_top_left, self.monster_bottom_right)]
        

    def town_(self):
        time.sleep(1)
        
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
        
    
        
        
    def find_bank_location(self):
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
            updated_location = self.controller.find_template(self.inn_letter_path, confidence=0.75)
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
    
    def display_predictions(self, screenshot, rectangles, predictions, locked_slots):
        """Overlay predictions on the screenshot within the rectangles."""
        for idx, (rect, pred) in enumerate(zip(rectangles, predictions)):
            center_x = rect[0] + (rect[2] - rect[0]) // 2
            center_y = rect[1] + (rect[3] - rect[1]) // 2

            
            # Change text color for locked slots
            text_color = (255, 0, 0) if idx in locked_slots else (255, 255, 255)
            
            text = str(pred)
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            text_x = center_x - text_width // 2
            text_y = center_y + text_height // 2
            
            cv2.putText(screenshot, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(screenshot, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)


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
    
    def go_slot(self):
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
                self.controller.drag_(500, 400, direction='right', move_distance=100, num_moves=2, button='right', delay=0.5)
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
    
        for step in range(15):
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
            print("Failed to find inn location after 15 attempts. Exiting with found=False.")
            return False
    
        
    
        for _ in range(3):  # Max step 3
            time.sleep(0.5)
            self.controller.press_('z')
            time.sleep(1)
            commander_flag = self.controller.find_template(self.dtc_4_path, confidence=0.7)
            
            if commander_flag is None:
                self.controller.press_('d', hold_duration=3)
                self.controller.press_('w', hold_duration=3)
                self.controller.click_(880, 260, clicks=5)
            elif commander_flag:
                print("Slot found")
                self.controller.press_('r', hold_duration=3)
                self.controller.press_('s')
                self.controller.drag_(500, 400, direction='right', move_distance=200, num_moves=3, button='right', delay=0.25)

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

        # Increment the counter
        return bank_flag, predictions
    
    def check_support_actions(self):
        values_to_check = {
            0: '7', #buff
            1: '8', #ac
            2: '9', # kitap
            3: '0' #el
        }

        for attempt in range(5):  # Attempt loop
            print(f"Support check attempt {attempt + 1}/10")
            screenshot = self.wincap.get_screenshot()
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
            processed_images = self.preprocess_rectangles(screenshot, self.support_rects)
            predictions = self.support_model.predict(processed_images, verbose=0)
            predictions = np.argmax(predictions, axis=1)
            print(f"Predictions for attempt {attempt + 1}:", predictions)

            # Check for missing values
            missing_values = [value for value in values_to_check if value not in predictions]

            if missing_values:
                #print(f"Attempt {attempt + 1} - Missing values:", missing_values)
                for value, output in values_to_check.items():
                    if value in missing_values:
                        self.controller.click_(512, 414)
                        self.controller.press_(output)
                        time.sleep(1.25)
                        
            else:
                #print(f"Attempt {attempt + 1} - All values present in matrix. Stopping checks.")
                self.last_support_predictions = predictions  # Update last predictions
                return predictions  # Stop checks if all values are present

            time.sleep(0.05)  # Pause between attempts

        # Return the last predictions after all attempts
        self.last_support_predictions = predictions
        return predictions

    def step_check_support_actions(self):
        # Check every 10 steps
        if self.support_step_counter % self.support_check_step == 0:
            predictions = self.check_support_actions()
            print("SUPPORT ACTIONS CHECKED!!!!!!!!!!!!!!!!!")
        else:
            predictions = self.last_support_predictions  # Use last known predictions

        # Increment the counter
        
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
        
    
    def get_screenshot(self):
        """Optimized screenshot capture with caching"""
        current_time = time.time()
        
        # Return cached screenshot if it's recent enough
        if (self.screenshot_cache is not None and 
            current_time - self.screenshot_cache_time < self.screenshot_cache_duration):
            return self.screenshot_cache

        # Get new screenshot and update cache
        self.screenshot_cache = self.wincap.get_screenshot()
        self.screenshot_cache_time = current_time
        return self.screenshot_cache
    
    def update_fps(self):
        self.fps_counter += 1
        current_time = time.time()
        elapsed_time = current_time - self.fps_start_time
    
        if elapsed_time >= self.fps_update_interval:
            # Calculate current FPS and update history
            self.fps = self.fps_counter / elapsed_time
            self.fps_history.append(self.fps)
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            print(f"Current FPS: {self.fps:.2f} | Average FPS: {avg_fps:.2f}")
    
            # Reset counter and start time for the next interval
            self.fps_counter = 0
            self.fps_start_time = current_time

        
    
    def reset(self):
            
        
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
        self.previous_fill_percentage = None
        self.fill_percentage = None
        self.reward = 0
        
        # dead_location = self.controller.find_template(self.dead_path, confidence = 0.7)
        # if dead_location:
        #     time.sleep(2)
        #     self.controller.click_(dead_location[0], dead_location[1]) # check et. 
        
        self.town_()
        time.sleep(1)
        
        # Initial bank visit check
        self.path_flag, _ = self.check_visiting_bank()
        self.slot_flag = False
        print(f"Initial path_flag status: {self.path_flag}")
            
        # Bank location and storage search
        while not self.path_flag:
            time.sleep(1)
            last_known_bank_location = self.find_bank_location()
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
            dtc_found = self.go_slot()
            
            if dtc_found:
                self.slot_flag = True
                #self.check_support_actions()
                #self.check_durability()

                print("DTC slot found and support actions checked")
                obs = self.get_obs()
                print("resetten dönen obs calıstı..")
                return obs
                
                
                
        

    
    
    def calculate_reward_inc(self, action):
        if action == 6:
            self.reward = 0
            return self.reward
        
        if action == 9:
            self.reward = 0
            return self.reward
        
        
        screenshot = self.wincap.get_screenshot()
        # Coordinates for monster HP bar
        hp_bar_top_left = (406, 40)
        hp_bar_bottom_right = (600, 55)
        
        # Crop the HP bar area and convert to grayscale
        cropped_hp_bar = screenshot[hp_bar_top_left[1]:hp_bar_bottom_right[1], 
                                  hp_bar_top_left[0]:hp_bar_bottom_right[0]]
        gray_cropped_hp_bar = cv2.cvtColor(cropped_hp_bar, cv2.COLOR_BGR2GRAY)
        self.hp_bar_array = gray_cropped_hp_bar
        
        commander_flag = self.controller.find_template(self.dtc_3_path, confidence=0.7)
        
        # Initialize reward
        self.reward = 0
        
        # Store current fill percentage as previous before updating
        if not hasattr(self, 'previous_fill_percentage'):
            self.previous_fill_percentage = None
        if not hasattr(self, 'fill_percentage'):
            self.fill_percentage = None
            
        old_percentage = self.fill_percentage
        
        # If no commander flag, HP is full
        if commander_flag is None:
            self.fill_percentage = 100.0
            self.previous_fill_percentage = self.fill_percentage
            return 0
        
        # Threshold the HP bar to separate filled and empty parts
        _, binary_hp_bar = cv2.threshold(self.hp_bar_array, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours of the filled HP area
        contours, _ = cv2.findContours(binary_hp_bar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (filled portion of HP bar)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, width, height = cv2.boundingRect(largest_contour)
            
            # Calculate fill percentage
            full_hp_bar_width = self.hp_bar_array.shape[1]
            current_fill = (width / full_hp_bar_width) * 100
            
            # Clamp fill percentage between 0 and 100
            self.fill_percentage = max(0, min(100, current_fill))
            
            # Calculate reward only if we have a previous value
            if old_percentage is not None:
                # Calculate HP decrease (positive when HP decreases)
                hp_change = old_percentage - self.fill_percentage
                
                # Only give positive reward when HP decreases
                if hp_change > 0:
                    # Scale reward based on the percentage change
                    self.reward = hp_change * 250  # Adjust multiplier as needed
                else:
                    self.reward = 0
        else:
            # No HP bar detected, assume empty
            self.fill_percentage = 0
            if old_percentage is not None:
                self.reward = old_percentage  # Reward for depleting remaining HP
            
        # Update previous fill percentage
        self.previous_fill_percentage = self.fill_percentage
        
        return self.reward

    
    
   
    
    def check_done_and_calculate_reward(self):
        location = self.controller.find_template(self.dead_path, confidence=0.7)
        bank_flag, predictions = self.step_check_bank()
        info = location
        if location:
            print("ÖLDÜM AMK, CEZALANDIRILDIM")
            time.sleep(2)
            done = True
            reward = -2500
            info = location
        
            
           
            
        elif bank_flag == None:
            done = True
            print("BAŞARDIM AMK, ÖDÜLLENDIRILDIM")          
            reward = 2500
        elif self.episode_step_counter >= 50 and self.total_score < 0:
            time.sleep(10)
            location = self.controller.find_template(self.dead_path, confidence=0.7)
            if location:
                time.sleep(0.5)
                self.controller.click_(location[0], location[1])
            print("Beyinsizim Hala Atak Yapamadım")
            done = True
            reward = -2500
        else:
            #print("Hayat Bi Şekil Devam Ediyor")
            done = False
            reward = 0
        
        return done, reward, info
        
        
    
    def step(self, action):
        if action not in self.action_space:
            raise ValueError("Invalid Action")
        durability_preds = self.step_check_durability()
        support_preds = self.step_check_support_actions()
        bank_flag, predictions = self.step_check_bank()       
        self.execute_action(action)
        #info = ([durability_preds, support_preds])
        
        
        # Calculate reward
        reward = self.calculate_reward_inc(action)
        print("reward from inc________________________________________!!", reward)
        #valic_acts = [2,4,8]
        
        if action == 0:     
            reward -= 0.1
        elif action == 1:
            reward -= 0.05
       
        else:
            reward-=0.033
        
        
        
        # Batch process checks when possible
        
        done, d_reward, info = self.check_done_and_calculate_reward()
        reward += d_reward
        #reward -= 50
        
        
        
        
        
        
        
        
        self.total_score += reward
        self.episode_step_counter += 1
        self.support_step_counter += 1
        self.durab_step_counter += 1
        self.bank_step_counter += 1
        
        observation = self.get_obs()
        
        return observation, reward, done, info
        
        
        
       
    
    def get_obs(self):
        """Optimized observation processing"""
        screenshot = self.get_screenshot()  # Use cached screenshot
        
        # Resize and normalize
        resized_image = cv2.resize(screenshot, (128, 128), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image / 255.0  # Normalize observation
        
        # Ensure 3-channel format for OpenCV display
        if len(resized_image.shape) == 2:  # If grayscale, convert to 3 channels
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
        
        return resized_image  # Return without adding extra dimension
   
    
    
    def execute_action(self, action):
        
        if action == 0:
            self.controller.press_("1") # hp pot
            print("HP POT BASILDI")
        elif action == 1:
            self.controller.press_("2") # mp pot
            print("MP POT BASILDI")
        
        elif action == 2:
            self.controller.press_("z")
            self.controller.press_("r")
            self.controller.press_("3") # attack
            print("z + R + ATTACK ......")
        
        elif action == 3:
            self.controller.press_("4") # malice
            print("MALICE ............")
          
        elif action == 4:
            self.controller.press_("5") # heal
            print("HEAL................")
        
        elif action == 5:
            pass
            print("pass................")


        elif action == 6:
            self.controller.drag_(500, 400, direction='left', move_distance=200, num_moves=1, button='right', delay=0.05)

            print("drag secreen")

        
        elif action == 7: # get monster and run to it
            self.controller.press_('z')
            self.controller.press_('6')
            print("Z + 6 + UZAKTAN CAGIR................")

        
        elif action == 8:
            self.controller.press_('r')
            self.controller.press_('3')
            print("R + ATTACK...")
        
        elif action == 9:
            print("CLİCK MIDDLE TO MOVE")
            self.controller.click_(512, 414, clicks=1)

            

            

step = 1
repeat = 3
    
if __name__ == '__main__':
    
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)
#     print('Starting...')
    env = KnightOnline()
    found = env.go_slot()
    print(found)
    env = RepeatActionAndMaxFrameKnightOnline(env)
    env = StackFramesKnightOnline(env, repeat)
    while True:
        #screenshot = env.wincap.get_screenshot()
        # cv2.putText(screenshot, f'Mouse Position: ({env.wincap.mouse_x}, {env.wincap.mouse_y})', 
        #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        
        #fps = 1 / (curr_time - env.loop_time)
        
        #cv2.putText(screenshot, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if step == 1:
            env.town_()
            env.go_slot()
            step += 4
            
        #     #prediction2s = env.check_support_actions()  # Removed step counter as it was limiting reset to once only
        #     #bank_flag, predictions = env.check_visiting_bank()
        #     #durability_check = env.check_durability()
        #     #print(durability_check)
        #     #predictions = env.check_support_actions()
        #     #print(predictions)
        #     step += 19
        # action = 1
        # observation, reward, done, info = env.step(action)
        # print("observation shape: expected(221, 221, 3)", observation.shape)
        # print("total_score:", env.total_score)
        # print("done_flag", done)
        # if done == True:
        #     print("env should reset here..")
        
        #print(env.episode_step_counter)
        #env.update_fps()
        #print(observation[:, :, :1])
        
        #cv2.imshow('Computer Vision', screenshot)
        #cv2.imshow('Computer Vision', observation[:, :, :3])

        if cv2.waitKey(1) == ord('q'):
            print("Exiting search loop.")
            cv2.destroyAllWindows()
            break