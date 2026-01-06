# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 00:08:49 2025

@author: ismt
"""

import cv2
import numpy as np
import win32gui
import win32con
import pydirectinput
from PIL import ImageGrab
import time
import re

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    OCR_AVAILABLE = True
    print("✅ Tesseract OCR başarıyla yüklendi!")
except ImportError:
    OCR_AVAILABLE = False
    print("❌ pytesseract yüklü değil!")
except Exception as e:
    OCR_AVAILABLE = False
    print(f"❌ Tesseract hatası: {e}")

class GameController:
    def __init__(self, window_name):
        self.window_name = window_name
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception(f'Window not found: {window_name}')
        
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.window_x, self.window_y, self.window_right, self.window_bottom = window_rect
        self.game_width = self.window_right - self.window_x
        self.game_height = self.window_bottom - self.window_y

        win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
        pydirectinput.FAILSAFE = True
        
        # Movement history tracking for anti-stuck W mechanism
        self.consecutive_w_presses = 0
        self.w_press_threshold = 6  # After 6 W presses, inject D
        self.max_w_sequence = 10    # Total W presses to track
        
        #OPEN NPC PATHS
        self.upgrade_item_anvil_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\upgrade_item_anvil.png"
        self.trade_armor_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\trade_armor.png"
        self.use_storage_bank_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\use_storage_bank.png"
        
        
        self.magic_anvil_path =  r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\magic_anvil.png"
        self.magic_anvil_2_path =  r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\magic_anvil_2.png"
        self.magic_anvil_3_path =  r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\magic_anvil_3.png"
        self.magic_anvil_4_path =  r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\magic_anvil_4.png"
        self.magic_anvil_5_path =  r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\magic_anvil_5.png"
        
        
        self.upgrade_succeed_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\upgrade_succeed.png"
        self.upgrade_succeed_2_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\upgrade_succeed_2.png"
        self.upgrade_succeed_3_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\upgrade_succeed_3.png"
        self.upgrade_succeed_4_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\upgrade_succeed_4.png"
        self.upgrade_succeed_5_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\upgrade_succeed_5.png"


        self.upgrade_failed_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\upgrade_failed.png"
        self.upgrade_failed_2_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\upgrade_failed_2.png"
        self.upgrade_failed_3_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\upgrade_failed_3.png"
        self.upgrade_failed_4_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\upgrade_failed_4.png"
        self.upgrade_failed_5_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\upgrade_failed_5.png"
        self.upgrade_failed_6_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\upgrade_failed_6.png"

        
        
        self.upgrade_cannot_perform_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\upgrade_cannot_perform.png"
        self.upgrade_cannot_perform_2_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\upgrade_cannot_perform_2.png"
        self.upgrade_cannot_perform_3_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\upgrade_cannot_perform_3.png"

        

        #bank path
        self.bank_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\bank.png"
        self.bank_2_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\bank_2.png"
        self.bank_3_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\bank_3.png"
        self.bank_4_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\bank_4.png"
        self.bank_5_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\bank_5.png"






        #PURCHASE ARMOR PATH
        self.purchase_armor_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\purchase_armor.png"
        self.confirm_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\confirm.png"
        self.confirm_anvil_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\confirm_anvil.png"
            
        
        self.helard_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\helard.png"
        self.helard_2_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\helard_2.png"

        
 
        
        self.gargamet_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\gargamet.png"
        self.gargamet_2_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\gargamet_2.png"
        self.gargamet_3_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\gargamet_3.png"
        self.gargamet_4_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\gargamet_4.png"
        self.gargamet_5_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\gargamet_5.png"
        self.gargamet_6_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\gargamet_6.png"




        

        
        self.hesta_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\hesta.png"
        self.hesta_2_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\hesta_2.png"
        self.hesta_3_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\hesta_3.png"
        self.hesta_4_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\hesta_4.png"
        self.hesta_5_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\hesta_5.png"
        self.hesta_6_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\hesta_6.png"
        self.hesta_7_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\hesta_7.png"
        self.hesta_8_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\hesta_8.png"







        
        self.inventory_check_path = r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\inventory_check.png"

    def move_(self, x, y, duration=0.1, relative=False):
        # Koordinatların oyun alanı içinde olup olmadığını kontrol et
        if not (0 <= x <= self.game_width and 0 <= y <= self.game_height):
            print("Hedef koordinatlar oyun alanı dışında, işlem iptal edildi.")
            return 512, 384   # Tıklama işlemini durdur
        
        # Offset değerleri
        offset_x = 0
        offset_y = 0
        
        # Absolut hedef koordinatları hesapla
        target_x = self.window_x + x - offset_x
        target_y = self.window_y + y - offset_y
        
        # Eğer göreceli hareket yapılacaksa
        if relative:
            current_x, current_y = pydirectinput.position()
            target_x = current_x + x
            target_y = current_y + y
        
        # Fareyi hedef konuma taşı
        pydirectinput.moveTo(target_x, target_y, duration=duration)
        return target_x, target_y

    def click_(self, x, y, button='left', clicks=1, interval=0.1):
        self.move_(x, y)
        time.sleep(0.1)
        for _ in range(clicks):
            pydirectinput.mouseDown(button=button)
            time.sleep(0.05)
            pydirectinput.mouseUp(button=button)
            if clicks > 1:
                time.sleep(interval)

    def drag_(self, start_x, start_y, direction='left', move_distance=50, num_moves=50, button='right', delay=0.05):
        self.move_(start_x, start_y)
        time.sleep(0.1)
        
        dx, dy = 0, 0
        if direction == 'left': dx = -move_distance
        elif direction == 'right': dx = move_distance
        elif direction == 'up': dy = -move_distance
        elif direction == 'down': dy = move_distance
        
        try:
            pydirectinput.mouseDown(button=button)
            for _ in range(num_moves):
                pydirectinput.move(dx, dy)
                time.sleep(delay)
        finally:
            pydirectinput.mouseUp(button=button)
            
    def press_(self, key, hold_duration=0.05):
        # Anti-stuck mechanism for consecutive W presses
        if key.lower() == 'w':
            self.consecutive_w_presses += 1
            print(f"🎮 W press #{self.consecutive_w_presses}/{self.max_w_sequence}")
            
            # If we've reached the threshold, inject a D press to break the pattern
            if self.consecutive_w_presses == self.w_press_threshold:
                print(f"🔄 ANTI-STUCK: Detected {self.w_press_threshold} consecutive W presses!")
                print("🎮 Injecting D press to break movement pattern...")
                
                # Execute the original W press first
                pydirectinput.keyDown(key)
                print("pressed key:", key)
                time.sleep(hold_duration)
                pydirectinput.keyUp(key)
                
                # Small delay between W and D
                time.sleep(0.1)
                
                # Inject D press to change direction
                pydirectinput.keyDown('d')
                print("pressed key: d (anti-stuck injection)")
                time.sleep(0.2)  # Slightly longer D press
                pydirectinput.keyUp('d')
                
                # Don't reset counter yet - let it continue to max_w_sequence
                return
            
            # Reset counter if we've reached max sequence
            elif self.consecutive_w_presses >= self.max_w_sequence:
                print(f"🔄 ANTI-STUCK: Completed {self.max_w_sequence} W press sequence, resetting counter")
                self.consecutive_w_presses = 0
        else:
            # Reset W counter for any non-W key press
            if self.consecutive_w_presses > 0:
                print(f"🔄 ANTI-STUCK: Non-W key '{key}' pressed, resetting W counter from {self.consecutive_w_presses}")
                self.consecutive_w_presses = 0
        
        # Execute normal key press
        pydirectinput.keyDown(key)
        print("pressed key:", key)
        time.sleep(hold_duration)
        pydirectinput.keyUp(key)

    def capture_game_window(self):
        bbox = (self.window_x, self.window_y, self.window_right, self.window_bottom)
        screenshot = ImageGrab.grab(bbox=bbox)
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    def find_template(self, template_path, confidence=0.95):
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise Exception(f"Template image not found at path: {template_path}")
        
        screenshot = self.capture_game_window()
        gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        result = cv2.matchTemplate(gray_screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= confidence:
            # Template'in boyutlarını al
            template_height, template_width = template.shape
            
            # Orta noktayı hesapla
            center_x = max_loc[0] + template_width // 2
            center_y = max_loc[1] + template_height // 2
            
            return (center_x, center_y)
        return None
    
    def find_template_non_gray(self, template_path, confidence=0.65):
        # Read template in grayscale
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise Exception(f"Template image not found at path: {template_path}")
        
        # Capture screenshot and convert to grayscale
        screenshot = self.capture_game_window()
        gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        # Perform template matching
        result = cv2.matchTemplate(gray_screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= confidence:
            return max_loc
        return None
    
    def find_monster_template(self, processed_screenshot, template_path, confidence=0.65):
        # Read the template image in color (RGB)
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if template is None:
            raise Exception(f"Template image not found at path: {template_path}")
        
        # Convert the processed screenshot and template to grayscale
        gray_screenshot = cv2.cvtColor(processed_screenshot, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Perform template matching
        result = cv2.matchTemplate(gray_screenshot, gray_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Return the location of the match if the confidence threshold is met
        if max_val >= confidence:
            return max_loc
        
        return None

    def read_item_level_from_slot(self, slot_index, anvil_rects):
        """Read the item level (+X) from a specific anvil slot using OCR."""
        if not OCR_AVAILABLE:
            print("   ⚠️ OCR not available, cannot read item level")
            return None
            
        try:
            # Capture screenshot
            screenshot = self.capture_game_window()
            
            # Get the specific slot rectangle
            slot_rect = anvil_rects[slot_index]
            x1, y1 = slot_rect[0]
            x2, y2 = slot_rect[1]
            roi = screenshot[y1:y2, x1:x2]
            
            # OCR preprocessing
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            enlarged = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            contrast = cv2.convertScaleAbs(enlarged, alpha=2.0, beta=50)
            _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR config for reading +X levels
            config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=+0123456789'
            text = pytesseract.image_to_string(thresh, config=config)
            
            # Clean and extract level
            cleaned = text.strip().replace('\n', '').replace(' ', '')
            
            # Look for +X pattern
            level_match = re.search(r'\+(\d+)', cleaned)
            if level_match:
                level = int(level_match.group(1))
                print(f"   🔍 Slot {slot_index} OCR result: +{level}")
                return level
            
            # Look for just numbers (in case + is missed)
            numbers = re.findall(r'\d+', cleaned)
            if numbers:
                level = int(numbers[0])
                print(f"   🔍 Slot {slot_index} OCR result (no +): {level}")
                return level
                
            print(f"   ❌ Could not read item level from slot {slot_index}, OCR text: '{cleaned}'")
            return None
            
        except Exception as e:
            print(f"   ❌ Error reading item level from slot {slot_index}: {e}")
            return None

    def read_item_level_from_slot_template(self, slot_index, anvil_rects):
        """Read the item level (+X) from a specific anvil slot using template matching like navigate_rectangles."""
        try:
            # Get the specific slot rectangle
            slot_rect = anvil_rects[slot_index]
            center_x = (slot_rect[0][0] + slot_rect[1][0]) // 2
            center_y = (slot_rect[0][1] + slot_rect[1][1]) // 2
            
            # Move mouse to the slot (like navigate_rectangles does)
            print(f"   🖱️ Moving mouse to slot {slot_index} center: ({center_x}, {center_y})")
            self.move_(center_x, center_y)
            time.sleep(0.3)  # Brief pause for mouse movement
            
            # Template paths for upgrade levels (same as navigate_rectangles)
            template_paths = {
                r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\plus_one.png": 1,
                r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\plus_two.png": 2,
                r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\plus_three.png": 3,
                r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\plus_four.png": 4,
                r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\plus_five.png": 5,
                r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\plus_six.png": 6,
                r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\plus_seven.png": 7
            }
            
            # Check each template
            for template_path, level in template_paths.items():
                found = self.find_template(template_path, confidence=0.9)
                if found:
                    print(f"   🔍 Slot {slot_index} template result: +{level}")
                    return level
            
            # No upgrade level found
            print(f"   🔍 Slot {slot_index} template result: No upgrade detected (base item)")
            return 0
            
        except Exception as e:
            print(f"   ❌ Error reading item level from slot {slot_index}: {e}")
            return None

    def perform_single_upgrade_with_template_detection(self, anvil_rects, locked_slots, item_slot_to_upgrade, expected_initial_level, inventory_data=None):
        """
        Perform upgrade and detect success by checking item level increase via template matching.
        
        Parameters:
        - anvil_rects: Anvil rectangle coordinates
        - locked_slots: Set of locked slot indices 
        - item_slot_to_upgrade: Index of the item slot to upgrade
        - expected_initial_level: The current level of the item (e.g., 4 for +4 item)
        
        Returns:
        - str: "success", "failed", "cannot_perform", or "error"
        """
        print(f"\n🔧 PERFORMING UPGRADE ON SLOT {item_slot_to_upgrade} (Template Detection)")
        
        try:
            # Step 1: Read initial item level to confirm
            print("   📋 Step 1: Reading initial item level...")
            initial_level = self.read_item_level_from_slot_template(item_slot_to_upgrade, anvil_rects)
            if initial_level is None:
                print("   ❌ Could not read initial item level!")
                return "error"
            
            if initial_level != expected_initial_level:
                print(f"   ⚠️ Warning: Expected +{expected_initial_level}, but found +{initial_level}")
            
            # Step 2: Find and right-click on upgrade scroll
            upgrade_scroll_slot = 0  # Default to slot 0
            
            # If we have inventory data, find the actual upgrade scroll location
            if inventory_data:
                for slot_idx, slot_data in inventory_data.items():
                    if slot_data.get('prediction') == 0:  # 0 = upgrade scroll
                        upgrade_scroll_slot = slot_idx
                        break
            
            print(f"   📜 Step 2: Selecting upgrade scroll from slot {upgrade_scroll_slot}...")
            scroll_rectangle = anvil_rects[upgrade_scroll_slot]
            self.click_middle_of_rectangle(scroll_rectangle, button='right')
            time.sleep(0.25)
            
            # Step 3: Right-click on target item to upgrade
            print(f"   🎯 Step 3: Selecting item to upgrade from slot {item_slot_to_upgrade}...")
            item_rectangle = anvil_rects[item_slot_to_upgrade]
            self.click_middle_of_rectangle(item_rectangle, button='right')
            time.sleep(0.25)
            
            # Step 4: Look for confirm_anvil and click it
            print("   ✅ Step 4: Looking for confirmation button...")
            confirm_pos = self.find_template(self.confirm_anvil_path, confidence=0.85)
            if confirm_pos:
                self.click_(confirm_pos[0], confirm_pos[1], button='left')
                print(f"   ✅ Clicked confirm at: {confirm_pos}")
                time.sleep(0.5)
            else:
                print("   ❌ Confirm button not found!")
                return "error"
            
            # Step 5: Press Enter to start upgrade
            print("   🚀 Step 5: Pressing Enter to start upgrade...")
            pydirectinput.press('enter')
            
            # Step 6: Wait for upgrade result (5 seconds)
            print("   ⏳ Step 6: Waiting 5 seconds for upgrade result...")
            time.sleep(5.0)
            
            # Step 7: Check item level after upgrade
            print("   🔍 Step 7: Reading item level after upgrade...")
            final_level = self.read_item_level_from_slot_template(item_slot_to_upgrade, anvil_rects)
            
            if final_level is None:
                print("   💥 UPGRADE FAILED - ITEM BURNED! (Cannot read level = item destroyed)")
                return "failed"
            
            if final_level == initial_level + 1:
                print(f"   🎉 UPGRADE SUCCESSFUL! (+{initial_level} → +{final_level})")
                return "success"
            
            # Special case: +6 → +7 might show as +6 → +2 due to template detection issues
            if initial_level == 6 and final_level == 2:
                print(f"   🎉 UPGRADE SUCCESSFUL! (+6 → +7, detected as +2)")
                return "success"
            
            if final_level == initial_level:
                print(f"   🚫 UPGRADE LIMIT REACHED - Level unchanged (+{initial_level})")
                print("   🎮 Initiating game quit sequence...")
                return "cannot_perform"
            
            if final_level == 0 and initial_level > 0:
                print(f"   💥 UPGRADE FAILED - ITEM BURNED! (+{initial_level} → destroyed)")
                return "failed"
            
            # Any other unexpected case
            print(f"   ❓ Unexpected result: +{initial_level} → +{final_level}")
            return "failed"
            
        except Exception as e:
            print(f"   ❌ Error during upgrade process: {e}")
            return "error"

    def perform_single_upgrade_with_ocr_detection(self, anvil_rects, locked_slots, item_slot_to_upgrade, expected_initial_level):
        """
        Perform upgrade and detect success by checking item level increase via OCR.
        
        Parameters:
        - anvil_rects: Anvil rectangle coordinates
        - locked_slots: Set of locked slot indices 
        - item_slot_to_upgrade: Index of the item slot to upgrade
        - expected_initial_level: The current level of the item (e.g., 4 for +4 item)
        
        Returns:
        - str: "success", "failed", "cannot_perform", or "error"
        """
        print(f"\n🔧 PERFORMING UPGRADE ON SLOT {item_slot_to_upgrade} (OCR Detection)")
        
        try:
            # Step 1: Read initial item level to confirm
            print("   📋 Step 1: Reading initial item level...")
            initial_level = self.read_item_level_from_slot(item_slot_to_upgrade, anvil_rects)
            if initial_level is None:
                print("   ❌ Could not read initial item level!")
                return "error"
            
            if initial_level != expected_initial_level:
                print(f"   ⚠️ Warning: Expected +{expected_initial_level}, but found +{initial_level}")
            
            # Step 2: Right-click on upgrade scroll (slot 0 - locked slot)
            print("   📜 Step 2: Selecting upgrade scroll from slot 0...")
            scroll_rectangle = anvil_rects[0]  # Slot 0 has the upgrade scroll
            self.click_middle_of_rectangle(scroll_rectangle, button='right')
            time.sleep(0.25)
            
            # Step 3: Right-click on target item to upgrade
            print(f"   🎯 Step 3: Selecting item to upgrade from slot {item_slot_to_upgrade}...")
            item_rectangle = anvil_rects[item_slot_to_upgrade]
            self.click_middle_of_rectangle(item_rectangle, button='right')
            time.sleep(0.25)
            
            # Step 4: Look for confirm_anvil and click it
            print("   ✅ Step 4: Looking for confirmation button...")
            confirm_pos = self.find_template(self.confirm_anvil_path, confidence=0.85)
            if confirm_pos:
                self.click_(confirm_pos[0], confirm_pos[1], button='left')
                print(f"   ✅ Clicked confirm at: {confirm_pos}")
                time.sleep(0.5)
            else:
                print("   ❌ Confirm button not found!")
                return "error"
            
            # Step 5: Press Enter to start upgrade
            print("   🚀 Step 5: Pressing Enter to start upgrade...")
            pydirectinput.press('enter')
            
            # Step 6: Wait for upgrade result (5 seconds)
            print("   ⏳ Step 6: Waiting 5 seconds for upgrade result...")
            time.sleep(5.0)
            
            # Step 7: Check item level after upgrade
            print("   🔍 Step 7: Reading item level after upgrade...")
            final_level = self.read_item_level_from_slot(item_slot_to_upgrade, anvil_rects)
            
            if final_level is None:
                print("   💥 UPGRADE FAILED - ITEM BURNED! (Cannot read level = item destroyed)")
                return "failed"
            
            if final_level == initial_level + 1:
                print(f"   🎉 UPGRADE SUCCESSFUL! (+{initial_level} → +{final_level})")
                return "success"
            
            if final_level == initial_level:
                print(f"   🚫 UPGRADE LIMIT REACHED - Level unchanged (+{initial_level})")
                print("   🎮 Initiating game quit sequence...")
                return "cannot_perform"
            
            # Unexpected case
            print(f"   ❓ Unexpected result: +{initial_level} → +{final_level}")
            return "error"
            
        except Exception as e:
            print(f"   ❌ Error during upgrade process: {e}")
            return "error"

    def organize_inventory_after_restock(self, model, inventory_rects, locked_slots):
        """
        Organize inventory after restock by moving upgrade scroll to first slot.
        
        Steps:
        1. Press ESC
        2. Open inventory
        3. Click coordinates to organize
        4. Use ML to find upgrade scroll
        5. Drag upgrade scroll to first slot
        
        Parameters:
        - model: ML model for item prediction
        - inventory_rects: Rectangle coordinates for inventory grid
        - locked_slots: Set of locked slot indices
        """
        print("🎒 ORGANIZING INVENTORY AFTER RESTOCK...")
        
        try:
            # Step 1: Press ESC
            print("   ⌨️ Step 1: Pressing ESC...")
            pydirectinput.press('escape')
            time.sleep(0.5)
            
            # Step 2: Open inventory
            print("   🎒 Step 2: Opening inventory...")
            inventory_opened = self.open_inventory()
            if not inventory_opened:
                print("   ❌ Failed to open inventory")
                return False
            
            time.sleep(1)
            
            # Step 3: Click organizing coordinates
            print("   🖱️ Step 3: Clicking organize coordinates...")
            self.click_(823, 407, button='left')
            time.sleep(0.5)
            self.click_(770, 404, button='left')
            time.sleep(0.5)
            
            
            
            # Step 7: Close inventory
            print("   🎒 Step 6: Closing inventory...")
            self.close_inventory()
            time.sleep(1)
            
            print("✅ INVENTORY ORGANIZATION COMPLETED!")
            return True
            
        except Exception as e:
            print(f"   ❌ Error during inventory organization: {e}")
            return False

    def preprocess_rectangles(self, image, rectangles, target_size=(21, 21)):
        """Batch preprocess rectangle regions from the image."""
        preprocessed_images = []

        for rect_coords in rectangles:
            x1, y1 = rect_coords[0]
            x2, y2 = rect_coords[1]

            roi = image[y1:y2, x1:x2]
            roi_resized = cv2.resize(roi, target_size)
            
            roi_normalized = roi_resized / 255.0
            roi_normalized = np.expand_dims(roi_normalized, axis=-1)
            preprocessed_images.append(roi_normalized)

        return np.array(preprocessed_images)

    def draw_rectangle_grid(self, screenshot, top_left, bottom_right, rows, cols, spacing, color=(0, 0, 255), thickness=2):
        """Draw rectangle grid and return coordinates of all rectangles."""
        rect_width = bottom_right[0] - top_left[0]
        rect_height = bottom_right[1] - top_left[1]
        rectangles = []

        for row in range(rows):
            for col in range(cols):
                current_top_left = (
                    top_left[0] + col * (rect_width + spacing),
                    top_left[1] + row * (rect_height + spacing)
                )
                current_bottom_right = (
                    current_top_left[0] + rect_width,
                    current_top_left[1] + rect_height
                )
                rectangles.append((current_top_left, current_bottom_right))
        
        return rectangles

    def display_predictions(self, screenshot, rectangles, predictions, locked_slots):
        """Overlay predictions on the screenshot within the rectangles."""
        for idx, (rect, pred) in enumerate(zip(rectangles, predictions)):
            center_x = rect[0][0] + (rect[1][0] - rect[0][0]) // 2
            center_y = rect[0][1] + (rect[1][1] - rect[0][1]) // 2
            
            # Change text color for locked slots
            text_color = (0, 0, 255) if idx in locked_slots else (255, 255, 255)  # BGR formatında
            
            text = str(pred)
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            text_x = center_x - text_width // 2
            text_y = center_y + text_height // 2
            
            cv2.putText(screenshot, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(screenshot, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

    def predict_and_draw_rectangles(self, screenshot, model, rectangles, locked_slots, rect_name="Unknown", rect_color=(0, 255, 0)):
        """
        Belirli bir rectangle grubu için prediction yapar ve çizer
        
        Parameters:
        - screenshot: Ekran görüntüsü
        - model: ML modeli
        - rectangles: Rectangle koordinatları listesi
        - locked_slots: Kilitli slot indeksleri
        - rect_name: Rectangle grubunun adı (debug için)
        - rect_color: Rectangle rengi (BGR formatında)
        
        Returns:
        - predictions: Prediction sonuçları
        """
        try:
            # Prediction yap
            processed_images = self.preprocess_rectangles(screenshot, rectangles)
            predictions = model.predict(processed_images, verbose=0)
            predictions = np.argmax(predictions, axis=1)
            
            # Rectangle'ları çiz
            for idx, rect in enumerate(rectangles):
                # Kilitli slotlar için kırmızı, diğerleri için verilen renk
                color = (0, 0, 255) if idx in locked_slots else rect_color  # BGR formatında
                cv2.rectangle(screenshot, rect[0], rect[1], color, 2)
            
            # Prediction'ları göster
            self.display_predictions(screenshot, rectangles, predictions, locked_slots)
            
            # Debug bilgisi - only print once during scanning
            if rect_name == "Inventory":
                pass  # Don't print inventory predictions repeatedly
            else:
                print(f"{rect_name} predictions: {predictions}")
            
            return predictions
            
        except Exception as e:
            print(f"Error in {rect_name} prediction: {e}")
            return None

    def read_location_with_ocr(self, screenshot, rectangles):
        if not OCR_AVAILABLE:
            return "❌ OCR kullanılamıyor"
            
        for rect in rectangles:
            x1, y1 = rect[0]
            x2, y2 = rect[1]
            roi = screenshot[y1:y2, x1:x2]

            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            enlarged = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            contrast = cv2.convertScaleAbs(enlarged, alpha=2.0, beta=50)
            _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789,'
            text = pytesseract.image_to_string(thresh, config=config)

            cleaned = text.strip().replace('\n', '').replace(' ', '')
            numbers = re.findall(r'\d+', cleaned)

            if numbers:
                combined = ''.join(numbers)
                if len(combined) == 6 or len(combined) == 7:
                    x = combined[:3]
                    y = combined[-3:]
                    return f"{x},{y}"
                else:
                    return ','.join(numbers)

        return "❌ Koordinat bulunamadı"

    def navigate_rectangles(self, rectangles, click=False, button='left', wait_time=0.5, ml_predictions=None):
        """
        Rectangle setinde mouse'u sırayla gezdiren fonksiyon
        Tüm rectangle'ları tarayarak her birinde template arama yapar
        
        Parameters:
        - rectangles: Rectangle koordinatları listesi [(top_left, bottom_right), ...]
        - click: Bool - tıklama yapılıp yapılmayacağı (default: False)
        - button: str - hangi buton ile tıklanacağı ('left' veya 'right', default: 'left')
        - wait_time: float - her rectangle arasındaki bekleme süresi (saniye, default: 0.5)
        - ml_predictions: list - ML prediction results for each slot (6 = empty, skip template check)
        
        Returns:
        - dictionary: {rectangle_index: found_value} for all rectangles
        """
        # Template pathları ve değerleri
        template_paths = {
            r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\plus_one.png": 1,
            r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\plus_two.png": 2,
            r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\plus_three.png": 3,
            r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\plus_four.png": 4,
            r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\plus_five.png": 5,
            r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\plus_six.png": 6,
            r"C:\Users\ismt\Desktop\python-ann\knightonlinefarmagentproject\ko_upgrade\game_images\plus_seven.png": 7
        }
        
        # Dictionary to store results
        rectangle_values = {}
        
        print(f"🔄 {len(rectangles)} rectangle üzerinde gezinme başlatılıyor...")
        print(f"   Tıklama: {'Evet' if click else 'Hayır'}")
        if click:
            print(f"   Buton: {button}")
        print(f"   Bekleme süresi: {wait_time}s")
        if ml_predictions is not None and len(ml_predictions) > 0:
            empty_slots = sum(1 for pred in ml_predictions if pred == 6)
            print(f"   🚀 OPTIMIZATION: ML predictions available - {empty_slots} empty slots will be skipped")
        print(f"   Template search: {'Optimized (skip empty slots)' if ml_predictions is not None else 'all rectangles'}")
        
        # Tüm rectangle'ları tarayalım (index 0'dan başlayarak)
        for idx in range(len(rectangles)):
            rect = rectangles[idx]
            try:
                # Check if we can skip this slot completely due to ML prediction
                if ml_predictions is not None and idx < len(ml_predictions) and ml_predictions[idx] == 6:
                    print(f"\n📍 Rectangle {idx+1}/{len(rectangles)}: SKIPPED")
                    print(f"   🚀 OPTIMIZATION: ML prediction shows empty slot (6) - Completely skipping slot")
                    rectangle_values[idx] = 0
                    continue  # Skip all mouse movement, template checking, and waiting
                
                # Rectangle'ın merkez koordinatlarını hesapla
                x1, y1 = rect[0]  # top_left
                x2, y2 = rect[1]  # bottom_right
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                print(f"\n📍 Rectangle {idx+1}/{len(rectangles)}: ({center_x}, {center_y})")
                
                if click:
                    # Tıklama yap
                    self.click_(center_x, center_y, button=button)
                    print(f"   ✅ {button.upper()} click yapıldı")
                else:
                    # Sadece mouse'u hareket ettir
                    self.move_(center_x, center_y)
                    print(f"   🖱️ Mouse pozisyonu: ({center_x}, {center_y})")
                
                # Kısa bir bekleme (screenshot için)
                time.sleep(0.2)
                
                # Template arama - Only for non-empty slots
                found_value = 0
                print(f"   🔍 Template aranıyor...")
                
                for template_path, value in template_paths.items():
                    try:
                        result = self.find_template(template_path)
                        if result is not None:
                            found_value = value
                            print(f"   ✅ Template bulundu: +{value}")
                            break
                    except Exception as e:
                        print(f"   ⚠️  Template arama hatası ({value}): {e}")
                        continue
                
                if found_value == 0:
                    print(f"   ❌ Hiçbir template bulunamadı - Değer: 0")
                
                # Store the value in dictionary
                rectangle_values[idx] = found_value
                print(f"   📊 Rectangle {idx+1} Değeri: {found_value}")
                
                # Bekleme (son rectangle'da değil ise)
                if idx < len(rectangles) - 1:
                    print(f"   ⏳ {wait_time}s bekleniyor...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                print(f"❌ Rectangle {idx+1} işlenirken hata: {e}")
                rectangle_values[idx] = 0
                print(f"   📊 Rectangle {idx+1} Değeri: 0")
                continue
        
        print("\n✅ Rectangle gezinme ve template arama tamamlandı!")
        print(f"📋 Sonuçlar: {rectangle_values}")
        return rectangle_values

    def open_npc(self, npc="anvil"):
        opened = False
        self.press_("b", hold_duration=0.5)
        
        # NPC path'lerini belirle
        npc_paths = {
            "anvil": self.upgrade_item_anvil_path,
            "armor": self.trade_armor_path,
            "bank": self.use_storage_bank_path
        }
        
        if npc not in npc_paths:
            print(f"Geçersiz NPC: {npc}")
            return False
        
        selected_path = npc_paths[npc]
        
        # Ekranın ortası (1024x768 için)
        center_x = 512
        center_y = 285 if npc == "anvil" else 390
        
        for i in range(20):
            if i < 10:
                # İlk 5 adım sola (30'ar birim)
                target_x = center_x - (i + 1) * 10
            else:
                # Son 5 adım sağa (30'ar birim)
                target_x = center_x + (i - 4) * 10
            
            target_y = center_y
            
            # Hesaplanan konuma sağ tıklama
            self.click_(target_x, target_y, button='right', clicks=2)
            time.sleep(0.5)
            
            # Seçilen NPC path'ini ara
            npc_pos = self.find_template(selected_path, confidence=0.85)
            
            if npc_pos:
                # Path bulunduğunda sol tıkla
                self.click_(npc_pos[0], npc_pos[1], button='left')
                opened = True
                break
                
            time.sleep(0.5)
        
        return opened
    
    def open_inventory(self):
        opened = False
        for i in range(5):
            loc = self.find_template(self.inventory_check_path, confidence=0.9)
            print(loc)
            if loc is not None:
                opened = True
                return opened
            self.click_(811, 778)
            time.sleep(1)
            
            
    def close_inventory(self):
        closed = False
        for i in range(5):
            loc = self.find_template(self.inventory_check_path, confidence=0.9)
            print(f"   🔍 Inventory check attempt {i+1}/5: {loc}")
            if loc is None:
                closed = True
                print("   ✅ Inventory successfully closed")
                return closed
            print(f"   🖱️ Clicking close button (attempt {i+1}/5)")
            self.click_(811, 778)
            time.sleep(1)
        
        print("   ❌ Failed to close inventory after 5 attempts")
        return closed

    def town_(self):
        max_attempts = 3
        target_x, target_y = 814, 540
        tolerance = 10
        
        for attempt in range(max_attempts):
            print(f"Town attempt {attempt + 1}/{max_attempts}")
            
            # Execute town sequence
            self.click_(903, 334, clicks=2)
            time.sleep(1)
            pydirectinput.press('h')
            time.sleep(1)
            self.click_(893, 365, clicks=2)
            pydirectinput.press('h')
            time.sleep(2)  # Wait for teleport to complete
            
            # Verify location with OCR
            if OCR_AVAILABLE:
                screenshot = self.capture_game_window()
                # Define OCR region for coordinates (using same coordinates as process_inventory_test)
                loc_top_left = (44, 105)
                loc_bottom_right = (163, 122)
                ocr_rectangles = self.draw_rectangle_grid(None, loc_top_left, loc_bottom_right, rows=1, cols=1, spacing=0)
                
                location_text = self.read_location_with_ocr(screenshot, ocr_rectangles)
                print(f"Current location: {location_text}")
                
                # Parse coordinates
                if ',' in location_text:
                    try:
                        coords = location_text.split(',')
                        current_x = int(coords[0])
                        current_y = int(coords[1])
                        
                        # Check if location is near town
                        distance_x = abs(current_x - target_x)
                        distance_y = abs(current_y - target_y)
                        
                        if distance_x <= tolerance and distance_y <= tolerance:
                            print(f"✅ Successfully reached town at ({current_x}, {current_y})")
                            return True
                        else:
                            print(f"❌ Not at town. Current: ({current_x}, {current_y}), Target: ({target_x}, {target_y})")
                    except (ValueError, IndexError):
                        print(f"❌ Could not parse coordinates: {location_text}")
                else:
                    print(f"❌ Invalid location format: {location_text}")
            else:
                print("⚠️ OCR not available, cannot verify location")
                return True  # Assume success if OCR is not available
            
            if attempt < max_attempts - 1:
                print(f"Retrying town function in 2 seconds...")
                time.sleep(2)
        
        print(f"❌ Failed to reach town after {max_attempts} attempts")
        return False

    def startup_inventory_scan(self, model, inventory_rects, locked_slots):
        """
        Complete startup sequence: Go to town → Open inventory → Scan items
        
        Parameters:
        - model: ML model for item prediction
        - inventory_rects: Rectangle coordinates for inventory grid
        - locked_slots: Set of locked slot indices
        
        Returns:
        - dictionary: Complete inventory data with ML predictions and template values
        """
        print("🚀 Starting game startup sequence...")
        
        # Step 1: Go to town
        print("🏠 Step 1: Going to town...")
        town_success = self.town_()
        if not town_success:
            print("❌ Failed to reach town, aborting startup sequence")
            return None
        
        time.sleep(2)
        
        # Step 2: Open inventory
        print("🎒 Step 2: Opening inventory...")
        inventory_opened = self.open_inventory()
        if not inventory_opened:
            print("❌ Failed to open inventory, aborting startup sequence")
            return None
        
        time.sleep(1)
        
        # Step 3: Capture screenshot and get ML predictions
        print("🧠 Step 3: Getting ML predictions...")
        screenshot = self.capture_game_window()
        
        predictions = self.predict_and_draw_rectangles(
            screenshot, model, inventory_rects, locked_slots, 
            rect_name="Inventory", rect_color=(0, 255, 0)
        )
        
        if predictions is None:
            print("❌ Failed to get ML predictions")
            return None
        
        # Display ML predictions grid
        print("\n🧠 ML PREDICTIONS GRID (Item Types):")
        print("┌" + "─" * 29 + "┐")
        for row in range(4):
            row_display = "│"
            for col in range(7):
                slot_idx = row * 7 + col
                if slot_idx < len(predictions):
                    pred_val = predictions[slot_idx]
                    row_display += f" {pred_val:2d} │"
                else:
                    row_display += "    │"
            print(row_display)
            if row < 3:
                print("├" + "─" * 29 + "┤")
        print("└" + "─" * 29 + "┘")
        print("Legend: 0=Scroll 1=Pauldron 2=Pant 3=Boot 4=Hat 5=Glove 6=Empty")
        
        # Step 4: Get template values using navigate_rectangles (optimized with ML predictions)
        print("\n🎯 Step 4: Scanning for upgrade levels...")
        template_values = self.navigate_rectangles(inventory_rects, click=False, wait_time=0.3, ml_predictions=predictions)
        
        # Display template values grid
        print("\n🎯 TEMPLATE VALUES GRID (Upgrade Levels):")
        print("┌" + "─" * 29 + "┐")
        for row in range(4):
            row_display = "│"
            for col in range(7):
                slot_idx = row * 7 + col
                template_val = template_values.get(slot_idx, 0)
                row_display += f" {template_val:2d} │"
            print(row_display)
            if row < 3:
                print("├" + "─" * 29 + "┤")
        print("└" + "─" * 29 + "┘")
        print("Legend: 0=No upgrade 1=+1 2=+2 3=+3 4=+4 5=+5 6=+6 7=+7")
        
        # Step 5: Combine data
        print("\n📋 Step 5: Processing inventory data...")
        inventory_data = {}
        
        # Item type mapping
        item_type_map = {
            0: "Upgrade Scroll",
            1: "Pauldron", 
            2: "Pant",
            3: "Boot",
            4: "Hat",
            5: "Glove",
            6: "Empty"
        }
        
        for slot_idx, prediction in enumerate(predictions):
            template_value = template_values.get(slot_idx, 0)
            
            # Handle ML prediction values (item types)
            if slot_idx in locked_slots:
                item_type = "LOCKED"
                if prediction == 0:
                    item_status = "Locked slot - Upgrade Scroll"
                else:
                    item_status = f"Locked slot - {item_type_map.get(prediction, 'Unknown')}"
            elif prediction == 6:
                item_type = "Empty"
                item_status = "Empty slot"
            elif prediction in item_type_map:
                item_type = item_type_map[prediction]
                item_status = f"{item_type}"
            else:
                item_type = f"UNKNOWN({prediction})"
                item_status = f"Unknown item type: {prediction}"
            
            # Handle template values (upgrade levels)
            if template_value > 0:
                upgrade_level = f"+{template_value}"
                template_status = f"Upgrade level: {upgrade_level}"
                # Combine item type with upgrade level
                if item_type not in ["Empty", "LOCKED"] and prediction != 6:
                    final_item_status = f"{item_type} {upgrade_level}"
                else:
                    final_item_status = item_status
            else:
                upgrade_level = "No upgrade"
                template_status = "No upgrade detected"
                final_item_status = item_status
            
            inventory_data[slot_idx] = {
                'prediction': prediction,
                'item_type': item_type,
                'status': final_item_status,
                'template_value': template_value,
                'upgrade_level': upgrade_level,
                'template_status': template_status
            }
        
        # Display final summary
        print("\n" + "="*80)
        print("📦 INVENTORY STARTUP SCAN RESULTS:")
        print("="*80)
        print(f"{'Slot':<4} {'ML':<2} {'Item Type':<12} {'Tmpl':<4} {'Upgrade':<8} {'Final Status':<25}")
        print("-"*80)
        for slot_idx, data in inventory_data.items():
            print(f"{slot_idx:<4} {data['prediction']:<2} {data['item_type']:<12} {data['template_value']:<4} {data['upgrade_level']:<8} {data['status']:<25}")
        print("="*80)
        
        # Step 6: Close inventory
        print("🎒 Step 6: Closing inventory...")
        self.close_inventory()
        time.sleep(1)
        
        print("✅ Startup inventory scan completed successfully!")
        return inventory_data

    def find_correct_npc_at_armor(self):
        """
        Find the correct NPC at armor shop by checking all gargamet and hesta variations
        If any gargamet is found, rotate right and press 'b' to find hesta
        Only one attempt - if fails, return False to trigger restart from town
        
        Returns:
        - True: If any hesta NPC is found/reached
        - False: If NPC finding failed, should restart from town
        """
        print("🔍 Searching for correct NPC at armor shop (checking all variations)...")
        
        # Define all NPC template paths
        gargamet_paths = [
            (self.gargamet_path, "gargamet"),
            (self.gargamet_2_path, "gargamet_2"), 
            (self.gargamet_3_path, "gargamet_3"),
            (self.gargamet_4_path, "gargamet_4"),
            (self.gargamet_5_path, "gargamet_5"),
            (self.gargamet_6_path, "gargamet_6")

        ]
        
        hesta_paths = [
            (self.hesta_path, "hesta"),
            (self.hesta_2_path, "hesta_2"),
            (self.hesta_3_path, "hesta_3"),
            (self.hesta_4_path, "hesta_4"),
            (self.hesta_5_path, "hesta_5"),
            (self.hesta_6_path, "hesta_6"),
            (self.hesta_7_path, "hesta_7"),
            (self.hesta_8_path, "hesta_8")

        ]
        
        # Check for any gargamet variation first
        self.press_("b")
        time.sleep(1)
        
        gargamet_found = False
        gargamet_type = None
        
        for gargamet_path, gargamet_name in gargamet_paths:
            if self.find_template(gargamet_path, confidence=0.75):
                print(f"👨 {gargamet_name} detected!")
                gargamet_found = True
                gargamet_type = gargamet_name
                break
        
        if gargamet_found:
            # Move to hesta position: press D (right) and W (forward)
            print(f"🔄 Found {gargamet_type}, moving to Hesta position...")
            print("🎮 Pressing D (right) to rotate toward Hesta...")
            self.press_("d", hold_duration=0.5)
            time.sleep(0.5)
            
            print("🎮 Pressing W (forward) to move toward Hesta...")
            self.press_("w", hold_duration=1.0)
            time.sleep(0.5)
            
            self.press_("b", hold_duration=0.5)
            time.sleep(0.5)
            
            print("🎮 Pressing W (forward) to move toward Hesta...")
            self.press_("w", hold_duration=1.0)
            time.sleep(0.5)
            
            # Press 'b' to interact/refresh view
            print("🔄 Pressing 'b' to refresh view...")
            self.press_("b", hold_duration=0.5)
            time.sleep(1.0)
            
            # Check for any hesta variation after rotation
            for hesta_path, hesta_name in hesta_paths:
                if self.find_template(hesta_path, confidence=0.75):
                    print(f"✅ {hesta_name} found after rotation!")
                    return True
            
            print("❌ No hesta variation found after rotation - will restart from town")
            return False
        else:
            # Check for any hesta variation directly
            for hesta_path, hesta_name in hesta_paths:
                if self.find_template(hesta_path, confidence=0.75):
                    print(f"✅ {hesta_name} found directly!")
                    
                    # Hesta found directly: Press B + W (1 sec) + approach NPC
                    print("🎮 Hesta found directly - executing approach sequence...")
                    print("🎮 Step 1: Pressing B to interact...")
                    self.press_("b", hold_duration=0.5)
                    time.sleep(0.5)
                    
                    print("🎮 Step 2: Pressing W (forward) for 1 second to approach...")
                    self.press_("w", hold_duration=0.5)
                    time.sleep(0.5)
                    
                    self.press_("b", hold_duration=0.5)
                    time.sleep(0.5)
                    
                    print("🎮 Step 2: Pressing W (forward) for 1 second to approach...")
                    self.press_("w", hold_duration=0.5)
                    time.sleep(0.5)
                    
                    print("🎮 Step 3: Final B press to get near NPC...")
                    self.press_("b", hold_duration=0.5)
                    time.sleep(1.0)
                    
                    print("✅ Hesta approach sequence completed!")
                    return True
            
            print("❌ No gargamet or hesta variations found - will restart from town")
            return False

    def interact_with_anvil(self):
        """
        Try to interact with the anvil with proper approach sequence
        - First press 'b' and check for helard or anvil
        - If helard found: press 'a' for 0.5s, then 'w' for 1.5s, check again
        - If anvil found: press 'b' for 0.5s, 'w' for 0.5s, 'b' again, 'w' again, then open anvil NPC
        
        Returns:
        - True: If anvil opened successfully
        - False: If failed to find/open anvil
        """
        print("🔧 Attempting to interact with anvil with approach sequence...")
        time.sleep(1)
        
        # Define search paths
        magic_anvil_paths = {
            "magic_anvil": self.magic_anvil_path,
            "magic_anvil_2": self.magic_anvil_2_path,
            "magic_anvil_3": self.magic_anvil_3_path,
            "magic_anvil_4": self.magic_anvil_4_path,
            "magic_anvil_5": self.magic_anvil_5_path
        }
        
        helard_paths = {
            "helard": self.helard_path,
            "helard_2": self.helard_2_path
        }
        
        def check_for_npcs():
            """Check for both helard and anvil NPCs, return type and variant"""
            # Check for helard first
            for variant_name, path in helard_paths.items():
                try:
                    result = self.find_template(path, confidence=0.65)
                    if result:
                        return "helard", variant_name
                except Exception as e:
                    print(f"   ⚠️ Error checking {variant_name}: {e}")
                    continue
            
            # Check for anvil
            for variant_name, path in magic_anvil_paths.items():
                try:
                    result = self.find_template(path, confidence=0.65)
                    if result:
                        return "anvil", variant_name
                except Exception as e:
                    print(f"   ⚠️ Error checking {variant_name}: {e}")
                    continue
            
            return None, None
        
        # Step 1: Initial search
        print("   🎮 Pressing 'b' to search for NPCs...")
        self.press_("b")
        time.sleep(1.0)
        
        print("   🔍 Looking for helard and anvil NPCs...")
        npc_type, found_variant = check_for_npcs()
        
        if npc_type == "helard":
            print(f"   👤 Helard found: {found_variant}")
            print("   🎮 Moving to approach anvil: pressing 'a' for 0.5s...")
            self.press_("a", hold_duration=1)
            time.sleep(0.1)
            print("   🎮 Moving forward: pressing 'w' for 1.5s...")
            self.press_("w", hold_duration=1)
            time.sleep(0.5)
            
            # Check again after movement
            print("   🎮 Pressing 'b' to search again after movement...")
            self.press_("b")
            time.sleep(1.0)
            
            print("   🔍 Looking for anvil after movement...")
            npc_type, found_variant = check_for_npcs()
        
        if npc_type == "anvil":
            print(f"   🔨 Anvil found: {found_variant}")
            print("   🎮 Executing anvil approach sequence...")
            
            # Anvil approach sequence: b(0.5s) -> w(0.5s) -> b -> w
            print("   🎮 Step 1: Pressing 'b' for 0.5s...")
            self.press_("b", hold_duration=0.5)
            time.sleep(0.1)
            
            print("   🎮 Step 2: Pressing 'w' for 0.5s...")
            self.press_("w", hold_duration=1)
            time.sleep(0.1)
            
            print("   🎮 Step 3: Pressing 'b' again...")
            self.press_("b")
            time.sleep(0.5)
            
            print("   🎮 Step 4: Pressing 'w' again...")
            self.press_("w", hold_duration=1)
            time.sleep(0.5)
            
            # Try to open anvil NPC
            print("   🔧 Opening anvil NPC...")
            anvil_opened = self.open_npc("anvil")
            
            if anvil_opened:
                print("   ✅ Anvil NPC opened successfully!")
                return True
            else:
                print("   ❌ Failed to open anvil NPC")
                return False
        else:
            print("   ❌ No anvil found after approach sequence")
            return False

    def interact_with_bank(self):
        """
        Try to interact with the bank with proper approach sequence
        - Press 'b' and check for bank variants (bank, bank_2, bank_3, bank_4, bank_5)
        - If bank found: press 'b' for 0.5s, 'w' for 0.5s, repeat until positioned, then open bank NPC
        
        Returns:
        - True: If bank opened successfully
        - False: If failed to find/open bank
        """
        print("🏦 Attempting to interact with bank with approach sequence...")
        time.sleep(1)
        
        # Define search paths for bank variants
        bank_paths = {
            "bank": self.bank_path,
            "bank_2": self.bank_2_path,
            "bank_3": self.bank_3_path,
            "bank_4": self.bank_4_path,
            "bank_5": self.bank_5_path
        }
        
        def check_for_bank():
            """Check for bank NPCs, return type and variant"""
            for variant_name, path in bank_paths.items():
                try:
                    result = self.find_template(path, confidence=0.65)
                    if result:
                        return "bank", variant_name
                except Exception as e:
                    print(f"   ⚠️ Error checking {variant_name}: {e}")
                    continue
            
            return None, None
        
        # Step 1: Initial search
        print("   🎮 Pressing 'b' to search for bank NPCs...")
        self.press_("b")
        time.sleep(1.0)
        
        print("   🔍 Looking for bank NPCs...")
        npc_type, found_variant = check_for_bank()
        
        if npc_type == "bank":
            print(f"   🏦 Bank found: {found_variant}")
            print("   🎮 Executing bank approach sequence...")
            
            # Bank approach sequence: b(0.5s) -> w(0.5s) -> repeat until positioned
            max_approach_attempts = 3
            
            for attempt in range(max_approach_attempts):
                print(f"   🎮 Approach attempt {attempt + 1}: Pressing 'b' for 0.5s...")
                self.press_("b", hold_duration=0.5)
                time.sleep(0.1)
                
                print(f"   🎮 Approach attempt {attempt + 1}: Pressing 'w' for 0.5s...")
                self.press_("w", hold_duration=0.5)
                time.sleep(0.5)
                
                # Check if we're still near the bank
                npc_type, found_variant = check_for_bank()
                if npc_type == "bank":
                    print(f"   ✅ Bank still detected: {found_variant}")
                else:
                    print(f"   ⚠️ Bank lost after approach attempt {attempt + 1}")
            
            # Try to open bank NPC
            print("   🏛️ Opening bank NPC...")
            bank_opened = self.open_npc("bank")
            
            if bank_opened:
                print("   ✅ Bank NPC opened successfully!")
                return True
            else:
                print("   ❌ Failed to open bank NPC")
                return False
        else:
            print("   ❌ No bank found - will try fallback navigation")
            return False

    def execute_anvil_interaction_with_retry(self, nav_system, max_attempts=3):
        """
        Try to interact with anvil, with retry logic that goes back to town if anvil not found
        
        Parameters:
        - nav_system: Navigation system for going back to anvil
        - max_attempts: Maximum number of attempts to find and interact with anvil
        
        Returns:
        - True: If anvil interaction was successful
        - False: If all attempts failed
        """
        for attempt in range(max_attempts):
            print(f"\n🔧 ANVIL INTERACTION ATTEMPT {attempt + 1}/{max_attempts}")
            
            # Try to interact with anvil
            anvil_success = self.interact_with_anvil()
            
            if anvil_success:
                print("✅ Anvil interaction successful!")
                return True
            else:
                print("❌ Anvil interaction failed")
                
                if attempt < max_attempts - 1:  # Don't go to town on last attempt
                    print("🔄 Going back to town and retrying...")
                    
                    # Go back to town
                    town_success = self.town_()
                    if town_success:
                        print("✅ Returned to town successfully")
                        time.sleep(2)
                        
                        # Try to go back to anvil using town_to_anvil
                        print("🗺️ Attempting to reach anvil again from town...")
                        anvil_nav_success = nav_system.navigate_with_agent("town_to_anvil", max_steps=200, distance_threshold=4)
                        
                        if anvil_nav_success:
                            print("✅ Successfully navigated back to anvil")
                        else:
                            print("❌ Failed to navigate back to anvil")
                            return False
                    else:
                        print("❌ Failed to return to town")
                        return False
        
        print(f"❌ Failed to interact with anvil after {max_attempts} attempts")
        return False
        
    def click_middle_of_rectangle(self, rectangle, button='left'):
        """
        Click on the middle of a given rectangle
        
        Parameters:
        - rectangle: tuple ((x1, y1), (x2, y2)) - rectangle coordinates
        - button: str - mouse button to use ('left' or 'right')
        
        Returns:
        - tuple: (center_x, center_y) - coordinates of the click
        """
        x1, y1 = rectangle[0]  # top_left
        x2, y2 = rectangle[1]  # bottom_right
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        self.click_(center_x, center_y, button=button)
        return (center_x, center_y)

    def count_inventory_items(self, inventory_data, locked_slots):
        """
        Count the number of each item type in inventory (excluding locked slots and empty slots)
        
        Parameters:
        - inventory_data: Dictionary with inventory predictions
        - locked_slots: Set of locked slot indices
        
        Returns:
        - dict: Item counts {item_type_id: count}
        """
        item_counts = {
            1: 0,  # Pauldron
            2: 0,  # Pant  
            3: 0,  # Boot
            4: 0,  # Hat
            5: 0   # Glove
        }
        
        empty_slots = 0
        
        for slot_idx, data in inventory_data.items():
            if slot_idx in locked_slots:
                continue  # Skip locked slots
                
            prediction = data['prediction']
            if prediction == 6:  # Empty slot
                empty_slots += 1
            elif prediction in item_counts:
                item_counts[prediction] += 1
        
        print(f"\n📊 INVENTORY ITEM COUNTS:")
        print(f"   Pauldron (1): {item_counts[1]}")
        print(f"   Pant (2): {item_counts[2]}")
        print(f"   Boot (3): {item_counts[3]}")
        print(f"   Hat (4): {item_counts[4]}")
        print(f"   Glove (5): {item_counts[5]}")
        print(f"   Empty slots: {empty_slots}")
        
        return item_counts, empty_slots

    def calculate_items_to_buy(self, item_counts, empty_slots):
        """
        Calculate which items to buy to fill ALL empty slots
        Priority: buy the item type with the lowest count, cycling through all types
        
        Parameters:
        - item_counts: Dictionary with current item counts
        - empty_slots: Number of empty inventory slots
        
        Returns:
        - list: List of sundires slots to buy from, in order
        """
        # Item type mapping for sundires rectangles (indices 0-4)
        # Corrected mapping: 0=Pauldron, 1=Pant, 2=Hat, 3=Glove, 4=Boot
        sundires_item_map = {
            0: 1,  # Sundires slot 0 = Pauldron (item type 1)
            1: 2,  # Sundires slot 1 = Pant (item type 2)  
            2: 4,  # Sundires slot 2 = Hat (item type 4)
            3: 5,  # Sundires slot 3 = Glove (item type 5)
            4: 3   # Sundires slot 4 = Boot (item type 3)
        }
        
        # Reverse mapping: item_type -> sundires_slot
        item_to_sundires = {v: k for k, v in sundires_item_map.items()}
        
        if empty_slots <= 0:
            print("❌ No empty slots available for purchasing items")
            return []
        
        print(f"\n🛒 PURCHASE CALCULATION:")
        print(f"   Empty slots to fill: {empty_slots}")
        print(f"   Goal: Fill ALL empty slots")
        
        # Create a working copy of item counts for simulation
        working_counts = item_counts.copy()
        purchase_order = []
        
        print(f"   Starting item counts: {working_counts}")
        
        # Buy items to fill all empty slots
        for purchase_round in range(empty_slots):
            # Find item type with lowest count
            min_count = min(working_counts.values())
            lowest_items = [item_type for item_type, count in working_counts.items() if count == min_count]
            
            # If multiple items have same lowest count, pick the first one
            item_to_buy = lowest_items[0]
            sundires_slot = item_to_sundires[item_to_buy]
            
            purchase_order.append({
                'sundires_slot': sundires_slot,
                'item_type_id': item_to_buy,
                'current_count': working_counts[item_to_buy],
                'purchase_round': purchase_round + 1
            })
            
            # Update working counts (simulate the purchase)
            working_counts[item_to_buy] += 1
        
        print(f"\n🎯 COMPLETE PURCHASE PLAN ({len(purchase_order)} items):")
        item_names = {1: "Pauldron", 2: "Pant", 3: "Boot", 4: "Hat", 5: "Glove"}
        
        for i, item in enumerate(purchase_order):
            item_name = item_names.get(item['item_type_id'], f"Item{item['item_type_id']}")
            print(f"   {i+1:2d}. Sundires slot {item['sundires_slot']}: {item_name} (was {item['current_count']})")
        
        print(f"\n📊 FINAL COUNTS AFTER ALL PURCHASES:")
        for item_type, final_count in working_counts.items():
            item_name = item_names.get(item_type, f"Item{item_type}")
            original_count = item_counts[item_type]
            bought = final_count - original_count
            print(f"   {item_name}: {original_count} → {final_count} (+{bought})")
        
        return purchase_order

    def buy_items_from_sundires(self, sundires_rects, purchase_order):
        """
        Buy items from sundires rectangles based on purchase order
        Uses right-click to select items, then purchase_armor_path and confirm_path for confirmation
        
        Parameters:
        - sundires_rects: List of sundires rectangle coordinates
        - purchase_order: List of items to buy from calculate_items_to_buy
        
        Returns:
        - bool: True if all purchases were successful
        - list: List of purchased items for inventory update
        """
        if not purchase_order:
            print("❌ No items to purchase")
            return False, []
        
        print(f"\n🛒 STARTING ITEM PURCHASES:")
        print(f"   Total items to buy: {len(purchase_order)}")
        print(f"   Goal: Fill ALL empty inventory slots")
        
        success_count = 0
        purchased_items = []
        
        for i, item_info in enumerate(purchase_order):
            try:
                sundires_slot = item_info['sundires_slot']
                if sundires_slot >= len(sundires_rects):
                    print(f"❌ Invalid sundires slot {sundires_slot} (max: {len(sundires_rects)-1})")
                    continue
                
                rectangle = sundires_rects[sundires_slot]
                item_names = {1: "Pauldron", 2: "Pant", 3: "Boot", 4: "Hat", 5: "Glove"}
                item_name = item_names.get(item_info['item_type_id'], f"Item{item_info['item_type_id']}")
                
                print(f"\n🎯 Purchase {i+1}/{len(purchase_order)}: Buying {item_name} from sundires slot {sundires_slot}...")
                print(f"   Rectangle coordinates: {rectangle}")
                
                # Step 1: Right-click on the item in sundires rectangle
                clicked_coords = self.click_middle_of_rectangle(rectangle, button='right')
                print(f"   ✅ Right-clicked at: {clicked_coords}")
                time.sleep(0.5)
                
                # Step 2: Click on purchase_armor_path with left click
                print(f"   🛒 Looking for purchase armor button...")
                purchase_pos = self.find_template(self.purchase_armor_path, confidence=0.85)
                if purchase_pos:
                    self.click_(purchase_pos[0], purchase_pos[1], button='left')
                    print(f"   ✅ Clicked purchase armor at: {purchase_pos}")
                    time.sleep(1.0)
                    
                    # Step 3: Click on confirm_path
                    print(f"   ✅ Looking for confirmation button...")
                    confirm_pos = self.find_template(self.confirm_path, confidence=0.85)
                    if confirm_pos:
                        self.click_(confirm_pos[0], confirm_pos[1], button='left')
                        print(f"   ✅ Clicked confirm at: {confirm_pos}")
                        time.sleep(1.0)
                        
                        # Add to purchased items list (new items start at +1)
                        purchased_items.append({
                            'item_type_id': item_info['item_type_id'],
                            'item_name': item_name,
                            'upgrade_level': 1  # New items start at +1
                        })
                        success_count += 1
                        print(f"   ✅ Successfully purchased {item_name} +1")
                        
                    else:
                        print(f"   ❌ Confirm button not found for {item_name}")
                        continue
                else:
                    print(f"   ❌ Purchase armor button not found for {item_name}")
                    continue
                
            except Exception as e:
                print(f"❌ Error purchasing item {i+1}: {e}")
                continue
        
        print(f"\n📊 PURCHASE SUMMARY:")
        print(f"   Successful purchases: {success_count}/{len(purchase_order)}")
        print(f"   Purchase {'completed successfully!' if success_count == len(purchase_order) else 'partially completed'}")
        
        if purchased_items:
            print(f"\n🎁 PURCHASED ITEMS:")
            for item in purchased_items:
                print(f"   - {item['item_name']} +{item['upgrade_level']}")
        
        return success_count == len(purchase_order), purchased_items

    def update_inventory_with_purchases(self, inventory_data, purchased_items, locked_slots):
        """
        Update inventory data with newly purchased items
        Places new items in empty slots and assigns +1 upgrade level
        
        Parameters:
        - inventory_data: Current inventory data dictionary
        - purchased_items: List of purchased items from buy_items_from_sundires
        - locked_slots: Set of locked slot indices
        
        Returns:
        - dict: Updated inventory data
        """
        if not purchased_items:
            print("❌ No purchased items to add to inventory")
            return inventory_data
        
        print(f"\n📦 UPDATING INVENTORY WITH {len(purchased_items)} NEW ITEMS:")
        
        # Find empty slots (excluding locked slots)
        empty_slots = []
        for slot_idx, data in inventory_data.items():
            if slot_idx not in locked_slots and data['prediction'] == 6:  # 6 = Empty
                empty_slots.append(slot_idx)
        
        print(f"   Available empty slots: {empty_slots}")
        
        # Add purchased items to empty slots
        items_added = 0
        for i, item in enumerate(purchased_items):
            if i >= len(empty_slots):
                print(f"   ⚠️ Not enough empty slots for all purchased items")
                break
            
            slot_idx = empty_slots[i]
            item_type_id = item['item_type_id']
            item_name = item['item_name']
            upgrade_level = item['upgrade_level']
            
            # Update inventory data for this slot
            inventory_data[slot_idx] = {
                'prediction': item_type_id,
                'item_type': item_name,
                'status': f"{item_name} +{upgrade_level}",
                'template_value': upgrade_level,
                'upgrade_level': f"+{upgrade_level}",
                'template_status': f"Upgrade level: +{upgrade_level}"
            }
            
            print(f"   ✅ Added {item_name} +{upgrade_level} to slot {slot_idx}")
            items_added += 1
        
        print(f"\n📊 INVENTORY UPDATE SUMMARY:")
        print(f"   Items successfully added: {items_added}/{len(purchased_items)}")
        
        # Display detailed slot-by-slot inventory
        print(f"\n" + "="*80)
        print("📦 DETAILED UPDATED INVENTORY (SLOT BY SLOT):")
        print("="*80)
        print(f"{'Slot':<4} {'Item Type':<12} {'Upgrade':<8} {'Status':<25}")
        print("-"*80)
        
        item_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # Pauldron, Pant, Boot, Hat, Glove
        empty_count = 0
        locked_count = 0
        
        # Sort slots by number for ordered display
        sorted_slots = sorted(inventory_data.keys())
        
        for slot_idx in sorted_slots:
            data = inventory_data[slot_idx]
            
            if slot_idx in locked_slots:
                locked_count += 1
                print(f"{slot_idx:<4} {'LOCKED':<12} {'-':<8} {'Locked slot':<25}")
            elif data['prediction'] == 6:  # Empty slot
                empty_count += 1
                print(f"{slot_idx:<4} {'Empty':<12} {'-':<8} {'Empty slot':<25}")
            else:
                # Item slot
                item_type = data['item_type']
                upgrade_level = data['upgrade_level']
                template_value = data['template_value']
                
                if data['prediction'] in item_counts:
                    item_counts[data['prediction']] += 1
                
                # Show upgrade level more clearly
                if template_value > 0:
                    upgrade_display = f"+{template_value}"
                    status = f"{item_type} {upgrade_display}"
                else:
                    upgrade_display = "No upgrade"
                    status = item_type
                
                print(f"{slot_idx:<4} {item_type:<12} {upgrade_display:<8} {status:<25}")
        
        print("="*80)
        
        # Display summary counts
        print(f"\n📊 UPDATED INVENTORY SUMMARY:")
        print(f"   Pauldron: {item_counts[1]}")
        print(f"   Pant: {item_counts[2]}")
        print(f"   Boot: {item_counts[3]}")
        print(f"   Hat: {item_counts[4]}")
        print(f"   Glove: {item_counts[5]}")
        print(f"   Empty slots: {empty_count}")
        print(f"   Locked slots: {locked_count}")
        print(f"   Total slots: {len(inventory_data)}")
        
        return inventory_data

    def perform_single_upgrade(self, anvil_rects, locked_slots, item_slot_to_upgrade):
        """
        Perform a single item upgrade using scroll in slot 0 and target item
        
        Parameters:
        - anvil_rects: Anvil rectangle coordinates (positions change when anvil is open)
        - locked_slots: Set of locked slot indices 
        - item_slot_to_upgrade: Index of the item slot to upgrade
        
        Returns:
        - str: "success", "failed", or "error"
        """
        print(f"\n🔧 PERFORMING UPGRADE ON SLOT {item_slot_to_upgrade}")
        
        try:
            # Step 1: Right-click on upgrade scroll (slot 0 - locked slot)
            print("   📜 Step 1: Selecting upgrade scroll from slot 0...")
            scroll_rectangle = anvil_rects[0]  # Slot 0 has the upgrade scroll
            self.click_middle_of_rectangle(scroll_rectangle, button='right')
            time.sleep(0.25)
            
            # Step 2: Right-click on target item to upgrade
            print(f"   🎯 Step 2: Selecting item to upgrade from slot {item_slot_to_upgrade}...")
            item_rectangle = anvil_rects[item_slot_to_upgrade]
            self.click_middle_of_rectangle(item_rectangle, button='right')
            time.sleep(0.25)
            
            # Step 3: Look for confirm_anvil and click it
            print("   ✅ Step 3: Looking for confirmation button...")
            confirm_pos = self.find_template(self.confirm_anvil_path, confidence=0.85)
            if confirm_pos:
                self.click_(confirm_pos[0], confirm_pos[1], button='left')
                print(f"   ✅ Clicked confirm at: {confirm_pos}")
                time.sleep(0.5)
            else:
                print("   ❌ Confirm button not found!")
                return "error"
            
            # Step 4: Press Enter to start upgrade
            print("   🚀 Step 4: Pressing Enter to start upgrade...")
            pydirectinput.press('enter')
            
            # Step 5: Wait for upgrade result (4.5 seconds)
            print("   ⏳ Step 5: Waiting 5.25 seconds for upgrade result...")
            time.sleep(5.25)
            
            # Step 6: Check for upgrade result
            print("   🔍 Step 6: Checking upgrade result...")
            
            # Check for success first (check all five success templates)
            success_found = self.find_template(self.upgrade_succeed_path, confidence=0.85)
            if success_found:
                print("   🎉 UPGRADE SUCCESSFUL! (Found: upgrade_succeed.png)")
                return "success"
            
            success_found = self.find_template(self.upgrade_succeed_2_path, confidence=0.9)
            if success_found:
                print("   🎉 UPGRADE SUCCESSFUL! (Found: upgrade_succeed_2.png)")
                return "success"
            
            success_found = self.find_template(self.upgrade_succeed_3_path, confidence=0.85)
            if success_found:
                print("   🎉 UPGRADE SUCCESSFUL! (Found: upgrade_succeed_3.png)")
                return "success"
            
            success_found = self.find_template(self.upgrade_succeed_4_path, confidence=0.85)
            if success_found:
                print("   🎉 UPGRADE SUCCESSFUL! (Found: upgrade_succeed_4.png)")
                return "success"
            
            success_found = self.find_template(self.upgrade_succeed_5_path, confidence=0.85)
            if success_found:
                print("   🎉 UPGRADE SUCCESSFUL! (Found: upgrade_succeed_5.png)")
                return "success"
            
            # Check for failure (check all six failure templates)
            failed_found = self.find_template(self.upgrade_failed_path, confidence=0.85)
            if failed_found:
                print("   💥 UPGRADE FAILED - ITEM BURNED! (Found: upgrade_failed.png)")
                return "failed"
            
            failed_found = self.find_template(self.upgrade_failed_2_path, confidence=0.85)
            if failed_found:
                print("   💥 UPGRADE FAILED - ITEM BURNED! (Found: upgrade_failed_2.png)")
                return "failed"
            
            failed_found = self.find_template(self.upgrade_failed_3_path, confidence=0.85)
            if failed_found:
                print("   💥 UPGRADE FAILED - ITEM BURNED! (Found: upgrade_failed_3.png)")
                return "failed"
            
            failed_found = self.find_template(self.upgrade_failed_4_path, confidence=0.85)
            if failed_found:
                print("   💥 UPGRADE FAILED - ITEM BURNED! (Found: upgrade_failed_4.png)")
                return "failed"
            
            failed_found = self.find_template(self.upgrade_failed_5_path, confidence=0.85)
            if failed_found:
                print("   💥 UPGRADE FAILED - ITEM BURNED! (Found: upgrade_failed_5.png)")
                return "failed"
            
            failed_found = self.find_template(self.upgrade_failed_6_path, confidence=0.85)
            if failed_found:
                print("   💥 UPGRADE FAILED - ITEM BURNED! (Found: upgrade_failed_6.png)")
                return "failed"
            
            # Check for cannot perform (check all three cannot_perform templates)
            cannot_perform_found = self.find_template(self.upgrade_cannot_perform_path, confidence=0.8)
            if not cannot_perform_found:
                cannot_perform_found = self.find_template(self.upgrade_cannot_perform_2_path, confidence=0.8)
            if not cannot_perform_found:
                cannot_perform_found = self.find_template(self.upgrade_cannot_perform_3_path, confidence=0.8)
            
            if cannot_perform_found:
                print("   🚫 UPGRADE CANNOT BE PERFORMED - GAME RESTRICTION!")
                print("   🎮 Initiating game quit sequence...")
                return "cannot_perform"
            
            # If none found, it's unclear
            print("   ❓ Upgrade result unclear - no success, failure, or restriction detected")
            return "failed"
            
        except Exception as e:
            print(f"   ❌ Error during upgrade process: {e}")
            return "error"

    def update_inventory_after_upgrade(self, inventory_data, slot_index, upgrade_result):
        """
        Update inventory data after an upgrade attempt
        
        Parameters:
        - inventory_data: Current inventory data dictionary
        - slot_index: Index of the slot that was upgraded
        - upgrade_result: "success" or "failed"
        
        Returns:
        - dict: Updated inventory data
        """
        if slot_index not in inventory_data:
            print(f"❌ Slot {slot_index} not found in inventory data")
            return inventory_data
        
        slot_data = inventory_data[slot_index]
        item_name = slot_data.get('item_type', 'Unknown')
        current_template_value = slot_data.get('template_value', 0)
        
        if upgrade_result == "success":
            # Item upgraded successfully - increase template value by 1
            new_template_value = current_template_value + 1
            new_upgrade_level = f"+{new_template_value}"
            
            inventory_data[slot_index].update({
                'template_value': new_template_value,
                'upgrade_level': new_upgrade_level,
                'status': f"{item_name} {new_upgrade_level}",
                'template_status': f"Upgrade level: {new_upgrade_level}"
            })
            
            print(f"   ✅ Updated slot {slot_index}: {item_name} +{current_template_value} → +{new_template_value}")
            
        elif upgrade_result == "failed":
            # Item burned - make slot empty
            inventory_data[slot_index] = {
                'prediction': 6,  # 6 = Empty
                'item_type': 'Empty',
                'status': 'Empty slot',
                'template_value': 0,
                'upgrade_level': 'No upgrade',
                'template_status': 'No upgrade detected'
            }
            
            print(f"   💥 Updated slot {slot_index}: {item_name} +{current_template_value} → BURNED (empty)")
        
        return inventory_data

    def find_items_to_upgrade(self, inventory_data, locked_slots, target_from_level=1, target_to_level=3):
        """
        Find all items that need to be upgraded from one level to another
        
        Parameters:
        - inventory_data: Current inventory data
        - locked_slots: Set of locked slot indices
        - target_from_level: Current upgrade level to look for (default: 1 for +1 items)
        - target_to_level: Target upgrade level (default: 3 for +3 items)
        
        Returns:
        - list: List of slot indices that have items at target_from_level
        """
        items_to_upgrade = []
        
        for slot_idx, data in inventory_data.items():
            if slot_idx in locked_slots:
                continue  # Skip locked slots
            
            if data['prediction'] == 6:  # Skip empty slots
                continue
            
            template_value = data.get('template_value', 0)
            if template_value == target_from_level:
                item_name = data.get('item_type', 'Unknown')
                items_to_upgrade.append({
                    'slot_idx': slot_idx,
                    'item_name': item_name,
                    'current_level': template_value
                })
        
        print(f"\n🎯 FOUND {len(items_to_upgrade)} ITEMS TO UPGRADE FROM +{target_from_level} TO +{target_to_level}:")
        for item in items_to_upgrade:
            print(f"   Slot {item['slot_idx']}: {item['item_name']} +{item['current_level']}")
        
        return items_to_upgrade

    def upgrade_all_items_to_target_level(self, anvil_rects, inventory_data, locked_slots, target_from_level=1, target_to_level=3):
        """
        Upgrade all items from one level to target level (e.g., all +1 items to +3)
        
        Parameters:
        - anvil_rects: Anvil rectangle coordinates
        - inventory_data: Current inventory data
        - locked_slots: Set of locked slot indices
        - target_from_level: Starting level (default: 1 for +1)
        - target_to_level: Target level (default: 3 for +3)
        
        Returns:
        - dict: Updated inventory data
        - dict: Upgrade statistics
        """
        print(f"\n🚀 STARTING UPGRADE SESSION: +{target_from_level} → +{target_to_level}")
        
        upgrade_stats = {
            'total_attempts': 0,
            'successful_upgrades': 0,
            'failed_upgrades': 0,
            'items_completed': []
        }
        
        current_level = target_from_level
        
        # Keep upgrading until we reach target level
        while current_level < target_to_level:
            print(f"\n🎯 UPGRADE ROUND: +{current_level} → +{current_level + 1}")
            
            # Find items at current level
            items_to_upgrade = self.find_items_to_upgrade(inventory_data, locked_slots, current_level, current_level + 1)
            
            if not items_to_upgrade:
                print(f"   ✅ No more +{current_level} items found. Moving to next level...")
                current_level += 1
                continue
            
            # Check if empty slots >= 15 before starting upgrades
            empty_slots = self.count_empty_slots(inventory_data)
            if empty_slots >= 15:
                print(f"🚨 Empty slots reached {empty_slots} (>=15) - going to town to make decision")
                return inventory_data, upgrade_stats
            
            # Check if +6 items < 10 for +7 upgrades
            if current_level == 6:
                plus_six_count = self.count_plus_six_items(inventory_data)
                if plus_six_count < 10:
                    print(f"⚠️ Only {plus_six_count} +6 items available (need 10+) - going to town to make decision")
                    return inventory_data, upgrade_stats
            
            # Upgrade each item found
            for item_info in items_to_upgrade:
                slot_idx = item_info['slot_idx']
                item_name = item_info['item_name']
                
                print(f"\n🔧 Upgrading {item_name} +{current_level} in slot {slot_idx}...")
                
                # Check empty slots after each upgrade
                empty_slots = self.count_empty_slots(inventory_data)
                if empty_slots >= 15:
                    print(f"🚨 Empty slots reached {empty_slots} (>=15) after upgrade - going to town to make decision")
                    return inventory_data, upgrade_stats
                
                # Perform the upgrade using template detection
                upgrade_stats['total_attempts'] += 1
                result = self.perform_single_upgrade_with_template_detection(anvil_rects, locked_slots, slot_idx, current_level, inventory_data)
                
                # Update inventory based on result
                inventory_data = self.update_inventory_after_upgrade(inventory_data, slot_idx, result)
                
                # Update statistics and handle results
                if result == "success":
                    upgrade_stats['successful_upgrades'] += 1
                    print(f"   ✅ {item_name} successfully upgraded to +{current_level + 1}")
                elif result == "failed":
                    upgrade_stats['failed_upgrades'] += 1
                    print(f"   💥 {item_name} failed upgrade - item burned")
                elif result == "cannot_perform":
                    print(f"   🚫 {item_name} upgrade cannot be performed - game restriction detected")
                    print("\n🚨 GAME RESTRICTION DETECTED - QUITTING GAME!")
                    print("📊 UPGRADE SESSION SUMMARY BEFORE QUIT:")
                    print(f"   Total attempts: {upgrade_stats['total_attempts']}")
                    print(f"   Successful upgrades: {upgrade_stats['successful_upgrades']}")
                    print(f"   Failed upgrades: {upgrade_stats['failed_upgrades']}")
                    
                    if upgrade_stats['total_attempts'] > 0:
                        success_rate = (upgrade_stats['successful_upgrades'] / upgrade_stats['total_attempts']) * 100
                        print(f"   Success rate: {success_rate:.1f}%")
                    
                    print("\n🎮 Initiating game quit sequence...")
                    self.quit_game()
                    print("✅ Game quit completed!")
                    print("🔚 PROGRAM TERMINATING - Game session ended")
                    
                    # Import sys for program termination
                    import sys
                    sys.exit(0)
                else:
                    print(f"   ❓ {item_name} upgrade result unclear")
                
                # Small delay between upgrades
                time.sleep(1.0)
            
            # Move to next level
            current_level += 1
        
        # Final statistics
        print(f"\n📊 UPGRADE SESSION COMPLETED!")
        print(f"   Total attempts: {upgrade_stats['total_attempts']}")
        print(f"   Successful upgrades: {upgrade_stats['successful_upgrades']}")
        print(f"   Failed upgrades: {upgrade_stats['failed_upgrades']}")
        
        if upgrade_stats['total_attempts'] > 0:
            success_rate = (upgrade_stats['successful_upgrades'] / upgrade_stats['total_attempts']) * 100
            print(f"   Success rate: {success_rate:.1f}%")
        
        return inventory_data, upgrade_stats

    def count_empty_slots(self, inventory_data):
        """Count empty slots in inventory"""
        total_slots = 28  # 0-27
        occupied_slots = len([slot for slot, data in inventory_data.items() if data.get('prediction') != -1])
        empty_slots = total_slots - occupied_slots
        return empty_slots
    
    def count_plus_six_items(self, inventory_data):
        """Count +6 items in inventory"""
        plus_six_count = 0
        for slot_idx, slot_data in inventory_data.items():
            template_value = slot_data.get('template_value', 0)
            if template_value == 6:  # +6 items
                plus_six_count += 1
        print(f"🔍 DEBUG: Found {plus_six_count} +6 items in inventory")
        return plus_six_count

    def check_plus_seven_success_condition(self, inventory_data, locked_slots, target_count=5):
        """
        Check if we have achieved 10+ +7 items (success condition)
        
        Parameters:
        - inventory_data: Current inventory data
        - locked_slots: Set of locked slot indices
        - target_count: Target number of +7 items (default: 10)
        
        Returns:
        - bool: True if success condition met
        - int: Current count of +7 items
        """
        plus_seven_count = 0
        plus_seven_items = []
        
        for slot_idx, data in inventory_data.items():
            if slot_idx in locked_slots:
                continue
            
            if data['prediction'] != 6 and data.get('template_value', 0) == 7:
                plus_seven_count += 1
                plus_seven_items.append({
                    'slot_idx': slot_idx,
                    'item_name': data.get('item_type', 'Unknown'),
                    'status': data.get('status', '')
                })
        
        success_achieved = plus_seven_count >= target_count
        
        print(f"\n🏆 +7 ITEMS SUCCESS CHECK:")
        print(f"   Current +7 items: {plus_seven_count}/{target_count}")
        
        if plus_seven_items:
            print(f"   +7 Items found:")
            for item in plus_seven_items:
                print(f"      Slot {item['slot_idx']}: {item['status']}")
        
        if success_achieved:
            print(f"   🎉 SUCCESS CONDITION MET! {plus_seven_count} >= {target_count}")
        else:
            print(f"   📊 Progress: {plus_seven_count}/{target_count} +7 items")
        
        return success_achieved, plus_seven_count

    def count_items_by_level(self, inventory_data, locked_slots):
        """
        Count items by their upgrade levels
        
        Parameters:
        - inventory_data: Current inventory data
        - locked_slots: Set of locked slot indices
        
        Returns:
        - dict: Item counts by level {level: count}
        """
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
        empty_count = 0
        
        for slot_idx, data in inventory_data.items():
            if slot_idx in locked_slots:
                continue
                
            if data['prediction'] == 6:  # Empty slot
                empty_count += 1
            else:
                template_value = data.get('template_value', 0)
                if template_value in level_counts:
                    level_counts[template_value] += 1
        
        return level_counts, empty_count

    def determine_upgrade_phase(self, level_counts):
        """
        Determine current upgrade phase based on inventory composition
        
        Parameters:
        - level_counts: Dictionary of item counts by level
        
        Returns:
        - str: Current phase ("foundation", "pipeline", "collecting_sevens", "success")
        """
        plus_six_count = level_counts.get(6, 0)
        plus_seven_count = level_counts.get(7, 0)
        
        if plus_seven_count >= 10:
            return "success"
        elif plus_six_count < 15:
            return "pipeline"
        elif plus_six_count >= 15:
            return "collecting_sevens"
        else:
            return "foundation"

    def calculate_restock_threshold(self, current_phase):
        """
        Calculate restocking threshold based on current phase
        
        Parameters:
        - current_phase: Current upgrade phase
        
        Returns:
        - int: Empty slots threshold for restocking
        """
        thresholds = {
            "foundation": 15,
            "pipeline": 12,
            "collecting_sevens": 8,
            "success": 0  # No restocking needed
        }
        
        return thresholds.get(current_phase, 10)

    def calculate_batch_size_for_plus_seven(self, plus_six_count):
        """
        Calculate optimal batch size for +6→+7 upgrades
        
        Parameters:
        - plus_six_count: Number of +6 items available
        
        Returns:
        - int: Recommended batch size
        """
        if plus_six_count > 25:
            return 10  # Aggressive batch
        elif plus_six_count > 20:
            return 8   # Moderate batch
        elif plus_six_count > 15:
            return 5   # Conservative batch
        else:
            return 0   # Need to rebuild pipeline first

    def remove_stored_plus_seven_items_from_inventory(self, inventory_data, locked_slots):
        """
        Remove all +7 items from inventory data after they've been stored in bank
        This updates the inventory to reflect that +7 items are no longer in inventory
        
        Parameters:
        - inventory_data: Current inventory data
        - locked_slots: Set of locked slot indices
        
        Returns:
        - dict: Updated inventory data with +7 items marked as empty
        """
        updated_inventory = inventory_data.copy()
        removed_count = 0
        
        for slot_idx, data in updated_inventory.items():
            if slot_idx in locked_slots:
                continue
            if data['prediction'] != 6 and data.get('template_value', 0) == 7:
                # Mark slot as empty (prediction = 6)
                updated_inventory[slot_idx] = {
                    'prediction': 6,
                    'item_type': 'Empty',
                    'status': 'Empty slot',
                    'template_value': 0,
                    'upgrade_level': '+0',
                    'template_status': 'Empty slot'
                }
                removed_count += 1
        
        print(f"📦 Marked {removed_count} +7 item slots as empty after bank storage")
        return updated_inventory

    def should_store_plus_seven_items(self, inventory_data, locked_slots, threshold=6):
        """
        Check if we have enough +7 items to trigger bank storage
        
        Parameters:
        - inventory_data: Current inventory data
        - locked_slots: Set of locked slot indices  
        - threshold: Minimum number of +7 items to trigger storage (default: 6)
        
        Returns:
        - bool: True if we should go to bank to store +7 items
        """
        plus_seven_count = 0
        
        for slot_idx, data in inventory_data.items():
            if slot_idx in locked_slots:
                continue
            if data['prediction'] != 6 and data.get('template_value', 0) == 7:
                plus_seven_count += 1
        
        print(f"🏦 Bank storage check: {plus_seven_count} +7 items found (threshold: {threshold})")
        
        if plus_seven_count >= threshold:
            print(f"✅ Bank storage triggered! {plus_seven_count} +7 items ready for storage")
            return True
        else:
            print(f"ℹ️ Bank storage not needed yet ({plus_seven_count}/{threshold})")
            return False

    def navigate_to_bank_and_store_plus_seven_items(self, nav_system, inventory_data, locked_slots):
        """
        Navigate to bank and store all +7 items, then go to town
        Navigation flow: anvil → armor → inn (bank) → store items → town
        After this, normal flow will handle restocking via town → anvil → armor
        
        Parameters:
        - nav_system: Navigation system for movement
        - inventory_data: Current inventory data
        - locked_slots: Set of locked slot indices
        
        Returns:
        - tuple: (success, updated_inventory_data) - updated inventory after storage only
        """
        print("\n🏦 INITIATING BANK STORAGE SEQUENCE FOR +7 ITEMS...")
        
        # Count +7 items first
        plus_seven_items = []
        for slot_idx, data in inventory_data.items():
            if slot_idx in locked_slots:
                continue
            if data['prediction'] != 6 and data.get('template_value', 0) == 7:
                plus_seven_items.append({
                    'slot_idx': slot_idx,
                    'item_name': data.get('item_type', 'Unknown'),
                    'status': data.get('status', '')
                })
        
        if not plus_seven_items:
            print("❌ No +7 items found to store!")
            return False, inventory_data
        
        print(f"🎯 Found {len(plus_seven_items)} +7 items to store:")
        for item in plus_seven_items:
            print(f"   Slot {item['slot_idx']}: {item['status']}")
        
        # Step 1: Navigate anvil → armor → inn (bank)
        print("\n🗺️ Step 1: Navigating to bank via anvil → armor → inn...")
        
        # Try direct path first: anvil → armor → inn
        print("🔨➡️🛡️ Substep 1a: Going from anvil to armor...")
        anvil_to_armor_success = nav_system.navigate_with_agent("anvil_to_armor", max_steps=200, distance_threshold=4)
        
        if anvil_to_armor_success:
            print("✅ Successfully reached armor from anvil!")
            time.sleep(2)
            
            print("🛡️➡️🏨 Substep 1b: Going from armor to inn...")
            armor_to_inn_success = nav_system.navigate_with_agent("armor_to_inn", max_steps=200, distance_threshold=4)
            
            if armor_to_inn_success:
                print("✅ Successfully reached inn (bank) from armor!")
                bank_nav_success = True
            else:
                print("❌ Failed to reach inn from armor")
                bank_nav_success = False
        else:
            print("❌ Failed to reach armor from anvil")
            bank_nav_success = False
        
        # Fallback: use town navigation if direct path failed
        if not bank_nav_success:
            print("\n🔄 Direct path failed - using fallback: town → anvil → armor → inn...")
            
            print("🏠 Going to town...")
            town_success = self.town_()
            if not town_success:
                print("❌ Failed to reach town")
                return False, inventory_data
            
            print("🏠➡️🔨 Going from town to anvil...")
            town_to_anvil_success = nav_system.navigate_with_agent("town_to_anvil", max_steps=200, distance_threshold=8)
            if not town_to_anvil_success:
                print("❌ Failed to reach anvil from town")
                return False, inventory_data
            
            print("🔨➡️🛡️ Going from anvil to armor...")
            anvil_to_armor_success = nav_system.navigate_with_agent("anvil_to_armor", max_steps=200, distance_threshold=4)
            if not anvil_to_armor_success:
                print("❌ Failed to reach armor from anvil")
                return False, inventory_data
            
            print("🛡️➡️🏨 Going from armor to inn...")
            armor_to_inn_success = nav_system.navigate_with_agent("armor_to_inn", max_steps=200, distance_threshold=4)
            if not armor_to_inn_success:
                print("❌ Failed to reach inn from armor")
                return False, inventory_data
            
            print("✅ Successfully reached inn via fallback route!")
        
        time.sleep(2)
        
        # Step 2: Interact with bank (approach and open)
        print("\n🏛️ Step 2: Interacting with bank...")
        bank_interaction_success = self.interact_with_bank()
        
        if not bank_interaction_success:
            print("❌ Direct bank interaction failed - trying fallback navigation...")
            
            # Fallback: town → anvil → armor → inn → try bank again
            print("\n🔄 FALLBACK: Using town navigation...")
            
            print("🏠 Going to town...")
            town_success = self.town_()
            if not town_success:
                print("❌ Failed to reach town")
                return False, inventory_data
            
            print("🏠➡️🔨 Going from town to anvil...")
            town_to_anvil_success = nav_system.navigate_with_agent("town_to_anvil", max_steps=200, distance_threshold=8)
            if not town_to_anvil_success:
                print("❌ Failed to reach anvil from town")
                return False, inventory_data
            
            print("🔨➡️🛡️ Going from anvil to armor...")
            anvil_to_armor_success = nav_system.navigate_with_agent("anvil_to_armor", max_steps=200, distance_threshold=4)
            if not anvil_to_armor_success:
                print("❌ Failed to reach armor from anvil")
                return False, inventory_data
            
            print("🛡️➡️🏨 Going from armor to inn...")
            armor_to_inn_success = nav_system.navigate_with_agent("armor_to_inn", max_steps=200, distance_threshold=4)
            if not armor_to_inn_success:
                print("❌ Failed to reach inn from armor")
                return False, inventory_data
            
            time.sleep(2)
            
            # Try bank interaction again
            print("🏦 Trying bank interaction again after fallback navigation...")
            bank_interaction_success = self.interact_with_bank()
            
            if not bank_interaction_success:
                print("❌ Bank interaction failed even after fallback navigation!")
                return False, inventory_data
            
            print("✅ Bank interaction successful after fallback!")
        else:
            print("✅ Bank interaction successful on first try!")
        
        # Step 3: Store +7 items in bank
        print(f"\n💎 Step 3: Storing {len(plus_seven_items)} +7 items in bank...")
        
        # Define bank inventory rectangle coordinates (from process_inventory_test)
        bank_inventory_top_left = (654, 430)
        bank_inventory_bottom_right = (691, 467)
        bank_inventory_rects = self.draw_rectangle_grid(None, bank_inventory_top_left, bank_inventory_bottom_right, rows=4, cols=7, spacing=13)
        
        # Define anvil rectangles for inventory items
        anvil_top_left = (652, 439)
        anvil_bottom_right = (695, 481)
        anvil_rects = self.draw_rectangle_grid(None, anvil_top_left, anvil_bottom_right, rows=4, cols=7, spacing=7)
        
        stored_count = 0
        
        for item in plus_seven_items:
            slot_idx = item['slot_idx']
            item_name = item['item_name']
            
            print(f"\n📦 Storing {item_name} from slot {slot_idx}...")
            
            try:
                # Right-click on the +7 item in inventory
                if slot_idx < len(anvil_rects):
                    inventory_rect = anvil_rects[slot_idx]
                    print(f"   🖱️ Right-clicking {item_name} at slot {slot_idx}...")
                    clicked_coords = self.click_middle_of_rectangle(inventory_rect, button='right')
                    print(f"   ✅ Right-clicked at: {clicked_coords}")
                    time.sleep(0.5)
                    
                    # Find the first empty slot in bank inventory
                    bank_slot_found = False
                    for bank_slot_idx, bank_rect in enumerate(bank_inventory_rects):
                        # For now, we'll use sequential slots (you can add empty slot detection later)
                        if bank_slot_idx < 28:  # 4x7 = 28 total bank slots
                            print(f"   🏦 Placing item in bank slot {bank_slot_idx}...")
                            bank_clicked_coords = self.click_middle_of_rectangle(bank_rect, button='left')
                            print(f"   ✅ Clicked bank slot at: {bank_clicked_coords}")
                            time.sleep(0.5)
                            
                            stored_count += 1
                            print(f"   💎 Successfully stored {item_name}!")
                            bank_slot_found = True
                            break
                    
                    if not bank_slot_found:
                        print(f"   ⚠️ No available bank slots for {item_name}")
                else:
                    print(f"   ❌ Invalid slot index {slot_idx}")
                    
            except Exception as e:
                print(f"   ❌ Error storing {item_name}: {e}")
                continue
        
        print(f"\n💎 BANK STORAGE COMPLETED!")
        print(f"📊 Successfully stored: {stored_count}/{len(plus_seven_items)} +7 items")
        
        if stored_count == 0:
            print("⚠️ No items were stored successfully")
            return False, inventory_data
        
        # Step 4: Go to town (let normal flow handle restocking)
        print(f"\n🏠 Step 4: Going to town...")
        print("💡 Normal flow will handle restocking via town → anvil → armor")
        
        town_success = self.town_()
        
        if not town_success:
            print("❌ Failed to reach town")
            # Update inventory anyway to reflect stored items
            updated_inventory = self.remove_stored_plus_seven_items_from_inventory(inventory_data, locked_slots)
            return False, updated_inventory
        
        print("✅ Successfully reached town!")
        
        # Update inventory data: remove stored +7 items only
        print("\n📦 Updating inventory data...")
        updated_inventory = self.remove_stored_plus_seven_items_from_inventory(inventory_data, locked_slots)
        
        print(f"\n🎉 BANK STORAGE SEQUENCE COMPLETED!")
        print(f"💎 Stored {stored_count} +7 items in bank safely")
        print("🏠 At town - normal flow will handle restocking and return to anvil")
        
        return True, updated_inventory

    def execute_restock_cycle(self, nav_system, inventory_data, locked_slots, empty_count):
        """
        Execute complete restocking cycle: town → anvil → armor → buy items → town
        Handles navigation from town (not anvil) for retry compatibility
        
        Parameters:
        - nav_system: Navigation system for movement
        - inventory_data: Current inventory data
        - locked_slots: Set of locked slot indices
        - empty_count: Number of empty slots to fill
        
        Returns:
        - tuple: (success, purchased_items) - updated inventory after restocking
        """
        print("\n🔄 EXECUTING COMPLETE RESTOCK CYCLE...")
        
        try:
            # Step 1: Navigate from town to anvil first
            print("🗺️ Step 1: Navigating from town to anvil...")
            anvil_nav_success = nav_system.navigate_with_agent("town_to_anvil", max_steps=200, distance_threshold=8)
            
            if not anvil_nav_success:
                print("❌ Failed to navigate to anvil from town")
                return False, []
            
            time.sleep(2)
            
            # Step 2: Navigate from anvil to armor shop
            print("🗺️ Step 2: Navigating from anvil to armor shop...")
            armor_nav_success = nav_system.navigate_with_agent("anvil_to_armor", max_steps=200, distance_threshold=4)
            
            if not armor_nav_success:
                print("❌ Failed to navigate to armor shop")
                return False, []
            
            time.sleep(2)
            
            # Step 3: Find correct NPC at armor shop
            print("🔍 Step 3: Finding correct NPC at armor shop...")
            npc_found = self.find_correct_npc_at_armor()
            
            if not npc_found:
                print("❌ Failed to find correct NPC at armor shop")
                return False, []
            
            # Step 4: Open armor shop
            print("🛡️ Step 4: Opening armor shop...")
            armor_opened = self.open_npc("armor")
            
            if not armor_opened:
                print("❌ Failed to open armor shop")
                return False, []
            
            # Step 5: Buy items to fill empty slots
            print(f"🛒 Step 5: Buying {empty_count} items to fill empty slots...")
            
            # Use existing purchase system (sundires rectangles for armor shop)
            sundires_top_left = (667, 270)
            sundires_bottom_right = (712, 312)
            sundires_rects = self.draw_rectangle_grid(None, sundires_top_left, sundires_bottom_right, rows=1, cols=5, spacing=7)
            
            # Calculate items to buy based on empty slots
            purchase_plan = self.calculate_restock_purchase_plan(empty_count)
            
            purchase_success, purchased_items = self.buy_items_from_sundires(sundires_rects, purchase_plan)
            
            if not purchase_success:
                print("❌ Failed to purchase items")
                return False, []
            
            print(f"✅ Successfully purchased {len(purchased_items)} items")
            
            # Step 6: Go to town (let normal flow handle going to anvil)
            print("🏠 Step 6: Going to town...")  
            print("💡 Normal flow will handle inventory check and return to anvil")
            
            town_success = self.town_()
            
            if not town_success:
                print("❌ Failed to reach town")
                return False, purchased_items
            
            print("✅ Successfully reached town!")
            print("✅ RESTOCK CYCLE COMPLETED SUCCESSFULLY!")
            print("🏠 At town - normal flow will handle going to anvil")
            
            return True, purchased_items
            
        except Exception as e:
            print(f"❌ Error during restock cycle: {e}")
            return False, []

    def calculate_restock_purchase_plan(self, empty_count):
        """
        Calculate what items to buy during restocking
        
        Parameters:
        - empty_count: Number of empty slots to fill
        
        Returns:
        - list: Purchase order for sundires items
        """
        print(f"\n🛒 CALCULATING RESTOCK PURCHASE PLAN FOR {empty_count} ITEMS...")
        
        # Item type mapping for sundires rectangles
        sundires_item_map = {
            0: 1,  # Pauldron
            1: 2,  # Pant
            2: 4,  # Hat
            3: 5,  # Glove
            4: 3   # Boot
        }
        
        purchase_order = []
        items_to_buy = min(empty_count, 25)  # Don't exceed inventory capacity
        
        # Distribute purchases evenly across item types
        items_per_type = items_to_buy // 5
        remaining_items = items_to_buy % 5
        
        for sundires_slot in range(5):
            item_type_id = sundires_item_map[sundires_slot]
            count = items_per_type
            
            # Add one extra item to first few types if there's remainder
            if sundires_slot < remaining_items:
                count += 1
            
            # Add items of this type to purchase order
            for _ in range(count):
                purchase_order.append({
                    'sundires_slot': sundires_slot,
                    'item_type_id': item_type_id,
                    'current_count': 0,  # New items
                    'purchase_round': len(purchase_order) + 1
                })
        
        print(f"🎯 RESTOCK PURCHASE PLAN ({len(purchase_order)} items):")
        item_names = {1: "Pauldron", 2: "Pant", 3: "Boot", 4: "Hat", 5: "Glove"}
        
        for i, item in enumerate(purchase_order):
            item_name = item_names.get(item['item_type_id'], f"Item{item['item_type_id']}")
            print(f"   {i+1:2d}. Sundires slot {item['sundires_slot']}: {item_name}")
        
        return purchase_order

    def rebuild_inventory_data_after_restock(self, predictions, anvil_rects, locked_slots):
        """
        Rebuild inventory data after restocking with new items
        
        Parameters:
        - predictions: ML predictions for inventory slots
        - anvil_rects: Anvil rectangle coordinates
        - locked_slots: Set of locked slot indices
        
        Returns:
        - dict: Updated inventory data
        """
        print("\n📦 REBUILDING INVENTORY DATA AFTER RESTOCK...")
        
        # Get template values for upgrade levels (optimized with ML predictions)
        template_values = self.navigate_rectangles(anvil_rects, click=False, wait_time=0.2, ml_predictions=predictions)
        
        # Item type mapping
        item_type_map = {
            0: "Upgrade Scroll",
            1: "Pauldron", 
            2: "Pant",
            3: "Boot",
            4: "Hat",
            5: "Glove",
            6: "Empty"
        }
        
        inventory_data = {}
        
        for slot_idx, prediction in enumerate(predictions):
            template_value = template_values.get(slot_idx, 0)
            
            if slot_idx in locked_slots:
                item_type = "LOCKED (Upgrade Scroll)"
                item_status = "Locked slot - Upgrade Scroll"
            elif prediction == 6:
                item_type = "Empty"
                item_status = "Empty slot"
            elif prediction in item_type_map:
                item_type = item_type_map[prediction]
                if template_value > 0:
                    item_status = f"{item_type} +{template_value}"
                else:
                    item_status = item_type
            else:
                item_type = f"UNKNOWN({prediction})"
                item_status = f"Unknown item type: {prediction}"
            
            inventory_data[slot_idx] = {
                'prediction': prediction,
                'item_type': item_type,
                'status': item_status,
                'template_value': template_value,
                'upgrade_level': f"+{template_value}" if template_value > 0 else "No upgrade",
                'template_status': f"Upgrade level: +{template_value}" if template_value > 0 else "No upgrade detected"
            }
        
        print("✅ Inventory data rebuilt successfully")
        return inventory_data
    
    def rebuild_inventory_data_with_template_scan(self, predictions, anvil_rects, locked_slots):
        """
        Rebuild inventory data after restocking WITH proper template value scanning
        This ensures +1 items are detected correctly for upgrade system
        """
        print("🔍 Rebuilding inventory data with template scanning...")
        
        inventory_data = {}
        item_type_map = {0: "Pant", 1: "Pauldron", 2: "Helmet", 3: "Glove", 4: "Boot", 5: "Pad", 6: "Empty"}
        
        # Take a fresh screenshot for template matching
        screenshot = self.capture_game_window()
        
        for slot_idx, rect in enumerate(anvil_rects):
            prediction = predictions[slot_idx] if slot_idx < len(predictions) else 6
            template_value = 0
            
            # For non-empty, non-locked slots, scan for upgrade level templates
            if not (slot_idx in locked_slots or prediction == 6):
                try:
                    # Extract the specific slot area for template matching
                    x1, y1, x2, y2 = rect
                    slot_area = screenshot[y1:y2, x1:x2]
                    
                    # Check for upgrade level templates (+1 through +7)
                    for level in range(1, 8):
                        template_path = f"game_images/plus_{['one', 'two', 'three', 'four', 'five', 'six', 'seven'][level-1]}.png"
                        try:
                            template = cv2.imread(template_path, 0)
                            if template is not None:
                                slot_gray = cv2.cvtColor(slot_area, cv2.COLOR_BGR2GRAY)
                                result = cv2.matchTemplate(slot_gray, template, cv2.TM_CCOEFF_NORMED)
                                _, max_val, _, _ = cv2.minMaxLoc(result)
                                
                                if max_val > 0.7:  # Template match threshold
                                    template_value = level
                                    break
                        except:
                            continue
                except:
                    template_value = 0
            
            # Build inventory entry
            if slot_idx in locked_slots:
                item_type = "LOCKED (Upgrade Scroll)"
                item_status = "Locked slot - Upgrade Scroll"
            elif prediction == 6:
                item_type = "Empty"
                item_status = "Empty slot"
            elif prediction in item_type_map:
                item_type = item_type_map[prediction]
                if template_value > 0:
                    item_status = f"{item_type} +{template_value}"
                else:
                    item_status = item_type
            else:
                item_type = f"UNKNOWN({prediction})"
                item_status = f"Unknown item type: {prediction}"
            
            inventory_data[slot_idx] = {
                'prediction': prediction,
                'item_type': item_type,
                'status': item_status,
                'template_value': template_value,
                'upgrade_level': f"+{template_value}" if template_value > 0 else "No upgrade",
                'template_status': f"Upgrade level: +{template_value}" if template_value > 0 else "No upgrade detected"
            }
        
        print("✅ Inventory data rebuilt with template scanning")
        return inventory_data
    
    def test_anvil_upgrade_with_auto_restock(self, nav_system, model):
        """
        Test function to be called when already at anvil
        - Scans inventory 
        - If empty slots > 10: navigate to armor, buy items, return to anvil
        - Otherwise: start upgrade system
        """
        print("\n🧪 TESTING ANVIL UPGRADE WITH AUTO RESTOCK")
        print("🔧 Assuming we're already at the anvil...")
        print("=" * 60)
        
        # Define rectangles and locked slots
        anvil_top_left = (652, 439)
        anvil_bottom_right = (695, 481)
        anvil_rects = self.draw_rectangle_grid(None, anvil_top_left, anvil_bottom_right, rows=4, cols=7, spacing=7)
        locked_slots = set([0])  # First slot locked
        
        # Step 1: Scan current inventory at anvil
        print("\n📊 Step 1: Scanning anvil inventory...")
        screenshot = self.capture_game_window()
        
        if screenshot is None:
            print("❌ Failed to capture game window")
            return False
            
        predictions = self.predict_and_draw_rectangles(
            screenshot, model, anvil_rects, locked_slots,
            rect_name="Anvil_Inventory", rect_color=(0, 255, 0)
        )
        
        if not predictions:
            print("❌ Failed to scan anvil inventory")
            return False
        
        # Build inventory data with proper template scanning
        inventory_data = self.rebuild_inventory_data_with_template_scan(predictions, anvil_rects, locked_slots)
        
        # Step 2: Count empty slots
        level_counts, empty_count = self.count_items_by_level(inventory_data, locked_slots)
        
        print(f"\n📊 INVENTORY ANALYSIS:")
        print(f"   +1:{level_counts[1]} +2:{level_counts[2]} +3:{level_counts[3]} +4:{level_counts[4]} +5:{level_counts[5]} +6:{level_counts[6]} +7:{level_counts[7]}")
        print(f"   Empty slots: {empty_count}")
        print(f"   🔍 Decision check: empty_count ({empty_count}) > 10? {empty_count > 10}")
        
        # Step 3a: PRIORITY CHECK - Bank storage for +7 items (higher priority than restocking)
        if self.should_store_plus_seven_items(inventory_data, locked_slots, threshold=6):
            print("\n🏦 BANK STORAGE TRIGGERED! (Higher priority than restocking)")
            print("💎 Going to store +7 items in bank before any restocking...")
            
            bank_success, updated_inventory = self.navigate_to_bank_and_store_plus_seven_items(nav_system, inventory_data, locked_slots)
            
            if bank_success:
                print("✅ Bank storage completed successfully!")
                print("🏠 Player is now at town - will restart normal flow")
                # Use the updated inventory data from the bank storage process
                inventory_data = updated_inventory
                # After bank storage, restart the entire process from town
                return self.test_anvil_upgrade_with_auto_restock(nav_system, model)
            else:
                print("⚠️ Bank storage failed, but continuing...")
                # Still use updated inventory if partial success occurred
                inventory_data = updated_inventory
                level_counts, empty_count = self.count_items_by_level(inventory_data, locked_slots)
        
        # Step 3b: Secondary check - restock if empty slots > 10 (only if bank storage didn't happen)
        elif empty_count > 10:
            print(f"\n🛒 RESTOCKING NEEDED: {empty_count} empty slots > 10")
            print("🔄 Starting restock cycle: anvil → armor → buy → anvil")
            print("⚠️ IGNORING ANY EXISTING +1 ITEMS - RESTOCKING FIRST!")
            
            # Execute restock cycle
            restock_success, purchased_items = self.execute_restock_cycle(nav_system, inventory_data, locked_slots, empty_count)
            
            if restock_success:
                print("✅ Restocking completed successfully!")
                print("🏠 Player is now at town - will restart normal flow")
                print(f"🎁 Purchased {len(purchased_items)} new items:")
                for item in purchased_items:
                    print(f"   - {item['item_name']} +{item['upgrade_level']}")
                
                # Update inventory data with purchased items
                print("\n📦 Updating inventory with purchased items...")
                inventory_data = self.update_inventory_with_purchases(inventory_data, purchased_items, locked_slots)
                # After restocking, restart the entire process from town
                return self.test_anvil_upgrade_with_auto_restock(nav_system, model)
                
                # Method 2: Re-scan inventory for verification (BACKUP)
                print("\n📊 Re-scanning inventory for verification...")
                screenshot = self.capture_game_window()
                predictions = self.predict_and_draw_rectangles(
                    screenshot, model, anvil_rects, locked_slots,
                    rect_name="Anvil_Post_Restock", rect_color=(0, 255, 0)
                )
                
                if predictions:
                    # Verify with template scanning and use if better
                    scanned_inventory = self.rebuild_inventory_data_with_template_scan(predictions, anvil_rects, locked_slots)
                    scanned_counts, scanned_empty = self.count_items_by_level(scanned_inventory, locked_slots)
                    purchase_counts, purchase_empty = self.count_items_by_level(inventory_data, locked_slots)
                    
                    print(f"📊 INVENTORY COMPARISON:")
                    print(f"   From purchases: +1:{purchase_counts[1]} empty:{purchase_empty}")
                    print(f"   From scanning:  +1:{scanned_counts[1]} empty:{scanned_empty}")
                    
                    # Use scanned data if it shows more +1 items (better detection)
                    if scanned_counts[1] >= purchase_counts[1]:
                        print("✅ Using scanned inventory data (more reliable)")
                        inventory_data = scanned_inventory
                    else:
                        print("✅ Using purchase-updated inventory data (more accurate)")
                    
                    level_counts, empty_count = self.count_items_by_level(inventory_data, locked_slots)
                    print(f"📦 Final inventory - Empty slots: {empty_count}")
                    print(f"📦 Item levels after restock: +1:{level_counts[1]} +2:{level_counts[2]} +3:{level_counts[3]} +4:{level_counts[4]} +5:{level_counts[5]} +6:{level_counts[6]} +7:{level_counts[7]}")
                    
                    # Now start upgrade system
                    print("\n🚀 Starting upgrade system after restocking...")
                    self.start_upgrade_system_after_restock(anvil_rects, inventory_data, locked_slots, nav_system)
                else:
                    print("⚠️ Template scanning failed, using purchase data only")
                    level_counts, empty_count = self.count_items_by_level(inventory_data, locked_slots)
                    print(f"📦 Final inventory - Empty slots: {empty_count}")
                    print(f"📦 Item levels after restock: +1:{level_counts[1]} +2:{level_counts[2]} +3:{level_counts[3]} +4:{level_counts[4]} +5:{level_counts[5]} +6:{level_counts[6]} +7:{level_counts[7]}")
                    
                    # Now start upgrade system
                    print("\n🚀 Starting upgrade system after restocking...")
                    self.start_upgrade_system_after_restock(anvil_rects, inventory_data, locked_slots, nav_system)
            else:
                print("❌ Restocking failed - continuing with current inventory")
                
        else:
            print(f"\n🔧 STARTING UPGRADES: {empty_count} empty slots ≤ 10")
            print("🚀 No restocking needed, proceeding with upgrades...")
            print(f"🎯 Will prioritize existing +1 items: {level_counts[1]} items")
            
            # Start upgrade system directly
            self.start_upgrade_system_after_restock(anvil_rects, inventory_data, locked_slots, nav_system)
        
        return True
    
    def start_upgrade_system_after_restock(self, anvil_rects, inventory_data, locked_slots, nav_system):
        """Start upgrade system after inventory check/restock - PRIORITIZE +1 ITEMS FIRST"""
        print("\n🎯 LAUNCHING UPGRADE SYSTEM WITH +1 PRIORITY...")
        
        # Check if we have +1 items to process first
        level_counts, empty_count = self.count_items_by_level(inventory_data, locked_slots)
        
        if level_counts[1] > 0:
            print(f"🌱 FOUND {level_counts[1]} NEW +1 ITEMS - UPGRADING THESE FIRST!")
            print("🎯 Starting with +1 → +3 upgrades...")
            
            # Upgrade +1 items to +3 first
            inventory_data, upgrade_result = self.upgrade_all_items_to_target_level(
                anvil_rects, inventory_data, locked_slots,
                target_from_level=1, target_to_level=3
            )
            
            if upgrade_result['successful_upgrades'] > 0:
                print(f"✅ Successfully upgraded {upgrade_result['successful_upgrades']} items from +1 to +3!")
            
            # Continue with the advanced system for remaining items
            print("\n🚀 Now launching advanced system for remaining upgrades...")
            updated_inventory, session_stats, success_achieved = self.advanced_upgrade_system_to_plus_seven(
                anvil_rects, inventory_data, locked_slots, nav_system, None
            )
            
            # Combine stats
            session_stats['successful_upgrades'] += upgrade_result['successful_upgrades']
            session_stats['total_attempts'] += upgrade_result['total_attempts']
            session_stats['failed_upgrades'] += upgrade_result['failed_upgrades']
            
        else:
            print("ℹ️ No +1 items found, launching standard advanced system...")
            updated_inventory, session_stats, success_achieved = self.advanced_upgrade_system_to_plus_seven(
                anvil_rects, inventory_data, locked_slots, nav_system, None
            )
        
        # Display results
        print(f"\n📊 UPGRADE SESSION RESULTS:")
        print(f"   Total attempts: {session_stats['total_attempts']}")
        print(f"   Successful upgrades: {session_stats['successful_upgrades']}")
        print(f"   Failed upgrades: {session_stats['failed_upgrades']}")
        print(f"   Restock cycles: {session_stats['restock_cycles']}")
        print(f"   +7 items achieved: {session_stats['plus_seven_achieved']}")
        
        if success_achieved:
            print("\n🎉 UPGRADE SESSION COMPLETED SUCCESSFULLY!")
            print("🏆 Target achieved: 10+ items at +7 level!")
        else:
            print("\n⚠️ Upgrade session ended")
        
        return updated_inventory, session_stats, success_achieved

    def advanced_upgrade_system_to_plus_seven(self, anvil_rects, inventory_data, locked_slots, nav_system, model=None):
        """
        Advanced upgrade system targeting 10+ +7 items with dynamic restocking
        ALWAYS STARTS WITH FRESH INVENTORY SCAN AT ANVIL
        
        Parameters:
        - anvil_rects: Anvil rectangle coordinates
        - inventory_data: Current inventory data (will be refreshed)
        - locked_slots: Set of locked slot indices
        - nav_system: Navigation system for restocking
        - model: ML model for inventory scanning
        
        Returns:
        - dict: Updated inventory data
        - dict: Session statistics
        - bool: True if success condition (10+ +7 items) achieved
        """
        print("\n🚀 STARTING ADVANCED +7 COLLECTION SYSTEM")
        print("🎯 TARGET: Collect 10+ items at +7 level")
        print("🔍 USING INVENTORY DATA FROM TOWN SCAN (NO DUPLICATE SCANNING)")
        print("=" * 60)
        
        # Use the inventory data that was already scanned at town - no need to scan again!
        print("\n📊 STEP 1: USING EXISTING INVENTORY DATA...")
        print("✅ Using inventory data from town scan - avoiding duplicate scanning")
        
        # STEP 2: ANALYZE INVENTORY AND DECIDE ACTION
        level_counts, empty_count = self.count_items_by_level(inventory_data, locked_slots)
        
        print(f"\n📊 FRESH INVENTORY ANALYSIS:")
        print(f"   +1:{level_counts[1]} +2:{level_counts[2]} +3:{level_counts[3]} +4:{level_counts[4]} +5:{level_counts[5]} +6:{level_counts[6]} +7:{level_counts[7]}")
        print(f"   Empty slots: {empty_count}")
        
        # STEP 3a: PRIORITY CHECK - Bank storage for +7 items (higher priority than restocking)
        if self.should_store_plus_seven_items(inventory_data, locked_slots, threshold=6):
            print("\n🏦 BANK STORAGE TRIGGERED! (Higher priority than restocking)")
            print("💎 Going to store +7 items in bank before any restocking...")
            
            bank_success, updated_inventory = self.navigate_to_bank_and_store_plus_seven_items(nav_system, inventory_data, locked_slots)
            
            if bank_success:
                print("✅ Bank storage completed successfully!")
                print("🏠 Player is now at town - will restart normal flow")
                # Use the updated inventory data from the bank storage process
                inventory_data = updated_inventory
                # After bank storage, restart the entire process from town
                return self.advanced_upgrade_system_to_plus_seven(anvil_rects, inventory_data, locked_slots, nav_system, model)
            else:
                print("⚠️ Bank storage failed, but continuing...")
                # Still use updated inventory if partial success occurred
                inventory_data = updated_inventory
                level_counts, empty_count = self.count_items_by_level(inventory_data, locked_slots)
        
        # STEP 3b: IMMEDIATE RESTOCK CHECK - IF >10 EMPTY, GO TO ARMOR FIRST (only if bank storage didn't happen)
        elif empty_count > 10:
            print(f"\n🛒 RESTOCKING REQUIRED: {empty_count} empty slots > 10")
            print("🔄 Going to armor shop to buy items first...")
            
            # Keep trying restock until successful - MANDATORY for >10 empty slots
            restock_attempt = 0
            restock_success = False
            
            while not restock_success:
                restock_attempt += 1
                print(f"\n🔄 RESTOCK ATTEMPT {restock_attempt} (MANDATORY - WILL NOT GIVE UP)")
                restock_success, purchased_items = self.execute_restock_cycle(nav_system, inventory_data, locked_slots, empty_count)
                
                if restock_success:
                    print(f"✅ RESTOCK SUCCESSFUL on attempt {restock_attempt}!")
                    print(f"🎁 Purchased {len(purchased_items)} new +1 items")
                    print("🏠 Restocking completed! Returning to town to restart process...")
                    
                    # Go to town after successful restock (same pattern as bank storage)
                    town_success = self.town_()
                    if town_success:
                        print("✅ Successfully returned to town after restocking")
                        print("🔄 Restarting upgrade process from town with restocked inventory...")
                        # Restart the entire process from town
                        return self.advanced_upgrade_system_to_plus_seven(anvil_rects, inventory_data, locked_slots, nav_system, model)
                    else:
                        print("❌ Failed to return to town after restocking!")
                        return False
                    
                else:
                    print(f"❌ Restock attempt {restock_attempt} FAILED")
                    print("🔄 GOING BACK TO TOWN AND RETRYING...")
                    print("⚠️ SYSTEM WILL NOT CONTINUE UNTIL RESTOCK SUCCEEDS")
                    
                    # Go back to town for next attempt
                    town_success = self.town_()
                    if town_success:
                        print("✅ Returned to town successfully")
                        print(f"🔄 Preparing for restock attempt {restock_attempt + 1}...")
                        time.sleep(3)  # Wait before next attempt
                    else:
                        print("❌ CRITICAL: Failed to return to town!")
                        print("🔄 Will still try to continue restock from current location...")
                        time.sleep(2)
        
        print(f"\n🚀 STEP 4: STARTING UPGRADE SESSION...")
        print(f"📊 Final inventory before upgrades: +1:{level_counts[1]} +2:{level_counts[2]} +3:{level_counts[3]} +4:{level_counts[4]} +5:{level_counts[5]} +6:{level_counts[6]} +7:{level_counts[7]}")
        print(f"📦 Empty slots: {empty_count}")
        
        session_stats = {
            'total_attempts': 0,
            'successful_upgrades': 0,
            'failed_upgrades': 0,
            'restock_cycles': 0,
            'plus_seven_achieved': 0,
            'session_start_time': time.time()
        }
        
        max_cycles = 50  # Prevent infinite loops
        cycle_count = 0
        
        while cycle_count < max_cycles:
            cycle_count += 1
            print(f"\n🔄 UPGRADE CYCLE {cycle_count}/{max_cycles}")
            
            # Check success condition first
            success_achieved, plus_seven_count = self.check_plus_seven_success_condition(inventory_data, locked_slots)
            
            if success_achieved:
                print("\n🎉 SUCCESS CONDITION ACHIEVED!")
                print("🏦 Proceeding to bank storage...")
                
                bank_success = self.navigate_to_bank_and_store_plus_seven_items(nav_system, inventory_data, locked_slots)
                
                if bank_success:
                    session_stats['plus_seven_achieved'] += plus_seven_count
                    print("\n💎 +7 ITEMS BANKED SUCCESSFULLY!")
                    print("🏠 Player is now at town - ending session to let town system make decision")
                    # After bank storage, return to let town system take over
                    return inventory_data, session_stats, False
                else:
                    print("⚠️ Bank storage failed, but +7 items achieved!")
                    print("🔄 CONTINUING UPGRADE SESSION...")
                    # Continue anyway
            
            # Analyze current inventory composition
            level_counts, empty_count = self.count_items_by_level(inventory_data, locked_slots)
            current_phase = self.determine_upgrade_phase(level_counts)
            restock_threshold = self.calculate_restock_threshold(current_phase)
            
            print(f"\n📊 CYCLE {cycle_count} ANALYSIS:")
            print(f"   Current phase: {current_phase}")
            print(f"   Item levels: +1:{level_counts[1]} +2:{level_counts[2]} +3:{level_counts[3]} +4:{level_counts[4]} +5:{level_counts[5]} +6:{level_counts[6]} +7:{level_counts[7]}")
            print(f"   Empty slots: {empty_count}")
            print(f"   Restock threshold: {restock_threshold}")
            
            # Check if we need to store +7 items in bank (before continuing upgrades)
            # NOTE: This is now primarily handled earlier, but kept as a safety check during upgrade cycles
            if self.should_store_plus_seven_items(inventory_data, locked_slots, threshold=6):
                print("\n🏦 BANK STORAGE TRIGGERED DURING UPGRADE CYCLE!")
                print("💎 Going to store +7 items in bank for safekeeping...")
                
                bank_success, updated_inventory = self.navigate_to_bank_and_store_plus_seven_items(nav_system, inventory_data, locked_slots)
                
                if bank_success:
                    print("✅ Bank storage completed successfully!")
                    print("🏠 Player is now at town - will restart normal flow")
                    # Use the updated inventory data from the bank storage process
                    inventory_data = updated_inventory
                    # After bank storage, restart the entire process from town
                    return self.advanced_upgrade_system_to_plus_seven(anvil_rects, inventory_data, locked_slots, nav_system, model)
                else:
                    print("⚠️ Bank storage failed, but continuing with upgrades...")
                    # Still use updated inventory if partial success occurred
                    inventory_data = updated_inventory
                    level_counts, empty_count = self.count_items_by_level(inventory_data, locked_slots)
            
            # RESTOCKING IS NOW HANDLED AT THE BEGINNING - SKIP THIS LOGIC
            # (Inventory has already been restocked if needed)
            
            # Determine what upgrades to perform and execute them
            upgrade_performed = False
            
            # Priority 1: Upgrade +1 items to +3 (safest)
            if level_counts[1] > 0:
                print(f"\n🌱 Upgrading {level_counts[1]} items: +1 → +3")
                inventory_data, upgrade_result = self.upgrade_all_items_to_target_level(
                    anvil_rects, inventory_data, locked_slots, 
                    target_from_level=1, target_to_level=3
                )
                session_stats['total_attempts'] += upgrade_result.get('total_attempts', 0)
                session_stats['successful_upgrades'] += upgrade_result.get('successful_upgrades', 0)
                session_stats['failed_upgrades'] += upgrade_result.get('failed_upgrades', 0)
                upgrade_performed = True
                
            # Priority 2: Upgrade +2 items to +3
            elif level_counts[2] > 0:
                print(f"\n🌿 Upgrading {level_counts[2]} items: +2 → +3")
                inventory_data, upgrade_result = self.upgrade_all_items_to_target_level(
                    anvil_rects, inventory_data, locked_slots,
                    target_from_level=2, target_to_level=3
                )
                session_stats['total_attempts'] += upgrade_result.get('total_attempts', 0)
                session_stats['successful_upgrades'] += upgrade_result.get('successful_upgrades', 0)
                session_stats['failed_upgrades'] += upgrade_result.get('failed_upgrades', 0)
                upgrade_performed = True
                
            # Priority 3: Upgrade +3 items to +6 (pipeline building)
            elif level_counts[3] > 0:
                print(f"\n⚡ Upgrading {level_counts[3]} items: +3 → +6 (pipeline building)")
                inventory_data, upgrade_result = self.upgrade_all_items_to_target_level(
                    anvil_rects, inventory_data, locked_slots,
                    target_from_level=3, target_to_level=6
                )
                session_stats['total_attempts'] += upgrade_result.get('total_attempts', 0)
                session_stats['successful_upgrades'] += upgrade_result.get('successful_upgrades', 0)
                session_stats['failed_upgrades'] += upgrade_result.get('failed_upgrades', 0)
                upgrade_performed = True
                
            # Priority 4: Upgrade +4 items to +6
            elif level_counts[4] > 0:
                print(f"\n💪 Upgrading {level_counts[4]} items: +4 → +6")
                inventory_data, upgrade_result = self.upgrade_all_items_to_target_level(
                    anvil_rects, inventory_data, locked_slots,
                    target_from_level=4, target_to_level=6
                )
                session_stats['total_attempts'] += upgrade_result.get('total_attempts', 0)
                session_stats['successful_upgrades'] += upgrade_result.get('successful_upgrades', 0)
                session_stats['failed_upgrades'] += upgrade_result.get('failed_upgrades', 0)
                upgrade_performed = True
                
            # Priority 5: Upgrade +5 items to +6
            elif level_counts[5] > 0:
                print(f"\n🔥 Upgrading {level_counts[5]} items: +5 → +6")
                inventory_data, upgrade_result = self.upgrade_all_items_to_target_level(
                    anvil_rects, inventory_data, locked_slots,
                    target_from_level=5, target_to_level=6
                )
                session_stats['total_attempts'] += upgrade_result.get('total_attempts', 0)
                session_stats['successful_upgrades'] += upgrade_result.get('successful_upgrades', 0)
                session_stats['failed_upgrades'] += upgrade_result.get('failed_upgrades', 0)
                upgrade_performed = True
                
            # Priority 6: Upgrade +6 items to +7 (final goal)
            elif level_counts[6] > 0:
                batch_size = self.calculate_batch_size_for_plus_seven(level_counts[6])
                if batch_size > 0:
                    print(f"\n🏆 Attempting +6 → +7 on {min(batch_size, level_counts[6])} items (batch processing)")
                    inventory_data, upgrade_result = self.upgrade_all_items_to_target_level(
                        anvil_rects, inventory_data, locked_slots,
                        target_from_level=6, target_to_level=7
                    )
                    session_stats['total_attempts'] += upgrade_result.get('total_attempts', 0)
                    session_stats['successful_upgrades'] += upgrade_result.get('successful_upgrades', 0)
                    session_stats['failed_upgrades'] += upgrade_result.get('failed_upgrades', 0)
                    upgrade_performed = True
                else:
                    print(f"\n🏆 Only {level_counts[6]} +6 items available - need 15+ for batch processing")
                    print("🏠 Going to town to make decision...")
                    town_success = self.town_()
                    
                    if town_success:
                        print("✅ At town - ending current session for decision-making")
                        break  # Exit the upgrade cycle to let town system take over
                    else:
                        print("❌ Failed to reach town, continuing with individual upgrades...")
                        inventory_data, upgrade_result = self.upgrade_all_items_to_target_level(
                            anvil_rects, inventory_data, locked_slots,
                            target_from_level=6, target_to_level=7
                        )
                        session_stats['total_attempts'] += upgrade_result.get('total_attempts', 0)
                        session_stats['successful_upgrades'] += upgrade_result.get('successful_upgrades', 0)
                        session_stats['failed_upgrades'] += upgrade_result.get('failed_upgrades', 0)
                        upgrade_performed = True
                    
            if not upgrade_performed:
                print("\n⏸️ No upgrades to perform in current state")
                if level_counts[7] < 10:  # Haven't reached goal yet
                    # Check if we need to go to town (high empty slot count or no upgradeable items)
                    total_upgradeable_items = sum(level_counts[i] for i in range(1, 7))  # +1 to +6 items
                    
                    if empty_count >= restock_threshold or total_upgradeable_items == 0:
                        print("🏠 Going to town to check inventory and make decision...")
                        town_success = self.town_()
                        
                        if town_success:
                            print("✅ At town - ending current session for decision-making")
                            break  # Exit the upgrade cycle to let town system take over
                        else:
                            print("❌ Failed to reach town")
                            time.sleep(2)
                    else:
                        print("🔄 May need different strategy...")
                        time.sleep(2)  # Pause before next cycle
                    
            # Safety break
            if cycle_count >= max_cycles:
                print(f"\n⏰ Maximum cycles ({max_cycles}) reached")
                break
                
            time.sleep(1)  # Brief pause between cycles
        
        # Session ended without achieving goal
        print(f"\n📊 SESSION ENDED AFTER {cycle_count} CYCLES")
        print(f"   Final +7 count: {plus_seven_count}/10")
        return inventory_data, session_stats, False

    def quit_game(self):
        self.click_(903, 334, clicks=2)
        time.sleep(1)
        pydirectinput.press('h')
        time.sleep(1)
        self.click_(891, 383, clicks=2)
        time.sleep(1)
        self.click_(521, 402, clicks=2)

    def build_inventory_data_structure(self, predictions, template_values, locked_slots):
        """
        Build complete inventory data structure from ML predictions and template values
        
        Parameters:
        - predictions: ML predictions for item types
        - template_values: Template matching results for upgrade levels  
        - locked_slots: Set of locked slot indices
        
        Returns:
        - dict: Complete inventory data structure
        """
        inventory_data = {}
        
        # Item type mapping
        item_type_map = {
            0: "Upgrade Scroll",
            1: "Pauldron", 
            2: "Pant",
            3: "Boot",
            4: "Hat",
            5: "Glove",
            6: "Empty"
        }
        
        for slot_idx, prediction in enumerate(predictions):
            template_value = template_values.get(slot_idx, 0)
            
            # Handle ML prediction values (item types)
            if slot_idx in locked_slots:
                item_type = "LOCKED"
                if prediction == 0:
                    item_status = "Locked slot - Upgrade Scroll"
                else:
                    item_status = f"Locked slot - {item_type_map.get(prediction, 'Unknown')}"
            elif prediction == 6:
                item_type = "Empty"
                item_status = "Empty slot"
            elif prediction in item_type_map:
                item_type = item_type_map[prediction]
                item_status = f"{item_type}"
            else:
                item_type = f"UNKNOWN({prediction})"
                item_status = f"Unknown item type: {prediction}"
            
            # Handle template values (upgrade levels)
            if template_value > 0:
                upgrade_level = f"+{template_value}"
                template_status = f"Upgrade level: {upgrade_level}"
                # Combine item type with upgrade level
                if item_type not in ["Empty", "LOCKED"] and prediction != 6:
                    final_item_status = f"{item_type} {upgrade_level}"
                else:
                    final_item_status = item_status
            else:
                upgrade_level = "No upgrade"
                template_status = "No upgrade detected"
                final_item_status = item_status
            
            inventory_data[slot_idx] = {
                'prediction': prediction,
                'item_type': item_type,
                'status': final_item_status,
                'template_value': template_value,
                'upgrade_level': upgrade_level,
                'template_status': template_status
            }
        
        return inventory_data

    def execute_bank_storage_with_retry(self, nav_system, inventory_data, locked_slots, max_attempts=3):
        """
        Execute bank storage with retry mechanism and fallback to town
        """
        print(f"🏦 EXECUTING BANK STORAGE WITH RETRY (max {max_attempts} attempts)")
        
        for attempt in range(max_attempts):
            print(f"\n🔄 BANK STORAGE ATTEMPT {attempt + 1}/{max_attempts}")
            
            try:
                # Try the direct bank storage function
                bank_success = self.navigate_to_bank_and_store_plus_seven_items(nav_system, inventory_data, locked_slots)
                
                if bank_success:
                    print(f"✅ Bank storage successful on attempt {attempt + 1}!")
                    return True
                else:
                    print(f"❌ Bank storage attempt {attempt + 1} failed")
                    
                    if attempt < max_attempts - 1:  # Not the last attempt
                        print("🏠 Going back to town for retry...")
                        town_success = self.town_()
                        if town_success:
                            print("✅ Returned to town successfully")
                            time.sleep(2)
                        else:
                            print("❌ Failed to return to town!")
                    
            except Exception as e:
                print(f"❌ Bank storage attempt {attempt + 1} failed with error: {e}")
                if attempt < max_attempts - 1:
                    print("🏠 Going back to town for retry...")
                    town_success = self.town_()
                    if town_success:
                        print("✅ Returned to town successfully")
                        time.sleep(2)
        
        print(f"❌ Bank storage failed after {max_attempts} attempts")
        return False

    def execute_restock_with_retry(self, nav_system, inventory_data, locked_slots, empty_count, max_attempts=3):
        """
        Execute restocking with retry mechanism and fallback to town
        """
        print(f"🛒 EXECUTING RESTOCK WITH RETRY (max {max_attempts} attempts)")
        
        for attempt in range(max_attempts):
            print(f"\n🔄 RESTOCK ATTEMPT {attempt + 1}/{max_attempts}")
            
            try:
                # Try the restock function
                restock_success, purchased_items = self.execute_restock_cycle(nav_system, inventory_data, locked_slots, empty_count)
                
                if restock_success:
                    print(f"✅ Restock successful on attempt {attempt + 1}!")
                    
                    # Organize inventory after successful restock using the proper function
                    inventory_top_left = (673, 460)
                    inventory_bottom_right = (715, 502)
                    inventory_rects = self.draw_rectangle_grid(None, inventory_top_left, inventory_bottom_right, rows=4, cols=7, spacing=7)
                    
                    # Use the organize_inventory_after_restock function (no locked_slots needed)
                    organize_success = self.organize_inventory_after_restock(None, inventory_rects, set())
                    
                    if organize_success:
                        print("✅ Inventory organized successfully!")
                    else:
                        print("⚠️ Inventory organization failed, but restock was successful")
                    
                    return True, purchased_items
                else:
                    print(f"❌ Restock attempt {attempt + 1} failed")
                    
                    if attempt < max_attempts - 1:  # Not the last attempt
                        print("🏠 Going back to town for retry...")
                        town_success = self.town_()
                        if town_success:
                            print("✅ Returned to town successfully")
                            time.sleep(2)
                        else:
                            print("❌ Failed to return to town!")
                    
            except Exception as e:
                print(f"❌ Restock attempt {attempt + 1} failed with error: {e}")
                if attempt < max_attempts - 1:
                    print("🏠 Going back to town for retry...")
                    town_success = self.town_()
                    if town_success:
                        print("✅ Returned to town successfully")
                        time.sleep(2)
        
        print(f"❌ Restock failed after {max_attempts} attempts")
        return False, []

def startup_town_based_system(controller, nav_system, model, locked_slots):
    """
    New startup system that begins at town, checks inventory, and routes to appropriate action
    """
    print("🏠 STARTING TOWN-BASED DECISION SYSTEM...")
    print("🎯 Will check inventory and route to: Bank Storage / Restocking / Upgrades")
    print("=" * 70)
    
    max_startup_attempts = 3
    
    for attempt in range(max_startup_attempts):
        print(f"\n🔄 STARTUP ATTEMPT {attempt + 1}/{max_startup_attempts}")
        
        # Step 1: Ensure we're at town
        print("\n🏠 STEP 1: NAVIGATING TO TOWN...")
        town_success = controller.town_()
        if not town_success:
            print(f"❌ Failed to reach town on attempt {attempt + 1}")
            if attempt == max_startup_attempts - 1:
                print("❌ CRITICAL: Cannot reach town after all attempts!")
                return False
            continue
        
        print("✅ Successfully at town!")
        time.sleep(2)
        
        # Step 2: Scan inventory from town (using inventory rects)
        print("\n📋 STEP 2: SCANNING INVENTORY AT TOWN...")
        
        # Define inventory rectangles for town-based scanning
        inventory_top_left = (670, 456)
        inventory_bottom_right = (707, 492)
        inventory_rects = controller.draw_rectangle_grid(None, inventory_top_left, inventory_bottom_right, rows=4, cols=7, spacing=12)
        
        # Scan inventory directly (already at town)
        print("🎒 Opening inventory...")
        inventory_opened = controller.open_inventory()
        if not inventory_opened:
            print(f"❌ Failed to open inventory on attempt {attempt + 1}")
            if attempt == max_startup_attempts - 1:
                print("❌ CRITICAL: Cannot open inventory after all attempts!")
                return False
            continue
        
        time.sleep(1)
        
        # Capture screenshot and get ML predictions
        print("🧠 Getting ML predictions...")
        screenshot = controller.capture_game_window()
        
        predictions = controller.predict_and_draw_rectangles(
            screenshot, model, inventory_rects, locked_slots, 
            rect_name="Inventory", rect_color=(0, 255, 0)
        )
        
        if predictions is None:
            print(f"❌ Failed to get ML predictions on attempt {attempt + 1}")
            if attempt == max_startup_attempts - 1:
                print("❌ CRITICAL: Cannot get ML predictions after all attempts!")
                return False
            continue
        
        # Display ML predictions grid
        print("\n🧠 ML PREDICTIONS GRID (Item Types):")
        print("┌" + "─" * 29 + "┐")
        for row in range(4):
            row_display = "│"
            for col in range(7):
                slot_idx = row * 7 + col
                if slot_idx < len(predictions):
                    pred_val = predictions[slot_idx]
                    row_display += f" {pred_val:2d} │"
                else:
                    row_display += "    │"
            print(row_display)
            if row < 3:
                print("├" + "─" * 29 + "┤")
        print("└" + "─" * 29 + "┘")
        print("Legend: 0=Scroll 1=Pauldron 2=Pant 3=Boot 4=Hat 5=Glove 6=Empty")
        
        # Get template values using navigate_rectangles (optimized with ML predictions)
        print("\n🎯 Scanning for upgrade levels...")
        template_values = controller.navigate_rectangles(inventory_rects, click=False, wait_time=0.3, ml_predictions=predictions)
        
        # Build inventory data structure
        inventory_data = controller.build_inventory_data_structure(predictions, template_values, locked_slots)
        
        if inventory_data is None:
            print(f"❌ Failed to scan inventory on attempt {attempt + 1}")
            if attempt == max_startup_attempts - 1:
                print("❌ CRITICAL: Cannot scan inventory after all attempts!")
                return False
            continue
        
        print("✅ Inventory scanned successfully!")
        
        # Close inventory after scanning
        print("🎒 Closing inventory...")
        controller.close_inventory()
        time.sleep(1)
        
        # Step 3: Analyze inventory and make decision
        print("\n🧠 STEP 3: ANALYZING INVENTORY AND MAKING DECISION...")
        level_counts, empty_count = controller.count_items_by_level(inventory_data, locked_slots)
        
        print(f"📊 INVENTORY STATUS:")
        print(f"   +1:{level_counts[1]} +2:{level_counts[2]} +3:{level_counts[3]} +4:{level_counts[4]} +5:{level_counts[5]} +6:{level_counts[6]} +7:{level_counts[7]}")
        print(f"   Empty slots: {empty_count}")
        
        # Decision logic: Bank Storage > Restocking > Upgrades
        if level_counts[7] >= 6:
            print(f"\n🏦 DECISION: BANK STORAGE ({level_counts[7]} +7 items ≥ 6)")
            print("🎯 Route: Town → Anvil → Armor → Inn → Bank")
            
            bank_success = controller.execute_bank_storage_with_retry(nav_system, inventory_data, locked_slots, max_attempts=3)
            if bank_success:
                print("✅ Bank storage completed! Restarting from town...")
                continue  # Restart the decision process
            else:
                print("❌ Bank storage failed after all attempts")
                if attempt == max_startup_attempts - 1:
                    return False
                continue
                
        elif empty_count >= 10:
            print(f"\n🛒 DECISION: RESTOCKING ({empty_count} empty slots ≥ 10)")
            print("🎯 Route: Town → Anvil → Armor → Buy Items")
            
            restock_success, purchased_items = controller.execute_restock_with_retry(nav_system, inventory_data, locked_slots, empty_count, max_attempts=3)
            if restock_success:
                print("✅ Restocking completed! Restarting from town...")
                continue  # Restart the decision process
            else:
                print("❌ Restocking failed after all attempts")
                if attempt == max_startup_attempts - 1:
                    return False
                continue
                
        else:
            print(f"\n🔧 DECISION: START UPGRADES")
            print(f"   +7 items: {level_counts[7]} (< 6), Empty slots: {empty_count} (< 10)")
            print("🎯 Route: Town → Anvil → Begin Upgrade System")
            
            # Navigate to anvil to start upgrades
            print("\n🔨 NAVIGATING TO ANVIL FOR UPGRADES...")
            anvil_success = nav_system.navigate_with_agent("town_to_anvil", max_steps=200, distance_threshold=4)
            
            if anvil_success:
                print("✅ Successfully reached anvil!")
                
                # Interact with anvil
                anvil_interaction_success = controller.execute_anvil_interaction_with_retry(nav_system, max_attempts=3)
                
                if anvil_interaction_success:
                    print("✅ Anvil opened successfully!")
                    
                    # Define anvil rectangles for upgrade system
                    anvil_top_left = (652, 439)
                    anvil_bottom_right = (695, 481)
                    anvil_rects = controller.draw_rectangle_grid(None, anvil_top_left, anvil_bottom_right, rows=4, cols=7, spacing=7)
                    
                    # Start the upgrade system
                    print("\n🚀 STARTING ADVANCED UPGRADE SYSTEM...")
                    return controller.advanced_upgrade_system_to_plus_seven(anvil_rects, inventory_data, locked_slots, nav_system, model)
                else:
                    print("❌ Failed to open anvil")
                    if attempt == max_startup_attempts - 1:
                        return False
                    continue
            else:
                print("❌ Failed to reach anvil")
                if attempt == max_startup_attempts - 1:
                    return False
                continue
    
    print("❌ STARTUP FAILED: All attempts exhausted")
    return False

def main():
    try:
        controller = GameController("Knight Online Client")
        time.sleep(2)
        
        print("🚀 STARTING TOWN-BASED UPGRADE SYSTEM")
        print("🏠 New Flow: Town → Check Inventory → Route Decision → Execute Action")
        print("=" * 70)
        
        # Load the ML model for inventory scanning
        try:
            from tensorflow.keras.models import load_model
            model = load_model('inventory_model.h5')
            print("✅ ML model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load ML model: {e}")
            return
        
        # Load navigation system
        try:
            from multi_agent_test import MultiAgentNavigationSystem
            nav_system = MultiAgentNavigationSystem()
            print("✅ Navigation system loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load navigation system: {e}")
            return
        
        # Define locked slots (slot 0 has upgrade scroll)
        locked_slots = set([50])
        
        # Execute the new town-based startup system
        result = startup_town_based_system(controller, nav_system, model, locked_slots)
        
        # Handle result from the startup system
        if result:
            if isinstance(result, tuple) and len(result) == 3:
                # Result from advanced_upgrade_system_to_plus_seven
                updated_inventory_data, session_stats, success_achieved = result
                
                # Display final session results
                print("\n" + "="*60)
                print("📊 TOWN-BASED UPGRADE SESSION RESULTS:")
                print("="*60)
                
                if success_achieved:
                    print("🎉 SESSION STATUS: SUCCESS!")
                    print(f"✅ Achieved {session_stats.get('plus_seven_achieved', 0)} items at +7 level")
                    print("🏦 All +7 items have been stored in bank for safekeeping")
                else:
                    print("⚠️ SESSION STATUS: INCOMPLETE")
                    print("📈 Session ended before achieving 10+ +7 items")
                
                print(f"\n📈 SESSION STATISTICS:")
                print(f"   Total upgrade attempts: {session_stats.get('total_attempts', 0)}")
                print(f"   Successful upgrades: {session_stats.get('successful_upgrades', 0)}")
                print(f"   Failed upgrades: {session_stats.get('failed_upgrades', 0)}")
                print(f"   Restock cycles: {session_stats.get('restock_cycles', 0)}")
                
                if session_stats.get('total_attempts', 0) > 0:
                    success_rate = (session_stats.get('successful_upgrades', 0) / session_stats.get('total_attempts', 1)) * 100
                    print(f"   Overall success rate: {success_rate:.1f}%")
                
                session_duration = time.time() - session_stats.get('session_start_time', time.time())
                print(f"   Session duration: {session_duration/60:.1f} minutes")
                
                print("="*60)
                
                if success_achieved:
                    print("\n🏆 CONGRATULATIONS! +7 COLLECTION MISSION ACCOMPLISHED!")
                else:
                    print("\n🔄 Session can be restarted to continue progress toward 10+ +7 items")
            else:
                print("✅ TOWN-BASED SYSTEM COMPLETED SUCCESSFULLY!")
        else:
            print("❌ TOWN-BASED SYSTEM FAILED!")
            print("🔄 System was unable to complete the requested operation")
        
    except Exception as e:
        print(f"Hata: {e}")
        print("Oyun penceresi bulunamadı veya başka bir sorun var")

if __name__ == "__main__":
    main()