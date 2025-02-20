# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 04:36:33 2024

@author: ismt
"""
import cv2
import numpy as np
import win32gui
import win32con
import pydirectinput
from PIL import ImageGrab
import time

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

    def move_(self, x, y, duration=0.1, relative=False):
        x = max(0, min(x, self.game_width))
        y = max(0, min(y, self.game_height))
        
        offset_x = 0
        offset_y = -25
        
        target_x = self.window_x + x - offset_x
        target_y = self.window_y + y - offset_y
        
        if relative:
            current_x, current_y = pydirectinput.position()
            target_x = current_x + x
            target_y = current_y + y
            
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
            
    def press_(self, key, hold_duration=0.1):
        pydirectinput.keyDown(key)
        print("pressed key:", key)
        time.sleep(hold_duration)
        pydirectinput.keyUp(key)

    def capture_game_window(self):
        bbox = (self.window_x, self.window_y, self.window_right, self.window_bottom)
        screenshot = ImageGrab.grab(bbox=bbox)
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    def find_template(self, template_path, confidence=0.65):
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise Exception(f"Template image not found at path: {template_path}")
        
        screenshot = self.capture_game_window()
        gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        result = cv2.matchTemplate(gray_screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= confidence:
            return max_loc
        return None

    def town_(self):
        pydirectinput.press('h')
        time.sleep(0.1)
        self.click_(903, 334, clicks=2)
        pydirectinput.press('h')
