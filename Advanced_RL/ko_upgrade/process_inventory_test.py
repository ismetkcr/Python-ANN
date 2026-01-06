# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 16:10:21 2025

@author: ismt
"""

import win32gui, win32ui, win32con
import numpy as np
import cv2 as cv
from time import time
from tensorflow.keras.models import load_model
from get_window import WindowCapture
import win32api
from game_controller import GameController
import time
from multi_agent_test import MultiAgentNavigationSystem

# Initialize components
controller = GameController('Knight Online Client')
wincap = WindowCapture('Knight Online Client')
model = load_model('inventory_model.h5')
loop_time = time.time()

# Initialize Multi-Agent Navigation System
nav_system = MultiAgentNavigationSystem()
print("🤖 Multi-Agent Navigation System initialized")

# Navigation control variables
navigation_executed = False
navigation_cycle_count = 0

# Coordinates for the various grids
inventory_top_left = (670, 456)
inventory_bottom_right = (707, 492)

bank_inventory_top_left = (654, 430)
bank_inventory_bottom_right = (691, 467)

bank_top_left = (672, 162)
bank_bottom_right = (712, 203)

anvil_top_left = (652, 439)
anvil_bottom_right = (695, 481)

loc_top_left = (44, 105)
loc_bottom_right = (163, 122)

sundires_top_left = (667,270)
sundires_bottom_right = (712,312)



# Define rectangles grids once
inventory_rects = controller.draw_rectangle_grid(None, inventory_top_left, inventory_bottom_right, rows=4, cols=7, spacing=12)
bank_rects = controller.draw_rectangle_grid(None, bank_top_left, bank_bottom_right, rows=4, cols=6, spacing=12)
bank_inventory_rects = controller.draw_rectangle_grid(None, bank_inventory_top_left, bank_inventory_bottom_right, rows=4, cols=7, spacing=13)
anvil_rects = controller.draw_rectangle_grid(None, anvil_top_left, anvil_bottom_right, rows=4, cols=7, spacing=7)
sundires_rects = controller.draw_rectangle_grid(None, sundires_top_left, sundires_bottom_right, rows=1, cols=5, spacing=7)
loc_rect = controller.draw_rectangle_grid(None, loc_top_left, loc_bottom_right, rows=1, cols=1, spacing=0)

# Define locked slots (first 1 slots and any additional ones you want to lock)
locked_slots = set([0])  # İlk 1 slot kilitli
k = 1

# Initial wait
time.sleep(3)

# ========== STARTUP SEQUENCE ==========
print("🚀 Executing startup sequence...")
#inventory_data = controller.startup_inventory_scan(model, inventory_rects, locked_slots)

# if inventory_data is None:
#     print("❌ Startup sequence failed!")
#     exit()

print("✅ Startup sequence completed! Inventory data ready.")

# ========== ACTION DECISION SYSTEM ==========
def check_inventory_full(inventory_data, locked_slots):
    """Check if inventory is full (excluding locked slots)"""
    total_slots = 28
    usable_slots = total_slots - len(locked_slots)
    
    # Count non-empty slots (excluding locked slots)
    occupied_slots = 0
    for slot_idx, data in inventory_data.items():
        if slot_idx not in locked_slots and data['prediction'] != 6:  # 6 = Empty
            occupied_slots += 1
    
    is_full = occupied_slots >= usable_slots
    
    print(f"\n📊 INVENTORY STATUS:")
    print(f"   Total slots: {total_slots}")
    print(f"   Locked slots: {len(locked_slots)} {list(locked_slots)}")
    print(f"   Usable slots: {usable_slots}")
    print(f"   Occupied slots: {occupied_slots}")
    print(f"   Empty slots: {usable_slots - occupied_slots}")
    print(f"   Inventory full: {'YES' if is_full else 'NO'}")
    
    return is_full

def execute_go_to_armor_from_town_sequence():
    """Execute the sequence: town → anvil → armor using trained agents with restart logic"""
    max_sequence_attempts = 3
    
    for sequence_attempt in range(max_sequence_attempts):
        print(f"\n🎯 EXECUTING GO TO ARMOR SEQUENCE - Attempt {sequence_attempt + 1}/{max_sequence_attempts}")
        print("=" * 60)
        
        # Step 1: town → anvil
        print("🏠➡️🔨 Step 1: Going from town to anvil...")
        town_to_anvil_success = nav_system.navigate_with_agent("town_to_anvil", max_steps=200, distance_threshold=8)
        
        if town_to_anvil_success:
            print("✅ Successfully reached anvil!")
            time.sleep(2)
            
            # Step 2: anvil → armor shop
            print("\n🔨➡️🛡️ Step 2: Going from anvil to armor shop...")
            anvil_to_armor_success = nav_system.navigate_with_agent("anvil_to_armor", max_steps=200, distance_threshold=4)
            
            if anvil_to_armor_success:
                print("✅ Successfully reached armor shop!")
                
                # Step 3: Find the correct NPC (Hesta) - single attempt
                print("\n🔍 Step 3: Finding correct NPC at armor shop...")
                npc_found = controller.find_correct_npc_at_armor()
                
                if npc_found:
                    print("✅ Successfully found Hesta NPC!")
                    
                    # Step 4: Open armor NPC
                    print("\n🛡️ Step 4: Opening armor shop...")
                    armor_opened = controller.open_npc("armor")
                    
                    if armor_opened:
                        print("✅ Armor shop opened successfully!")
                        print("🛒 Ready to buy armor items!")
                        return True
                    else:
                        print("❌ Failed to open armor shop")
                        print("🔄 Going back to town to restart sequence...")
                        
                        # Go back to town before retrying
                        town_success = controller.town_()
                        if town_success:
                            print("✅ Returned to town, will retry sequence...")
                            continue  # Retry the entire sequence
                        else:
                            print("❌ Failed to return to town!")
                            break
                else:
                    print("❌ Failed to find Hesta NPC")
                    print("🔄 Going back to town to restart sequence...")
                    
                    # Go back to town before retrying
                    town_success = controller.town_()
                    if town_success:
                        print("✅ Returned to town, will retry sequence...")
                        continue  # Retry the entire sequence
                    else:
                        print("❌ Failed to return to town!")
                        break
            else:
                print("❌ Failed to reach armor shop from anvil")
                print("🔄 Going back to town to restart sequence...")
                
                # Go back to town before retrying
                town_success = controller.town_()
                if town_success:
                    print("✅ Returned to town, will retry sequence...")
                    continue  # Retry the entire sequence
                else:
                    print("❌ Failed to return to town!")
                    break
        else:
            print("❌ Failed to reach anvil from town")
            print("🔄 Will retry sequence...")
            continue  # Retry the entire sequence
    
    print(f"❌ Failed to complete armor sequence after {max_sequence_attempts} attempts")
    return False

# ========== UPGRADE SESSION WORKFLOW ==========
print("\n🔧 STARTING UPGRADE SESSION WORKFLOW...")

# Check if inventory is full and take appropriate action
#is_inventory_full = check_inventory_full(inventory_data, locked_slots)

# if not is_inventory_full:
#     print("\n🛒 PHASE 1: INVENTORY NOT FULL - FILLING EMPTY SLOTS")
    
#     # Load required navigation agents for armor sequence
#     print("🤖 Loading navigation agents for armor sequence...")
#     town_to_anvil_agent = nav_system.load_agent("town_to_anvil")
#     anvil_to_armor_agent = nav_system.load_agent("anvil_to_armor")
#     armor_to_anvil_agent = nav_system.load_agent("armor_to_anvil")
    
#     if town_to_anvil_agent and anvil_to_armor_agent and armor_to_anvil_agent:
#         print("✅ Navigation agents loaded successfully!")
        
#         # Execute the armor sequence to fill empty slots
#         armor_sequence_success = execute_go_to_armor_from_town_sequence()
        
#         if armor_sequence_success:
#             print("\n🎉 ARMOR SEQUENCE COMPLETED SUCCESSFULLY!")
#             print("🛡️ Ready to buy armor items...")
            
#             # ========== ITEM PURCHASING SYSTEM ==========
#             print("\n🛒 STARTING ITEM PURCHASING SYSTEM...")
            
#             # Count current inventory items
#             item_counts, empty_slots = controller.count_inventory_items(inventory_data, locked_slots)
            
#             # Calculate which items to buy to fill ALL empty slots
#             purchase_order = controller.calculate_items_to_buy(item_counts, empty_slots)
            
#             if purchase_order:
#                 print(f"\n🎯 Executing purchase order for {len(purchase_order)} items...")
#                 print(f"🎯 Goal: Fill ALL {empty_slots} empty slots!")
                
#                 # Execute the purchases
#                 purchase_success, purchased_items = controller.buy_items_from_sundires(sundires_rects, purchase_order)
                
#                 if purchase_success:
#                     print("\n✅ ALL ITEM PURCHASES COMPLETED SUCCESSFULLY!")
                    
#                     # Update inventory data with new purchases
#                     if purchased_items:
#                         print("\n📦 UPDATING INVENTORY DATA WITH NEW PURCHASES...")
#                         inventory_data = controller.update_inventory_with_purchases(inventory_data, purchased_items, locked_slots)
#                         print("✅ Inventory data updated successfully!")
                    
#                     # ========== PHASE 2: GO TO ANVIL FOR UPGRADES ==========
#                     print("\n🔧 PHASE 2: INVENTORY FULL - GOING TO ANVIL FOR UPGRADES")
#                     print("🗺️ Using armor_to_anvil navigation...")
                    
#                     anvil_success = nav_system.navigate_with_agent("armor_to_anvil", max_steps=200, distance_threshold=3)
                    
#                     if anvil_success:
#                         print("✅ Successfully reached anvil from armor shop!")
                        
#                         # ========== ANVIL INTERACTION ==========
#                         print("\n🔧 STARTING ANVIL INTERACTION...")
#                         anvil_interaction_success = controller.execute_anvil_interaction_with_retry(nav_system, max_attempts=3)
                        
#                         if anvil_interaction_success:
#                             print("✅ Anvil opened successfully! Ready for upgrade operations!")
#                         else:
#                             print("❌ Failed to open anvil after multiple attempts")
#                             print("🔄 Will retry in next cycle...")
#                     else:
#                         print("❌ Failed to reach anvil from armor shop")
#                         print("🔄 Will retry in next cycle...")
                    
#                 else:
#                     print("\n⚠️ SOME ITEM PURCHASES FAILED!")
                    
#                     # Still update inventory with any successful purchases
#                     if purchased_items:
#                         print("\n📦 UPDATING INVENTORY DATA WITH PARTIAL PURCHASES...")
#                         inventory_data = controller.update_inventory_with_purchases(inventory_data, purchased_items, locked_slots)
#                         print("✅ Inventory data updated with partial purchases!")
#             else:
#                 print("\n❌ No items need to be purchased")
                
#                 # Even if no purchases, still go to anvil for upgrades
#                 print("\n🔧 PHASE 2: GOING TO ANVIL FOR UPGRADES")
#                 print("🗺️ Using armor_to_anvil navigation...")
                
#                 anvil_success = nav_system.navigate_with_agent("armor_to_anvil", max_steps=200, distance_threshold=3)
                
#                 if anvil_success:
#                     print("✅ Successfully reached anvil from armor shop!")
                    
#                     # ========== ANVIL INTERACTION ==========
#                     print("\n🔧 STARTING ANVIL INTERACTION...")
#                     anvil_interaction_success = controller.execute_anvil_interaction_with_retry(nav_system, max_attempts=3)
                    
#                     if anvil_interaction_success:
#                         print("✅ Anvil opened successfully! Ready for upgrade operations!")
#                     else:
#                         print("❌ Failed to open anvil after multiple attempts")
#                         print("🔄 Will retry in next cycle...")
#                 else:
#                     print("❌ Failed to reach anvil from armor shop")
            
#         else:
#             print("\n❌ ARMOR SEQUENCE FAILED!")
#             print("🔄 Will retry in next cycle...")
#     else:
#         print("❌ Failed to load required navigation agents!")
#         print("   Required: town_to_anvil, anvil_to_armor, armor_to_anvil")

# else:
#     print("\n🔧 PHASE 1: INVENTORY IS FULL - GOING DIRECTLY TO ANVIL")
#     print("🗺️ Using town_to_anvil navigation...")
    
#     # Load town_to_anvil agent for direct anvil access
#     town_to_anvil_agent = nav_system.load_agent("town_to_anvil")
    
#     if town_to_anvil_agent:
#         print("✅ Town to anvil agent loaded successfully!")
        
#         anvil_success = nav_system.navigate_with_agent("town_to_anvil", max_steps=200, distance_threshold=3)
        
#         if anvil_success:
#             print("✅ Successfully reached anvil from town!")
            
#             # ========== ANVIL INTERACTION ==========
#             print("\n🔧 STARTING ANVIL INTERACTION...")
#             anvil_interaction_success = controller.execute_anvil_interaction_with_retry(nav_system, max_attempts=3)
            
#             if anvil_interaction_success:
#                 print("✅ Anvil opened successfully! Ready for upgrade operations!")
#             else:
#                 print("❌ Failed to open anvil after multiple attempts")
#                 print("🔄 Will retry in next cycle...")
#         else:
#             print("❌ Failed to reach anvil from town")
#             print("🔄 Will retry in next cycle...")
#     else:
#         print("❌ Failed to load town_to_anvil agent!")

# print("\n🎮 Starting main game loop...")

# # Main loop variables  
# inventory_scanned = True  # Already scanned during startup
# scan_cycle_count = 0
# action_executed = True  # Flag to track if action has been executed

while True:
    screenshot = wincap.get_screenshot()
    
    # RENK DÜZELTMESİ: RGB'den BGR'ye çevir çünkü cv.imshow BGR bekliyor
    if screenshot is not None:
        screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)
        
        # OCR ile konum okuma
        result = controller.read_location_with_ocr(screenshot, loc_rect)
        #print(f"📍 OCR sonucu: {result}")
        
        # ========== INVENTORY DATA IS NOW AVAILABLE ==========
        # The inventory_data dictionary is already populated from startup_inventory_scan()
        # You can access it here for any additional processing or operations
        
        # # ========== AUTOMATIC NAVIGATION EXECUTION ==========
        # # Execute navigation from town to anvil automatically (once per session)
        # navigation_cycle_count += 1
        # if not navigation_executed and navigation_cycle_count > 5:  # Wait 5 cycles before starting
        #     print("🎯 Auto-executing navigation from town to anvil...")
        #     try:
        #         # Load the town_to_anvil agent
        #         agent = nav_system.load_agent("anvil_to_armor")
        #         if agent:
        #             print("✅ Agent loaded, starting automatic navigation...")
        #             navigation_executed = True  # Set flag to prevent re-execution
        #             success = nav_system.navigate_with_agent("anvil_to_armor", max_steps=200, distance_threshold=4)
        #             if success:
        #                 print("✅ Automatic navigation successful! Reached anvil!")
        #             else:
        #                 print("❌ Automatic navigation failed")
        #         else:
        #             print("❌ Failed to load town_to_anvil agent")
        #             navigation_executed = True  # Prevent retry
        #     except Exception as e:
        #         print(f"❌ Automatic navigation error: {e}")
        #         navigation_executed = True  # Prevent retry
        # ======================================================
    
    # ============= RECTANGLE GRUPLARINI AÇIP KAPATABİLİRSİNİZ =============
    
    # INVENTORY RECTANGLES (Yeşil renk) - ML Prediction - Show continuously
    # if inventory_scanned:  # Only show after initial scan is complete
    #     controller.predict_and_draw_rectangles(
    #         screenshot, model, inventory_rects, locked_slots, 
    #         rect_name="Inventory", rect_color=(0, 255, 0)
    #     )
    
    
    
    # if k == 1:
    #     time.sleep(2)
    #     controller.navigate_rectangles(inventory_rects)
    #     k += 1
    
    #ANVIL RECTANGLES (Yeşil renk) - ML Prediction
    # controller.predict_and_draw_rectangles(
    #     screenshot, model, anvil_rects, locked_slots, 
    #     rect_name="Anvil", rect_color=(0, 255, 0)
    # )
    
    # SUNDIRES RECTANGLES (Yeşil renk) - ML Prediction
    # controller.predict_and_draw_rectangles(
    #     screenshot, model, sundires_rects, locked_slots, 
    #     rect_name="Sundires", rect_color=(0, 255, 0)
    # )
    
    # BANK RECTANGLES (Mavi renk) - ML Prediction
    # controller.predict_and_draw_rectangles(
    #     screenshot, model, bank_rects, locked_slots, 
    #     rect_name="Bank", rect_color=(255, 0, 0)
    # )
    
    # BANK INVENTORY RECTANGLES (Sarı renk) - ML Prediction
    # controller.predict_and_draw_rectangles(
    #     screenshot, model, bank_inventory_rects, locked_slots, 
    #     rect_name="Bank_Inventory", rect_color=(0, 255, 255)
    # )
    
    # ======================================================================
    
    # Display mouse position (WindowCapture koordinatlarını kullan)
    mouse_text = f'Mouse: ({wincap.mouse_x}, {wincap.mouse_y})'
    cv.putText(screenshot, mouse_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # FPS göster
    fps = 1 / (time.time() - loop_time)
    fps_text = f'FPS: {fps:.2f}'
    cv.putText(screenshot, fps_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    loop_time = time.time()

    # Show window and calculate FPS
    cv.imshow('Computer Vision', screenshot)

    # Key controls
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        cv.destroyAllWindows()
        break
    elif key == ord('c'):
        # Mouse pozisyonunu konsola yazdır
        print(f"Current mouse position: ({wincap.mouse_x}, {wincap.mouse_y})")
    elif key == ord('t'):
        # Test click
        try:
            controller.click_(wincap.mouse_x, wincap.mouse_y + 25)  # offset düzeltmesi
            print(f"Test click at: ({wincap.mouse_x}, {wincap.mouse_y + 25})")
        except Exception as e:
            print(f"Click error: {e}")
    elif key == ord('1'):
        print("Inventory rectangles activated")
    elif key == ord('2'):
        print("Bank rectangles activated")
    elif key == ord('3'):
        print("Bank inventory rectangles activated")
    elif key == ord('4'):
        print("Location OCR activated")