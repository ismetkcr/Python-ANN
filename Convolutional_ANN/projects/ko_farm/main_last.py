import numpy as np
from KnightOnlineEnv_class_last import KnightOnline
import cv2
import keyboard
import time  # Import time module to manage sleep intervals

def toggle_pause():
    global pause
    pause = not pause
    if pause:
        print("Program paused. Press 'p' to continue.")
    else:
        print("Program continuing...")

keyboard.add_hotkey('p', toggle_pause)

if __name__ == '__main__':
    env = KnightOnline()
    pause = False
    cv2.namedWindow('Computer Vision', cv2.WINDOW_NORMAL)
    while True:  # Programın sürekli çalışmasını sağlayan dış döngü
        try:
            done = False         
            screenshot = env.reset()
            states = env.get_states(screenshot)
            
            mp_history = []
            while not done:  # Her bölümün ana döngüsü
                while pause:
                    time.sleep(0.05)  # Prevents busy-waiting while paused and allows interrupt detection
                
                env.update_fps()
                action = env.action_decider(states)
                screenshot, done = env.step(action)
                states = env.get_states(screenshot)
                
                # 50 adımdan sonra MP takibine başla
                if env.episode_step_counter < 100:
                    mp_history.append(states['character_hp'])
                    
                    # Son 25 adımı kontrol et
                    if len(mp_history) > 25:
                        mp_history.pop(0)  # İlk elemanı çıkar
                        
                        # Tüm MP değerleri aynı ise done=True yap
                        if len(set(mp_history)) == 1:
                            done = True
                
                # Basit text gösterimi (sağ alt köşe)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8  # Punto büyütüldü
                
                # FPS güncellemesi
                env.update_fps()
                fps_text = f"FPS: {env.fps:.1f}"
                
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
                    f"Monster Flag: {'Active' if states['monster_flag'] else 'Not Found'}",
                    f"Monster HP: {states['monster_hp']:.1f}%",
                    f"Char HP: {states['character_hp']:.1f}%",
                    f"Char MP: {states['character_mp']:.1f}%"
                ]
                
                # Sağ alt metinler
                y_offset = screenshot.shape[0] - 120  # Alt kenardan başlama noktası
                for text in texts:
                    text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
                    x_pos = screenshot.shape[1] - text_size[0] - 20  # Sağ kenardan 20 pixel içeride
                    cv2.putText(screenshot, text, (x_pos, y_offset), font, font_scale, (0, 255, 0), 2)  # Yeşil renk (0,255,0)
                    y_offset += 30  # Satır aralığı artırıldı
                
                # Display the screenshot
                cv2.imshow('Computer Vision', screenshot)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt  # Raises KeyboardInterrupt if 'q' is pressed
            
        except KeyboardInterrupt:
            print("\nExiting program...")
            break  # Sonsuz döngüden çık
        finally:
            cv2.destroyAllWindows()
            keyboard.unhook_all()