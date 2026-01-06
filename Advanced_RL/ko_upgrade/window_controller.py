import cv2
import numpy as np
import pydirectinput
from PIL import ImageGrab
import time
import ctypes
import pyperclip

class DesktopController:
    def __init__(self, monitor_type="all"):
        """
        monitor_type: "all" = tüm ekranlar, "primary" = ana ekran
        """
        # Windows API fonksiyonları
        user32 = ctypes.windll.user32
        
        if monitor_type == "all":
            # Tüm ekranları kontrol et
            self.x = user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
            self.y = user32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN  
            self.width = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
            self.height = user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN
            print(f"Tüm ekranlar kontrol ediliyor: {self.width}x{self.height} başlangıç: ({self.x}, {self.y})")
        else:
            # Sadece ana ekran
            self.x = 0
            self.y = 0
            self.width = user32.GetSystemMetrics(0)   # SM_CXSCREEN
            self.height = user32.GetSystemMetrics(1)   # SM_CYSCREEN
            print(f"Ana ekran kontrol ediliyor: {self.width}x{self.height}")
        
        # PyDirectInput ayarları
        pydirectinput.FAILSAFE = True
        pydirectinput.PAUSE = 0.01  # Komutlar arası küçük gecikme

    def convert_capture_to_screen_coords(self, capture_x, capture_y):
        """
        DesktopCapture koordinatlarını gerçek ekran koordinatlarına çevir
        capture_x, capture_y: DesktopCapture'dan gelen koordinatlar (görüntü koordinatları)
        Returns: gerçek ekran koordinatları
        """
        # DesktopCapture görüntüsü (0,0)'dan başlar ama gerçekte sanal ekranın (self.x, self.y)'sinden başlar
        screen_x = self.x + capture_x
        screen_y = self.y + capture_y
        
        print(f"Koordinat çevirme: Capture({capture_x}, {capture_y}) -> Screen({screen_x}, {screen_y})")
        return screen_x, screen_y

    def move_(self, x, y, duration=0.1, relative=False, from_capture=False):
        """
        Mouse'u belirli bir pozisyona hareket ettir
        x, y: Hedef koordinatlar
        duration: Hareket süresi
        relative: True ise mevcut pozisyondan relatif hareket
        from_capture: True ise koordinatlar DesktopCapture'dan geliyor
        """
        if from_capture:
            # DesktopCapture koordinatlarını gerçek ekran koordinatlarına çevir
            target_x, target_y = self.convert_capture_to_screen_coords(x, y)
        elif relative:
            # Mevcut pozisyondan relatif hareket
            current_x, current_y = pydirectinput.position()
            target_x = current_x + x
            target_y = current_y + y
            
            direction_x = "sağa" if x > 0 else "sola" if x < 0 else "yatay hareket yok"
            direction_y = "aşağı" if y > 0 else "yukarı" if y < 0 else "dikey hareket yok"
            
            print(f"{direction_x} {abs(x)} piksel, {direction_y} {abs(y)} piksel hareket")
            print(f"({current_x}, {current_y}) -> ({target_x}, {target_y})")
        else:
            # Mutlak koordinatlara hareket (eski davranış)
            target_x = self.x + x  # Sanal ekran offseti
            target_y = self.y + y
            
            # Sınırları kontrol et
            target_x = max(self.x, min(target_x, self.x + self.width - 1))
            target_y = max(self.y, min(target_y, self.y + self.height - 1))
            
            print(f"Mouse masaüstü koordinatlarına hareket: ({x}, {y}) -> Ekran: ({target_x}, {target_y})")
        
        # Mouse'u hareket ettir
        pydirectinput.moveTo(target_x, target_y, duration=duration)
        return target_x, target_y

    def click_(self, x, y, button='left', clicks=1, interval=0.1, from_capture=False):
        """
        Belirli bir koordinata tıkla
        x, y: Tıklanacak koordinatlar
        button: 'left', 'right', 'middle'
        clicks: Tıklama sayısı
        interval: Çoklu tıklamalar arası gecikme
        from_capture: True ise koordinatlar DesktopCapture'dan geliyor
        """
        # Önce pozisyona git
        self.move_(x, y, from_capture=from_capture)
        time.sleep(0.1)  # Mouse'un oturması için bekle
        
        # Tıklama işlemini gerçekleştir
        for i in range(clicks):
            pydirectinput.mouseDown(button=button)
            time.sleep(0.05)  # Kısa basış süresi
            pydirectinput.mouseUp(button=button)
            current_pos = pydirectinput.position()
            print(f"Tıklama {i+1}/{clicks} - Pozisyon: {current_pos}")
            if clicks > 1:
                time.sleep(interval)

    def click_capture_coords(self, capture_x, capture_y, button='left', clicks=2):
        """
        DesktopCapture koordinatlarını kullanarak tıkla (kısayol fonksiyon)
        """
        return self.click_(capture_x, capture_y, button=button, clicks=clicks, from_capture=True)

    def drag_(self, start_x, start_y, end_x, end_y, button='left', steps=20, delay=0.05):
        """
        Bir noktadan diğerine sürükle
        start_x, start_y: Başlangıç koordinatları
        end_x, end_y: Bitiş koordinatları
        button: Hangi mouse tuşu ile sürüklenecek
        steps: Kaç adımda sürüklenecek (pürüzsüzlük için)
        delay: Adımlar arası gecikme
        """
        # Başlangıç pozisyonuna git
        self.move_(start_x, start_y)
        time.sleep(0.1)
        
        try:
            # Mouse tuşunu bas ve tut
            pydirectinput.mouseDown(button=button)
            time.sleep(0.1)
            
            # Adım adım hedefe doğru hareket et
            for i in range(steps + 1):
                # Linear interpolation
                current_x = start_x + (end_x - start_x) * (i / steps)
                current_y = start_y + (end_y - start_y) * (i / steps)
                
                # Hedefe git
                target_x = self.x + current_x
                target_y = self.y + current_y
                pydirectinput.moveTo(target_x, target_y, duration=0)
                
                time.sleep(delay)
                
            print(f"Sürükleme tamamlandı: ({start_x}, {start_y}) -> ({end_x}, {end_y})")
            
        finally:
            # Mouse tuşunu bırak
            pydirectinput.mouseUp(button=button)

    def press_(self, key, hold_duration=0.1):
        """
        Klavye tuşuna bas
        key: Basılacak tuş ('w', 'a', 's', 'd', 'enter', 'space', vb.)
        hold_duration: Tuşu ne kadar süre basılı tut
        """
        print(f"Tuşa basılıyor: {key}")
        pydirectinput.keyDown(key)
        time.sleep(hold_duration)
        pydirectinput.keyUp(key)

    def type_(self, text, interval=0.05):
        """
        Metin yaz
        text: Yazılacak metin
        interval: Karakterler arası gecikme
        """
        print(f"Metin yazılıyor: '{text}'")
        pydirectinput.write(text, interval=interval)
        
    def hotkey_(self, *keys):
        """
        Birden fazla tuşa aynı anda bas (kombinasyon tuşları)
        Örnek: desktop.hotkey_('shift', '6') -> &
        """
        print(f"Hotkey basılıyor: {' + '.join(keys)}")
        
        # Tüm tuşları sırayla bas
        for key in keys:
            pydirectinput.keyDown(key)
            time.sleep(0.01)
        
        # Tüm tuşları tersten bırak
        for key in reversed(keys):
            pydirectinput.keyUp(key)
            time.sleep(0.01)


    def capture_desktop(self):
        """
        Tüm masaüstünün screenshot'ını al
        """
        import ctypes
        from PIL import ImageGrab
        
        # Tüm ekranları yakala
        user32 = ctypes.windll.user32
        x = user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
        y = user32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN  
        width = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
        height = user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN
        
        bbox = (x, y, x + width, y + height)
        screenshot = ImageGrab.grab(bbox=bbox)
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    def find_desktop_template(self, template_path, confidence=0.8):
        """
        Masaüstünde template matching yap
        template_path: Aranacak görüntünün yolu
        confidence: Eşleşme güven seviyesi
        Returns: (x, y, confidence) veya None
        """
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Template görüntüsü bulunamadı: {template_path}")
            return None
        
        screenshot = self.capture_desktop()
        gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        result = cv2.matchTemplate(gray_screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= confidence:
            # Template'in merkezini hesapla
            template_height, template_width = template.shape
            center_x = max_loc[0] + template_width // 2
            center_y = max_loc[1] + template_height // 2
            
            print(f"Desktop Template bulundu! Konum: ({center_x}, {center_y}), Güven: {max_val:.2f}")
            return center_x, center_y, max_val
        else:
            print(f"Desktop Template bulunamadı: {template_path} (En yüksek güven: {max_val:.2f})")
            return None

    def click_desktop_template(self, template_path, confidence=0.8, button='left', clicks=1):
        """
        Desktop'ta template'i bul ve tıkla
        """
        result = self.find_desktop_template(template_path, confidence)
        if result:
            center_x, center_y, conf = result
            
            # Desktop koordinatlarına çevir (sanal ekran offset'i ekle)
            import ctypes
            user32 = ctypes.windll.user32
            offset_x = user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
            offset_y = user32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN
            
            screen_x = offset_x + center_x
            screen_y = offset_y + center_y
            
            # Direkt screen koordinatlarına tıkla
            pydirectinput.moveTo(screen_x, screen_y)
            time.sleep(0.1)
            
            for i in range(clicks):
                pydirectinput.mouseDown(button=button)
                time.sleep(0.05)
                pydirectinput.mouseUp(button=button)
                if clicks > 1:
                    time.sleep(0.1)
            
            print(f"Desktop Template'e tıklandı: ({screen_x}, {screen_y})")
            return True
        else:
            print(f"Desktop Template bulunamadı, tıklama yapılamadı: {template_path}")
            return False

    def is_desktop_image_visible(self, template_path, confidence=0.8):
        """
        Desktop'ta belirli bir image görünür mü kontrol et
        Returns: True/False
        """
        result = self.find_desktop_template(template_path, confidence)
        return result is not None
    
    def wait_for_desktop_image(self, template_path, timeout=30, confidence=0.8):
        """
        Desktop'ta belirli bir image görünene kadar bekle
        """
        print(f"Desktop'ta image bekleniyor: {template_path}")
        
        for i in range(timeout):
            if self.is_desktop_image_visible(template_path, confidence):
                print(f"Desktop image bulundu! ({i+1} saniye sonra)")
                return True
            
            print(f"Bekleniyor... {timeout-i} saniye kaldı")
            time.sleep(1)
        
        print(f"Timeout! Desktop image bulunamadı: {template_path}")
        return False

def check_desktop_applications(self):
    """
    Desktop'taki uygulamaları kontrol et
    """
    # Image path'lerini tanımla
    anaconda_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_upgrade/desktop_images/anaconda_prompt.png"
    knight_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_upgrade/desktop_images/knight_online.png"
    edge_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_upgrade/desktop_images/microsoft_edge.png"
    
    results = {
        "anaconda_visible": self.is_desktop_image_visible(anaconda_path),
        "knight_visible": self.is_desktop_image_visible(knight_path),
        "edge_visible": self.is_desktop_image_visible(edge_path)
    }
    
    print(f"Desktop App Durumu: {results}")
    return results

    def manage_desktop_applications(self):
        """
        Desktop uygulamalarını yönet
        """
        # Image path'lerini tanımla
        anaconda_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_upgrade/desktop_images/anaconda_prompt.png"
        knight_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_upgrade/desktop_images/knight_online.png"
        edge_path = "c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_upgrade/desktop_images/microsoft_edge.png"
        
        # Edge varsa kapat
        if self.is_desktop_image_visible(edge_path):
            print("Microsoft Edge tespit edildi, kapatılıyor...")
            if self.click_desktop_template(edge_path):
                time.sleep(0.5)
                pydirectinput.hotkey('alt', 'f4')  # Alt+F4 ile kapat
        
        # Knight Online varsa tıkla
        if self.is_desktop_image_visible(knight_path):
            print("Knight Online tespit edildi, tıklanıyor...")
            self.click_desktop_template(knight_path)
            return "knight_clicked"
        
        # Anaconda Prompt varsa tıkla
        if self.is_desktop_image_visible(anaconda_path):
            print("Anaconda Prompt tespit edildi, tıklanıyor...")
            self.click_desktop_template(anaconda_path)
            return "anaconda_clicked"
        
        return "nothing_found"

    def get_mouse_position(self):
        """
        Mevcut mouse pozisyonunu al
        """
        pos = pydirectinput.position()
        # Sanal ekran koordinatlarına çevir
        desktop_x = pos[0] - self.x
        desktop_y = pos[1] - self.y
        return desktop_x, desktop_y
    
    def knight_online_login(self, username="BIYAX01", password="Tmmozmn16&"):
        """
        Knight Online'a login yapar
        username: Kullanıcı adı
        password: Şifre
        """
        print("Knight Online login başlatılıyor...")
        
        # İlk tıklama - launcher'da 2 kere tıkla
        self.click_capture_coords(2792, 820, clicks=2)
        time.sleep(3)
        
        # Start butonuna tıkla
        self.click_(2902, 726, clicks=1)
        
        # 45 saniye geri sayım (oyun yüklenirken)
        print("Oyun yükleniyor...")
        for i in range(45, 0, -1):
            print(f"Kalan süre: {i}")
            time.sleep(1)
        
        # Login ekranında tıklama
        self.click_(2444, 322)
        time.sleep(3)
        self.press_("enter")
        
        # Kullanıcı adı girme
        print(f"Kullanıcı adı giriliyor: {username}")
        self.press_("capslock")  # Caps Lock aç
        time.sleep(0.1)
        self.type_(username.lower())  # Küçük harf olarak gir
        self.press_("capslock")  # Caps Lock kapat
        
        # Şifre alanına geç
        self.press_("tab")
        time.sleep(1)
        
        # Şifre girme
        print("Şifre giriliyor...")
        for char in password:
            if char.isupper():
                self.press_("capslock")
                self.press_(char.lower())
                self.press_("capslock")
            elif char == "&":
                self.hotkey_('shift', '6')
            else:
                self.type_(char)
            time.sleep(0.1)
        
        # Login işlemini tamamla
        self.press_("enter")
        time.sleep(1)
        self.press_("enter")
        time.sleep(3)
        
        # Karakter seçim ekranı
        self.click_(2390, 188)
        time.sleep(0.5)
        self.click_(2612, 236, clicks=2)
        for i in range(45, 0, -1):
            print(f"Kalan süre: {i}")
            time.sleep(1) #<sımdılık elle girildiği için bekleme var
        
        # Oyuna giriş
        self.press_("enter")
        self.press_("enter")
        self.press_("enter")
        for i in range(45, 0, -1):
            print(f"Kalan süre: {i}")
            time.sleep(1) #oyun dolma ekranı bekleme süresi.

        
        print("Knight Online login tamamlandı!")
    
    def is_knight_online_open(self):
        """
        Knight Online açık mı kontrol et
        Returns: True/False
        """
        import win32gui
        
        def enum_handler(hwnd, results):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if "knight online" in title.lower():
                    results.append(True)
        
        results = []
        win32gui.EnumWindows(enum_handler, results)
        return len(results) > 0
    
    def start_upgrade_bot_first_time(self):
        """
        İlk kez bot başlatır (tam setup)
        """
        print("Upgrade bot başlatılıyor (İlk Açılış)...")
        time.sleep(5)
        
        # Terminal/Komut satırını aç
        self.click_(674, 1060)
        time.sleep(0.5)
        
        # "ana" yazarak arama
        self.type_("ana")
        time.sleep(1)
        
        # Sağ tıklama (context menu)
        self.click_(734, 440, button="right")
        time.sleep(1)
        
        # Sol tıklama (terminal aç)
        self.click_(822, 460, button="left")
        time.sleep(3)
        
        # Conda environment'ı aktive et
        pyperclip.copy("conda activate spyder6_env")
        self.hotkey_('ctrl', 'v')
        time.sleep(0.5)
        self.press_("enter")
        time.sleep(2)
        
        # Proje klasörüne git
        pyperclip.copy("cd c:/users/ismt/desktop/python-ann/knightonlinefarmagentproject/ko_upgrade")
        self.hotkey_('ctrl', 'v')
        self.press_("enter")
        time.sleep(1)
        
        # Bot scriptini çalıştır
        pyperclip.copy("python game_controller.py")
        self.hotkey_('ctrl', 'v')
        self.press_("enter")
        
        # Son tıklama (pencere minimize vs)
        self.click_(2818, 48, button="left")
        
        print("Upgrade bot başlatıldı (İlk Açılış)!")

    def start_upgrade_bot_existing(self):
        """
        Mevcut anaconda prompt'a tıklayarak bot başlatır
        """
        print("Upgrade bot başlatılıyor (Mevcut Prompt)...")
        time.sleep(2)
        
        # Mevcut anaconda prompt'a tıkla
        self.click_(3012, 480)
        time.sleep(1)
        
        # Sadece programı çalıştır
        pyperclip.copy("python game_controller.py")
        self.hotkey_('ctrl', 'v')
        self.press_("enter")
        
        print("Upgrade bot başlatıldı (Mevcut Prompt)!")
    
    def auto_knight_online_manager(self, check_interval=60):
        print("Knight Online Otomatik Yönetici başlatılıyor...")
        print(f"Her {check_interval} saniyede bir kontrol edilecek")
        print("Durdurmak için Ctrl+C basın")
        
        bot_running = False
        first_time = True  # İlk açılış mı?
        
        try:
            while True:
                print(f"\n--- {time.strftime('%H:%M:%S')} - Durum Kontrolü ---")
                
                if self.is_knight_online_open():
                    print("✓ Knight Online açık")
                    
                    if not bot_running:
                        if first_time:
                            print("→ Upgrade bot başlatılıyor (İlk Açılış)...")
                            self.start_upgrade_bot_first_time()
                            first_time = False
                        else:
                            print("→ Upgrade bot başlatılıyor (Mevcut Prompt)...")
                            self.start_upgrade_bot_existing()
                        
                        bot_running = True
                        print("✓ Bot başlatıldı")
                    else:
                        print("✓ Bot zaten çalışıyor")
                else:
                    print("✗ Knight Online kapalı")
                    
                    if bot_running:
                        print("→ Bot durduruldu (oyun kapandı)")
                        bot_running = False
                    
                    print("→ Oyuna giriş yapılıyor...")
                    self.knight_online_login()
                    print("✓ Oyuna giriş tamamlandı")
                    bot_running = False
                
                print(f"→ {check_interval} saniye bekleniyor...")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Otomatik yönetici durduruldu (Ctrl+C)")
        except Exception as e:
            print(f"\n❌ Hata oluştu: {e}")
            print("5 saniye sonra tekrar denenecek...")
            time.sleep(5)


def demo_desktop_control():
    desktop = DesktopController("all")
    
    # Sürekli çalışan sistem başlat
    desktop.auto_knight_online_manager(check_interval=20)  # 60 saniyede bir kontrol


if __name__ == "__main__":
    # Demo'yu çalıştır
    demo_desktop_control()