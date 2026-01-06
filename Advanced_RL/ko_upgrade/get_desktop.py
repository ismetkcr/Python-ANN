import win32gui, win32ui, win32con
import numpy as np
import cv2 as cv
from time import time
import ctypes

class DesktopCapture:
    def __init__(self):
        """
        Tüm ekranları yakalar (multi-monitor setup için)
        """
        # Windows API fonksiyonları
        user32 = ctypes.windll.user32
        
        # Tüm ekranları yakala (sanal ekran)
        self.x = user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
        self.y = user32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN  
        self.w = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
        self.h = user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN
        
        print(f"Tüm ekranlar yakalanıyor: {self.w}x{self.h} pozisyon: ({self.x}, {self.y})")
        
        # Fare konumu (görüntü koordinatları)
        self.mouse_x, self.mouse_y = 0, 0
        # Gerçek masaüstü koordinatları
        self.real_mouse_x, self.real_mouse_y = 0, 0
        # Scale faktörü
        self.scale_factor = 1.0
        
        # Pencereyi tanımla ve fare geri çağrısını ayarla
        cv.namedWindow('Desktop Capture')
        cv.setMouseCallback('Desktop Capture', self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_MOUSEMOVE:
            # Görüntü koordinatları (küçültülmüş)
            self.mouse_x, self.mouse_y = x, y
            
            # Gerçek masaüstü koordinatları (scale ile çarpılmış)
            self.real_mouse_x = int(x / self.scale_factor)
            self.real_mouse_y = int(y / self.scale_factor)

    def get_real_coordinates(self, display_x, display_y):
        """
        OpenCV penceresindeki koordinatları gerçek masaüstü koordinatlarına çevir
        """
        real_x = int(display_x / self.scale_factor)
        real_y = int(display_y / self.scale_factor)
        return real_x, real_y

    def get_screenshot(self):
        """Masaüstü görüntüsü yakala"""
        try:
            # Desktop DC'sini al
            desktop = win32gui.GetDesktopWindow()
            desktop_dc = win32gui.GetWindowDC(desktop)
            img_dc = win32ui.CreateDCFromHandle(desktop_dc)
            mem_dc = img_dc.CreateCompatibleDC()
            
            # Bitmap oluştur
            screenshot = win32ui.CreateBitmap()
            screenshot.CreateCompatibleBitmap(img_dc, self.w, self.h)
            mem_dc.SelectObject(screenshot)
            
            # Ekran bölgesini kopyala
            mem_dc.BitBlt((0, 0), (self.w, self.h), img_dc, (self.x, self.y), win32con.SRCCOPY)
            
            # OpenCV formatına çevir
            bmpinfo = screenshot.GetInfo()
            bmpstr = screenshot.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype='uint8')
            img.shape = (self.h, self.w, 4)
            
            # Kaynakları temizle
            mem_dc.DeleteDC()
            img_dc.DeleteDC()
            win32gui.ReleaseDC(desktop, desktop_dc)
            win32gui.DeleteObject(screenshot.GetHandle())
            
            # Alpha kanalını kaldır ve BGR'den RGB'ye çevir
            img = img[..., :3]
            img = np.ascontiguousarray(img)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            
            return img
            
        except Exception as e:
            print(f"Screenshot hatası: {e}")
            return None

def main():
    try:
        # Direkt tüm ekranları yakala
        desktop_cap = DesktopCapture()
        print(f"Masaüstü yakalanıyor! 'q' ile çık, 's' ile screenshot kaydet.")
        
        loop_time = time()
        frame_count = 0
        
        while True:
            # Masaüstü görüntüsünü al
            screenshot = desktop_cap.get_screenshot()
            
            if screenshot is None:
                print("Görüntü yakalanamadı!")
                break
            
            frame_count += 1
            
            # Görüntüyü göster (büyükse küçült)
            display_img = screenshot
            if screenshot.shape[1] > 1920:  # Çok büyükse küçült
                scale = 1920 / screenshot.shape[1]
                new_width = int(screenshot.shape[1] * scale)
                new_height = int(screenshot.shape[0] * scale)
                display_img = cv.resize(screenshot, (new_width, new_height))
                desktop_cap.scale_factor = scale  # Scale faktörünü kaydet
            else:
                desktop_cap.scale_factor = 1.0  # Scale yok
            
            # Bilgileri ekrana yaz
            mouse_text = f'Display Mouse: ({desktop_cap.mouse_x}, {desktop_cap.mouse_y})'
            cv.putText(display_img, mouse_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            real_mouse_text = f'Real Desktop: ({desktop_cap.real_mouse_x}, {desktop_cap.real_mouse_y})'
            cv.putText(display_img, real_mouse_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            fps_text = f'FPS: {1 / (time() - loop_time):.2f}'
            cv.putText(display_img, fps_text, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            frame_text = f'Frame: {frame_count}'
            cv.putText(display_img, frame_text, (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            scale_text = f'Scale: {desktop_cap.scale_factor:.2f}'
            cv.putText(display_img, scale_text, (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            loop_time = time()
            
            cv.imshow('Desktop Capture', display_img)
            
            # Tuş kontrolü
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Screenshot kaydet
                filename = f'desktop_screenshot_{frame_count}.png'
                cv.imwrite(filename, cv.cvtColor(screenshot, cv.COLOR_RGB2BGR))
                print(f"Screenshot kaydedildi: {filename}")
            elif key == ord('c'):
                # Mevcut gerçek koordinatları yazdır
                print(f"Mevcut pozisyon - Display: ({desktop_cap.mouse_x}, {desktop_cap.mouse_y})")
                print(f"Mevcut pozisyon - Real: ({desktop_cap.real_mouse_x}, {desktop_cap.real_mouse_y})")
        
        cv.destroyAllWindows()
        
    except Exception as e:
        print(f"Hata: {e}")

# Gerçek koordinatları almak için yardımcı fonksiyonlar
def get_real_coordinates_from_capture(desktop_capture, display_x, display_y):
    """
    DesktopCapture'dan alınan display koordinatları gerçek koordinatlara çevir
    """
    return desktop_capture.get_real_coordinates(display_x, display_y)

# Basit kullanım örneği
def coordinate_test():
    """
    Koordinat sistemini test et
    """
    desktop_cap = DesktopCapture()
    
    print("Koordinat testi başlıyor...")
    print("Mouse'u hareket ettir ve 'c' tuşuna bas")
    print("'q' ile çık")
    
    while True:
        screenshot = desktop_cap.get_screenshot()
        if screenshot is None:
            break
            
        # Scale faktörünü hesapla
        if screenshot.shape[1] > 1920:
            scale = 1920 / screenshot.shape[1]
            new_width = int(screenshot.shape[1] * scale)
            new_height = int(screenshot.shape[0] * scale)
            display_img = cv.resize(screenshot, (new_width, new_height))
            desktop_cap.scale_factor = scale
        else:
            display_img = screenshot
            desktop_cap.scale_factor = 1.0
        
        cv.imshow('Coordinate Test', display_img)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print(f"Display koordinat: ({desktop_cap.mouse_x}, {desktop_cap.mouse_y})")
            print(f"Gerçek koordinat: ({desktop_cap.real_mouse_x}, {desktop_cap.real_mouse_y})")
            print("---")
    
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
    # coordinate_test()  # Test için bu satırı uncomment et