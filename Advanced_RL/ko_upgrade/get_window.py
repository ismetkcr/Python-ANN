import win32gui, win32ui, win32con
import numpy as np
import cv2 as cv
from time import time
import ctypes
from ctypes import wintypes

class WindowCapture:
    def __init__(self, window_name):
        # Pencere adı ile handle'ı bul
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))
        
        # Pencere boyutunu al
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]
        
        # Pencere kenar ve başlık boyutlarını hesaba kat
        border_pixels = 0
        titlebar_pixels = 0
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels
        
        # Ofset değerleri, ekran pozisyonlarıyla eşleşir
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y
        
        # Fare konumu
        self.mouse_x, self.mouse_y = 0, 0
        self.last_printed_x, self.last_printed_y = -1, -1  # Son yazdırılan pozisyon
        
        # Pencereyi tanımla ve fare geri çağrısını ayarla
        cv.namedWindow('Computer Vision')
        cv.setMouseCallback('Computer Vision', self.mouse_callback)
        
        print(f"Pencere '{window_name}' yakalandı!")
        print(f"Pencere boyutu: {self.w}x{self.h}")
        print("Mouse hareket ettirin, pozisyon konsola yazılacak...")
        print("'c' tuşuna basarak mevcut pozisyonu manuel yazdırabilirsiniz")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_MOUSEMOVE:
            self.mouse_x, self.mouse_y = x, y
            
            # Pozisyon değiştiğinde konsola yazdır (her 5 piksel değişimde)
            if (abs(x - self.last_printed_x) > 5 or abs(y - self.last_printed_y) > 5):
                print(f"Mouse pozisyonu: ({x}, {y})")
                self.last_printed_x, self.last_printed_y = x, y
        
        elif event == cv.EVENT_LBUTTONDOWN:
            print(f"Sol tık: ({x}, {y})")
        
        elif event == cv.EVENT_RBUTTONDOWN:
            print(f"Sağ tık: ({x}, {y})")

    def get_screenshot(self):
        try:
            # İlk olarak PrintWindow dene
            screenshot = self.get_screenshot_printwindow()
            if screenshot is not None:
                return screenshot
            
            # PrintWindow başarısız olursa normal BitBlt dene
            return self.get_screenshot_bitblt()
            
        except Exception as e:
            print(f"Screenshot hatası: {e}")
            return None

    def get_screenshot_printwindow(self):
        """PrintWindow kullanarak görüntü yakala"""
        try:
            # Pencere DC'sini al
            wDC = win32gui.GetWindowDC(self.hwnd)
            dcObj = win32ui.CreateDCFromHandle(wDC)
            cDC = dcObj.CreateCompatibleDC()
            
            # Bitmap oluştur
            dataBitMap = win32ui.CreateBitmap()
            dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
            cDC.SelectObject(dataBitMap)
            
            # PrintWindow API'sini kullan
            user32 = ctypes.windll.user32
            user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
            user32.PrintWindow.restype = wintypes.BOOL
            
            # PrintWindow ile yakala
            result = user32.PrintWindow(self.hwnd, cDC.GetSafeHdc(), 2)  # 2 = PW_RENDERFULLCONTENT
            
            if not result:
                return None
            
            # Ham veriyi OpenCV formatına dönüştür
            signedIntsArray = dataBitMap.GetBitmapBits(True)
            img = np.frombuffer(signedIntsArray, dtype='uint8')
            
            if len(img) == 0:
                return None
                
            img.shape = (self.h, self.w, 4)
            
            # Kaynakları serbest bırak
            dcObj.DeleteDC()
            cDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, wDC)
            win32gui.DeleteObject(dataBitMap.GetHandle())
            
            # Alpha kanalını kaldır
            img = img[..., :3]
            img = np.ascontiguousarray(img)
            
            # BGR'den RGB'ye çevir
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            
            return img
            
        except:
            return None

    def get_screenshot_bitblt(self):
        """Orijinal BitBlt metodu"""
        try:
            # Pencere görüntüsünü al
            wDC = win32gui.GetWindowDC(self.hwnd)
            dcObj = win32ui.CreateDCFromHandle(wDC)
            cDC = dcObj.CreateCompatibleDC()
            dataBitMap = win32ui.CreateBitmap()
            dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
            cDC.SelectObject(dataBitMap)
            
            result = cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)
            
            if not result:
                return None
            
            # Ham veriyi OpenCV formatına dönüştür
            signedIntsArray = dataBitMap.GetBitmapBits(True)
            img = np.frombuffer(signedIntsArray, dtype='uint8')
            img.shape = (self.h, self.w, 4)
            
            # Kaynakları serbest bırak
            dcObj.DeleteDC()
            cDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, wDC)
            win32gui.DeleteObject(dataBitMap.GetHandle())
            
            # Alpha kanalını kaldır
            img = img[..., :3]
            img = np.ascontiguousarray(img)
            
            return img
            
        except:
            return None

# def main():
#     try:
#         # Sınıf örneğini oluştur
#         wincap = WindowCapture('Knight Online Client')
        
#         loop_time = time()
#         frame_count = 0
        
#         while True:
#             # Güncel ekran görüntüsünü al
#             screenshot = wincap.get_screenshot()
            
#             if screenshot is None:
#                 print("Görüntü yakalanamadı!")
#                 break
            
#             frame_count += 1
            
#             # Fare konumunu görüntüye ekle
#             mouse_text = f'Mouse Position: ({wincap.mouse_x}, {wincap.mouse_y})'
#             cv.putText(screenshot, mouse_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
#             # FPS hesapla ve ekrana yaz
#             fps_text = f'FPS: {1 / (time() - loop_time):.2f}'
#             cv.putText(screenshot, fps_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
#             # Frame sayısını ekle
#             frame_text = f'Frame: {frame_count}'
#             cv.putText(screenshot, frame_text, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
#             loop_time = time()
            
#             # Ekran görüntüsünü göster
#             cv.imshow('Computer Vision', screenshot)
            
#             # Tuş kontrolü
#             key = cv.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 print("Çıkış yapılıyor...")
#                 cv.destroyAllWindows()
#                 break
#             elif key == ord('c'):
#                 # Manuel pozisyon yazdırma
#                 print(f"Manuel pozisyon: ({wincap.mouse_x}, {wincap.mouse_y})")
#             elif key == ord('s'):
#                 # Screenshot kaydet
#                 filename = f'knight_screenshot_{frame_count}.png'
#                 cv.imwrite(filename, cv.cvtColor(screenshot, cv.COLOR_RGB2BGR))
#                 print(f"Screenshot kaydedildi: {filename}")
                
#     except Exception as e:
#         print(f"Hata: {e}")
#         print("Knight Online Client penceresinin açık ve görünür olduğundan emin olun")

# # main fonksiyonunu çağır
# if __name__ == "__main__":
#     main()