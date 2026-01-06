import win32gui, win32ui, win32con
import numpy as np
import cv2 as cv
from time import time

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
        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

        # Ofset değerleri, ekran pozisyonlarıyla eşleşir
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y

        # Fare konumu
        self.mouse_x, self.mouse_y = 0, 0

        # Pencereyi tanımla ve fare geri çağrısını ayarla
        cv.namedWindow('Computer Vision')
        cv.setMouseCallback('Computer Vision', self.mouse_callback)


    def mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_MOUSEMOVE:
            self.mouse_x, self.mouse_y = x, y

    def get_screenshot(self):
        # Pencere görüntüsünü al
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        # Ham veriyi OpenCV formatına dönüştür
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
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

# def main():
#     # Sınıf örneğini oluştur
#     wincap = WindowCapture('Knight Online Client')
#     loop_time = time()

#     while True:
#         # Güncel ekran görüntüsünü al
#         screenshot = wincap.get_screenshot()

#         # Fare konumunu görüntüye ekle
#         mouse_text = f'Mouse Position: ({wincap.mouse_x}, {wincap.mouse_y})'
#         cv.putText(screenshot, mouse_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#         # FPS hesapla ve ekrana yaz
#         fps_text = f'FPS: {1 / (time() - loop_time):.2f}'
#         cv.putText(screenshot, fps_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         loop_time = time()

#         # Ekran görüntüsünü göster
#         cv.imshow('Computer Vision', screenshot)

#         # Çıkış için 'q' tuşuna basın
#         if cv.waitKey(1) == ord('q'):
#             cv.destroyAllWindows()
#             break

# # main fonksiyonunu çağır
# if __name__ == "__main__":
#     main()
