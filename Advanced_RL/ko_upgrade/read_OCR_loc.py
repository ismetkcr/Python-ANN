# ocr_reader.py

import cv2
import re
import time
import numpy as np
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    OCR_AVAILABLE = True
except Exception as e:
    OCR_AVAILABLE = False
    print(f"❌ OCR yüklenemedi: {e}")
    exit()

from get_window import WindowCapture  # Pencere ekran görüntüsü için


def create_loc_rect():
    loc_top_left = (50, 106)
    loc_bottom_right = (160, 120)
    return [(loc_top_left, loc_bottom_right)]


def read_location_with_ocr(screenshot, rectangles):
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


if __name__ == "__main__":
    print("📡 OCR sürekli okuma başlatıldı. Çıkmak için Ctrl+C...")
    wincap = WindowCapture('Knight Online Client')
    loc_rect = create_loc_rect()

    try:
        while True:
            screenshot = wincap.get_screenshot()
            if screenshot is not None:
                result = read_location_with_ocr(screenshot, loc_rect)
                print(f"📍 OCR sonucu: {result}")
            else:
                print("❌ Ekran görüntüsü alınamadı")

            time.sleep(0.5)  # 0.5 saniye bekle
    except KeyboardInterrupt:
        print("\n🛑 Okuma sonlandırıldı.")
