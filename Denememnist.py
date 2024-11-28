import cv2
import os
import easyocr
from matplotlib import pyplot as plt
import numpy as np

reader = easyocr.Reader(['en'])
image = cv2.imread('Letters/corrupted_plate.png', cv2.IMREAD_GRAYSCALE)

threshold_value = 100
_, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

character_images = []
for contour in sorted_contours:
    x, y, w, h = cv2.boundingRect(contour)

    if w > 5 and h > 5:
        character = binary_image[y:y + h, x:x + w]
        character_images.append(character)

for idx, char_img in enumerate(character_images):
    cv2.imshow(f'char {idx + 1}', char_img)
    cv2.waitKey(0)

recognized_characters = []
for char_img in character_images:
    result = reader.readtext(char_img, paragraph="False")
    print(result)
    recognized_characters.append(result)

print(recognized_characters)
cv2.destroyAllWindows()
