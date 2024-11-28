import cv2
import os
import easyocr
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('Models/my_model.h5')
#reader = easyocr.Reader(['en'])
image = cv2.imread('Letters/onetonine.png', cv2.IMREAD_GRAYSCALE)

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
    resized_char = cv2.resize(char_img, (28, 28))
    normalized_char = resized_char / 255.0
    input_char = np.expand_dims(normalized_char, axis=(0, -1))
    print(input_char.shape)

    prediction = model.predict(input_char)
    recognized_characters.append(np.argmax(prediction))

print(recognized_characters)
cv2.destroyAllWindows()
