import cv2
import os
from tqdm import tqdm
image_path = [name for name in os.listdir('image_test') if name.endswith(('png', 'jpg', 'jpeg'))]

mode = ['atr', 'lip', 'pascal']

for folder in tqdm(mode):
    for name in image_path:
        image = cv2.imread('image_test/' + name)
        mask = cv2.imread('visualize/' + folder + '/' + name.split('.')[0] + '.png')
        img = cv2.addWeighted(image, 0.3, mask, 0.8, 0)
        cv2.imwrite(f'visualize/{folder}_image/' + name, img)
