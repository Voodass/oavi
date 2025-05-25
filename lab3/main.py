import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def create_output_directory():
    if not os.path.exists('results'):
        os.makedirs('results')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join('results', timestamp)
    os.makedirs(result_dir)
    
    return result_dir

def convert_to_grayscale(image):
    if len(image.shape) == 2:
        return image
    
    height, width, _ = image.shape
    grayscale_img = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            b, g, r = image[i, j]
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            grayscale_img[i, j] = gray_value
            
    return grayscale_img

def apply_predominant_tone_filter(image, window_size=3):
    height, width = image.shape
    filtered_image = np.zeros_like(image)
    
    padding = window_size // 2
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)
    
    for i in range(height):
        for j in range(width):
            window = padded_image[i:i+window_size, j:j+window_size].flatten()
            values, counts = np.unique(window, return_counts=True)
            predominant_tone = values[np.argmax(counts)]
            
            filtered_image[i, j] = predominant_tone
                
    return filtered_image

def create_difference_image(original, filtered):
    difference = cv2.absdiff(original, filtered)
    return difference

def process_image(image_path):
    result_dir = create_output_directory()
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return
    
    original_bgr = image.copy()
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    grayscale_image = convert_to_grayscale(image)
    
    filtered_3x3 = apply_predominant_tone_filter(grayscale_image, window_size=3)
    filtered_5x5 = apply_predominant_tone_filter(grayscale_image, window_size=5)
    
    difference_3x3 = create_difference_image(grayscale_image, filtered_3x3)
    difference_5x5 = create_difference_image(grayscale_image, filtered_5x5)
    
    cv2.imwrite(os.path.join(result_dir, 'original.png'), original_bgr)
    cv2.imwrite(os.path.join(result_dir, 'grayscale.png'), grayscale_image)
    cv2.imwrite(os.path.join(result_dir, 'filtered_3x3.png'), filtered_3x3)
    cv2.imwrite(os.path.join(result_dir, 'filtered_5x5.png'), filtered_5x5)
    cv2.imwrite(os.path.join(result_dir, 'difference_3x3.png'), difference_3x3)
    cv2.imwrite(os.path.join(result_dir, 'difference_5x5.png'), difference_5x5)
    
    
    
    print(f"Обработка завершена. Результаты сохранены в папку '{result_dir}'")

def process_multiple_images(image_paths):
    for image_path in image_paths:
        print(f"Обработка изображения: {image_path}")
        process_image(image_path)
        print("-" * 50)

def main():
    import sys
    
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
        process_multiple_images(image_paths)
    else:
        image_path = input("Введите путь к изображению: ")
        process_image(image_path)

if __name__ == "__main__":
    main()
