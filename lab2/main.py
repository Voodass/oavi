import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def create_output_directory():
    if not os.path.exists('results'):
        os.makedirs('results')

def convert_to_grayscale(image):
    height, width, _ = image.shape
    grayscale_img = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            b, g, r = image[i, j]
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            grayscale_img[i, j] = gray_value
            
    return grayscale_img

def apply_nick_threshold(image, window_size=3, k=-0.2):
    height, width = image.shape
    binary_image = np.zeros_like(image)
    
    padding = window_size // 2
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)
    
    for i in range(height):
        for j in range(width):
            window = padded_image[i:i+window_size, j:j+window_size]
            window_mean = np.mean(window)
            window_std = np.std(window)
            
            threshold = window_mean + k * window_std
            
            if image[i, j] > threshold:
                binary_image[i, j] = 255
            else:
                binary_image[i, j] = 0
                
    return binary_image

def process_image(image_path):
    create_output_directory()
    
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return
        
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    grayscale_image = convert_to_grayscale(original_image)
    
    binary_image_nick = apply_nick_threshold(grayscale_image, window_size=3)
    
    cv2.imwrite('results/original.png', original_image)
    cv2.imwrite('results/grayscale.bmp', grayscale_image)
    cv2.imwrite('results/binary_nick_3x3.png', binary_image_nick)
    


def main():
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Введите путь к изображению: ")
        
    process_image(image_path)

if __name__ == "__main__":
    main()

