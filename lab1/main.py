import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from color_model import *
from rediscretization import *

def create_output_dirs():
    """Создает папки для сохранения результатов"""
    dirs = ['results', 
            'results/color_models', 
            'results/rediscretization']
    
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

def save_and_show_image(image, filename, title):
    cv2.imwrite(filename, image)

def process_color_models(image):

    r_image, g_image, b_image = extract_rgb_components(image)
    
    save_and_show_image(r_image, 'results/color_models/r_component.png', 'R-компонента')
    save_and_show_image(g_image, 'results/color_models/g_component.png', 'G-компонента')
    save_and_show_image(b_image, 'results/color_models/b_component.png', 'B-компонента')
    
   
    h_display, s_display, i_display = convert_to_hsi(image)
    
   
    save_and_show_image(i_display, 'results/color_models/intensity_component.png', 'Интенсивность (I)')
    
    inverted_image = invert_brightness(image)
    save_and_show_image(inverted_image, 'results/color_models/inverted_brightness.png', 
                       'Инвертированная яркость')

def process_rediscretization(image, m_factor, n_factor):

    enlarged_image = enlarge_image(image, m_factor)
    save_and_show_image(enlarged_image, f'results/rediscretization/enlarged_{m_factor}x.png', 
                       f'Растяжение в {m_factor} раз')
    
    decimated_image = decimate_image(image, n_factor)
    save_and_show_image(decimated_image, f'results/rediscretization/decimated_{n_factor}x.png', 
                       f'Сжатие в {n_factor} раз')
    
    k_factor = m_factor / n_factor
    rediscretized_two_pass = two_pass_rediscretization(image, m_factor, n_factor)
    save_and_show_image(rediscretized_two_pass, 
                       f'results/rediscretization/rediscretized_two_pass_{k_factor}x.png', 
                       f'Передискретизация {k_factor}x (два прохода)')
    

    rediscretized_one_pass = one_pass_rediscretization(image, k_factor)
    save_and_show_image(rediscretized_one_pass, 
                       f'results/rediscretization/rediscretized_one_pass_{k_factor}x.png', 
                       f'Передискретизация {k_factor}x (один проход)')

def main():
   
    if len(sys.argv) != 4:
        print("Использование: python main.py <путь_к_изображению> <фактор_M> <фактор_N>")
        print("Пример: python main.py input.png 2 1.5")
        return
    
   
    image_path = sys.argv[1]
    m_factor = float(sys.argv[2])
    n_factor = float(sys.argv[3])
    
 
    if not os.path.isfile(image_path):
        print(f"Ошибка: файл {image_path} не найден")
        return
    

    _, ext = os.path.splitext(image_path)
    if ext.lower() not in ['.bmp', '.png']:
        print("Ошибка: используйте только изображения в форматах .bmp или .png")
        return
    
   
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return
    
   
    create_output_dirs()
    
   
    save_and_show_image(image, 'results/original.png', 'Исходное изображение')
    
    
    process_color_models(image)
    process_rediscretization(image, m_factor, n_factor)
    

if __name__ == "__main__":
    main()
