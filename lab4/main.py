import cv2
import numpy as np
import os

def apply_kroon_operator(image_path, output_dir="result", threshold=50):

    os.makedirs(output_dir, exist_ok=True)


    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Изображение {image_path} не найдено!")
    
    gray_image = convert_to_grayscale(image)
    
  
    kernel_gx = np.array([[17, 61, 17],
                          [0, 0, 0],
                          [-17, -61, -17]])
    
    kernel_gy = np.array([[17, 0, -17],
                          [61, 0, -61],
                          [17, 0, -17]])
    

    gx = cv2.filter2D(gray_image, cv2.CV_64F, kernel_gx)
    gy = cv2.filter2D(gray_image, cv2.CV_64F, kernel_gy)
    

    g = np.sqrt(gx**2 + gy**2)
    

    gx_normalized = cv2.normalize(gx, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    gy_normalized = cv2.normalize(gy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    g_normalized = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

  
    
    

    _, g_binary = cv2.threshold(g_normalized, threshold, 255, cv2.THRESH_BINARY)
    

    cv2.imwrite(f"{output_dir}/1_original_color.png", image)
    cv2.imwrite(f"{output_dir}/2_grayscale.png", gray_image)
    cv2.imwrite(f"{output_dir}/3_gx_normalized.png", gx_normalized)
    cv2.imwrite(f"{output_dir}/4_gy_normalized.png", gy_normalized)
    cv2.imwrite(f"{output_dir}/5_g_normalized.png", g_normalized)
    cv2.imwrite(f"{output_dir}/6_g_binary_th{threshold}.png", g_binary)
    
    print(f"Результаты сохранены в папку '{output_dir}' в формате PNG!")

def convert_to_grayscale(image):
    height, width, _ = image.shape
    grayscale_img = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            b, g, r = image[i, j]
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            grayscale_img[i, j] = gray_value
            
    return grayscale_img


image_path = "base.png" 
apply_kroon_operator(image_path, threshold=1) 
