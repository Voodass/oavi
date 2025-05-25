import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

def load_and_preprocess_image(file_path):
    
 
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {file_path}")
    
 
    _, binary_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    
    return img, binary_img

def calculate_profiles(binary_img):


    horizontal_profile = np.sum(binary_img, axis=1)
    

    vertical_profile = np.sum(binary_img, axis=0)
    
    return horizontal_profile, vertical_profile

def segment_lines(binary_img, horizontal_profile, min_line_height=5):
 

    threshold = 0.05 * np.max(horizontal_profile) if np.max(horizontal_profile) > 0 else 0
    

    line_positions = []
    in_line = False
    start_index = 0
    
    for i, value in enumerate(horizontal_profile):
        if not in_line and value > threshold:
            in_line = True
            start_index = i
        elif in_line and (value <= threshold or i == len(horizontal_profile) - 1):
            in_line = False
            if i - start_index > min_line_height: 
                line_positions.append((start_index, i))
    
  
    line_images = []
    for start_y, end_y in line_positions:
        line_img = binary_img[start_y:end_y, :]
        line_images.append((line_img, (0, start_y)))
    
    return line_images, line_positions

def segment_characters(line_img, vertical_profile, min_char_width=3, min_gap_width=1):

    threshold = 0.05 * np.max(vertical_profile) if np.max(vertical_profile) > 0 else 0
    

    char_positions = []
    in_char = False
    start_index = 0
    
    for i, value in enumerate(vertical_profile):
        if not in_char and value > threshold:
            in_char = True
            start_index = i
        elif in_char and (value <= threshold or i == len(vertical_profile) - 1):
            in_char = False
            if i - start_index > min_char_width: 
                char_positions.append((start_index, i))
    
   
    merged_positions = []
    if char_positions:
        current_start, current_end = char_positions[0]
        
        for i in range(1, len(char_positions)):
            start, end = char_positions[i]
            

            if start - current_end <= min_gap_width:
                current_end = end
            else:
                merged_positions.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged_positions.append((current_start, current_end))
    
    return merged_positions

def segment_text_to_characters(binary_img):

    h_profile, _ = calculate_profiles(binary_img)
    

    line_images, line_positions = segment_lines(binary_img, h_profile)
    
    all_character_boxes = []
    

    for line_idx, (line_img, (offset_x, offset_y)) in enumerate(line_images):

        _, v_profile = calculate_profiles(line_img)

        char_positions = segment_characters(line_img, v_profile)

        for start_x, end_x in char_positions:
  
            char_img = line_img[:, start_x:end_x]
            char_h_profile = np.sum(char_img, axis=1)
            
   
            non_zero_rows = np.where(char_h_profile > 0)[0]
            if len(non_zero_rows) > 0:
                start_y_relative = np.min(non_zero_rows)
                end_y_relative = np.max(non_zero_rows)
            else:
                start_y_relative = 0
                end_y_relative = line_img.shape[0] - 1
            
         
            start_y_absolute = offset_y + start_y_relative
            end_y_absolute = offset_y + end_y_relative
            
          
            all_character_boxes.append((
                start_x + offset_x,
                start_y_absolute,
                end_x + offset_x,
                end_y_absolute
            ))
    
    return all_character_boxes

def extract_characters(img, binary_img, char_boxes, output_dir="segmented_chars"):
  
    os.makedirs(output_dir, exist_ok=True)
    
    extracted_chars = []
    for i, (x1, y1, x2, y2) in enumerate(char_boxes):
      
        char_img = binary_img[y1:y2+1, x1:x2+1]
        
      
        char_filename = f"{output_dir}/char_{i+1:03d}.png"
        cv2.imwrite(char_filename, char_img)
        
        extracted_chars.append((char_img, char_filename))
    
    return extracted_chars

def calculate_character_profiles(char_img):
   
    h_profile = np.sum(char_img, axis=1)
    v_profile = np.sum(char_img, axis=0)
    
    
    h_profile = h_profile / np.max(h_profile) if np.max(h_profile) > 0 else h_profile
    v_profile = v_profile / np.max(v_profile) if np.max(v_profile) > 0 else v_profile
    
    return h_profile, v_profile

def save_character_profiles(binary_img, char_boxes, output_dir="char_profiles"):

    os.makedirs(output_dir, exist_ok=True)
    
  
    for i, (x1, y1, x2, y2) in enumerate(char_boxes):
    
        char_img = binary_img[y1:y2+1, x1:x2+1]
        
 
        h_profile, v_profile = calculate_character_profiles(char_img)
        
     
        plt.figure(figsize=(10, 6))
        
      
        plt.subplot(2, 2, 1)
        plt.imshow(char_img, cmap='gray')
        plt.title(f'Символ {i+1}')
        plt.axis('off')
        
     
        plt.subplot(2, 2, 2)
        plt.plot(h_profile, range(len(h_profile)))
        plt.title('Вертикальный профиль')
        plt.gca().invert_yaxis() 
        
       
        plt.subplot(2, 2, 3)
        plt.plot(range(len(v_profile)), v_profile)
        plt.title('Горизонтальный профиль')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/profile_char_{i+1:03d}.png")
        plt.close()  
    
  
    h_profile, v_profile = calculate_profiles(binary_img)
    
    plt.figure(figsize=(12, 8))
    
  
    plt.subplot(2, 1, 1)
    plt.plot(h_profile, range(len(h_profile)))
    plt.title('Горизонтальный профиль изображения')
    plt.gca().invert_yaxis()
    
 
    plt.subplot(2, 1, 2)
    plt.plot(range(len(v_profile)), v_profile)
    plt.title('Вертикальный профиль изображения')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/full_image_profiles.png")
    plt.close()

def save_segmentation_result(img_orig, char_boxes, output_path="segmentation_result.png"):

    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2RGB)
    

    for box in char_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 1)
    

    cv2.imwrite(output_path, img_rgb)

def main():

    image_path = "base.bmp"
    
    try:
      
        img_orig, binary_img = load_and_preprocess_image(image_path)
        
        char_boxes = segment_text_to_characters(binary_img)

        extracted_chars = extract_characters(img_orig, binary_img, char_boxes)
 
        save_character_profiles(binary_img, char_boxes)

        save_segmentation_result(img_orig, char_boxes)
        
        print(f"Обработка завершена. Обнаружено {len(char_boxes)} символов.")
        print("Результаты сохранены в файлах и папках:")
        print("- segmentation_result.png - выделенные символы")
        print("- segmented_chars/ - папка с изображениями отдельных символов")
        print("- char_profiles/ - папка с профилями символов и всего изображения")
        
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    main()
