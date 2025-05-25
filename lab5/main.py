import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import csv

def create_character_image(character, font_path, font_size=52, output_dir="output"):
    img_size = font_size * 2
    img = Image.new('L', (img_size, img_size), color=255)
    draw = ImageDraw.Draw(img)
    
    font = ImageFont.truetype(font_path, font_size)
    
    text_bbox = draw.textbbox((0, 0), character, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    position = ((img_size - text_width) // 2, (img_size - text_height) // 2)
    draw.text(position, character, font=font, fill=0)
    
    img_array = np.array(img)
    
    rows = np.any(img_array < 255, axis=1)
    cols = np.any(img_array < 255, axis=0)
    
    if np.any(rows) and np.any(cols):
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        padding = 0
        rmin = max(0, rmin - padding)
        rmax = min(img_size - 1, rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(img_size - 1, cmax + padding)
        
        img_cropped = img.crop((cmin, rmin, cmax + 1, rmax + 1))
    else:
        img_cropped = img
    
    os.makedirs(output_dir, exist_ok=True)
    
    if character in "\\/:*?\"<>|":
        file_name = f"{output_dir}/symbol_{ord(character)}.png"
    else:
        file_name = f"{output_dir}/{character}.png"
    
    img_cropped.save(file_name)
    
    return file_name, character

def generate_spain_alphabet_images(font_path="", font_size=52, characters="", output_dir="spain_alphabet"):
    if not characters:
        characters = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
    
    if not font_path:
        font_path = "C:\\Windows\\Fonts\\times.ttf"
    
    print(f"Генерация изображений испанских букв в папку {output_dir}...")
    
    generated_chars = []
    for char in characters:
        file_path, char = create_character_image(char, font_path, font_size, output_dir=output_dir)
        generated_chars.append(char)
        print(f"Создан файл: {file_path}")
    
    print("Генерация завершена!")
    return generated_chars

def calculate_features(image_path):
  
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    img_binary = 255 - img_array
    
    height, width = img_binary.shape
    

    half_height = height // 2
    half_width = width // 2
    quarters = [
        img_binary[:half_height, :half_width],       
        img_binary[:half_height, half_width:],       
        img_binary[half_height:, :half_width],      
        img_binary[half_height:, half_width:]         
    ]
    

    weights = [np.sum(quarter) for quarter in quarters]
    

    quarter_areas = [quarter.size for quarter in quarters]
    specific_weights = [weights[i] / quarter_areas[i] if quarter_areas[i] else 0 for i in range(4)]
    

    y_indices, x_indices = np.indices(img_binary.shape)
    total_weight = np.sum(img_binary)
    if total_weight > 0:
        center_x = np.sum(x_indices * img_binary) / total_weight
        center_y = np.sum(y_indices * img_binary) / total_weight
    else:
        center_x, center_y = 0, 0
    

    norm_center_x = center_x / width if width else 0
    norm_center_y = center_y / height if height else 0
    

    moment_x = np.sum(img_binary * (y_indices - center_y)**2)
    moment_y = np.sum(img_binary * (x_indices - center_x)**2)
    

    norm_moment_x = moment_x / (total_weight * height**2) if total_weight and height else 0
    norm_moment_y = moment_y / (total_weight * width**2) if total_weight and width else 0

    profile_x = np.sum(img_binary, axis=0)
    profile_y = np.sum(img_binary, axis=1)

    features = {
        'weights': weights,
        'specific_weights': specific_weights,
        'center': (center_x, center_y),
        'norm_center': (norm_center_x, norm_center_y),
        'moment': (moment_x, moment_y),
        'norm_moment': (norm_moment_x, norm_moment_y),
        'profile_x': profile_x,
        'profile_y': profile_y
    }
    
    return features

def save_features_to_csv(characters, features_dict, output_path):
    """Сохранение признаков в CSV файл."""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'char', 
            'weight_1', 'weight_2', 'weight_3', 'weight_4',
            'spec_weight_1', 'spec_weight_2', 'spec_weight_3', 'spec_weight_4',
            'center_x', 'center_y', 
            'norm_center_x', 'norm_center_y',
            'moment_x', 'moment_y',
            'norm_moment_x', 'norm_moment_y'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        
        writer.writeheader()
        for char in characters:
            features = features_dict[char]
            
            rounded_row = {
                'char': char,
                'weight_1': round(features['weights'][0], 3),
                'weight_2': round(features['weights'][1], 3),
                'weight_3': round(features['weights'][2], 3),
                'weight_4': round(features['weights'][3], 3),
                'spec_weight_1': round(features['specific_weights'][0], 3),
                'spec_weight_2': round(features['specific_weights'][1], 3),
                'spec_weight_3': round(features['specific_weights'][2], 3),
                'spec_weight_4': round(features['specific_weights'][3], 3),
                'center_x': round(features['center'][0], 3),
                'center_y': round(features['center'][1], 3),
                'norm_center_x': round(features['norm_center'][0], 3),
                'norm_center_y': round(features['norm_center'][1], 3),
                'moment_x': round(features['moment'][0], 3),
                'moment_y': round(features['moment'][1], 3),
                'norm_moment_x': round(features['norm_moment'][0], 3),
                'norm_moment_y': round(features['norm_moment'][1], 3)
            }
            
            writer.writerow(rounded_row)

def save_profiles(char, features, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    safe_char = char
    if char in "\\/:*?\"<>|":
        safe_char = f"symbol_{ord(char)}"
    

    plt.figure(figsize=(8, 6))
    plt.bar(range(len(features['profile_x'])), features['profile_x'])
    plt.title(f'Профиль X для символа "{char}"')
    plt.xlabel('X')
    plt.ylabel('Сумма')
    plt.grid(True)
    plt.xticks(range(0, len(features['profile_x']), max(1, len(features['profile_x'])//10)))
    plt.savefig(os.path.join(output_dir, f"{safe_char}_profile_x.png"))
    plt.close()
    

    plt.figure(figsize=(6, 8))
    plt.barh(range(len(features['profile_y'])), features['profile_y'])
    plt.title(f'Профиль Y для символа "{char}"')
    plt.ylabel('Y')
    plt.xlabel('Сумма')
    plt.grid(True)
    plt.xticks(range(0, len(features['profile_y']), max(1, len(features['profile_y'])//10)))
    plt.savefig(os.path.join(output_dir, f"{safe_char}_profile_y.png"))
    plt.close()

def main():
    
    font_path = "C:\\Windows\\Fonts\\times.ttf" 
    font_size = 52
    output_dir = "character_images"
    features_dir = "features"
    profiles_dir = "profiles"
    

    characters = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
    
 
    generated_chars = generate_spain_alphabet_images(font_path, font_size, characters, output_dir)
    

    features_dict = {}
    for char in generated_chars:
    
        safe_char = char
        if char in "\\/:*?\"<>|":
            safe_char = f"symbol_{ord(char)}"
            
        image_path = os.path.join(output_dir, f"{safe_char}.png")
        features = calculate_features(image_path)
        features_dict[char] = features
    

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    save_features_to_csv(generated_chars, features_dict, os.path.join(features_dir, "features.csv"))
    
 
    for char in generated_chars:
        save_profiles(char, features_dict[char], profiles_dir)
    
    print(f"Обработка завершена. Изображения сохранены в {output_dir}, признаки в {features_dir}, профили в {profiles_dir}")

if __name__ == "__main__":
    main()
