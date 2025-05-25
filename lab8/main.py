import cv2
import numpy as np
import os
from skimage.feature import hog
from skimage import exposure

KRONE_GX = np.array([
    [17, 61, 17],
    [0, 0, 0],
    [-17, -61, -17]
], dtype=np.float32)

KRONE_GY = np.array([
    [17, 0, -17],
    [61, 0, -61],
    [17, 0, -17]
], dtype=np.float32)

def apply_krone_operator(image):
    """Применение оператора Круна для выделения границ"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges_x = cv2.filter2D(gray, -1, KRONE_GX)
    edges_y = cv2.filter2D(gray, -1, KRONE_GY)
    edges = np.sqrt(edges_x**2 + edges_y**2)
    return edges.astype(np.uint8)

def create_output_directory():
    """Создает директорию для сохранения результатов, если ее нет"""
    if not os.path.exists('output'):
        os.makedirs('output')

def load_image(path):
    """Загрузка изображения и преобразование в HSL"""
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Изображение не найдено")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsl_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    return image, hsl_image

def apply_linear_brightness(hsl_image, alpha=1.3, beta=10):
    """Линейное преобразование яркости: L = alpha*L + beta"""
    l_channel = hsl_image[:, :, 1].astype(np.float32)
    l_channel = alpha * l_channel + beta
    l_channel = np.clip(l_channel, 0, 255).astype(np.uint8)
    hsl_modified = hsl_image.copy()
    hsl_modified[:, :, 1] = l_channel
    return hsl_modified

def compute_hog_features(image):
    """Вычисление HOG с предварительным выделением границ оператором Круна"""
    # Применение оператора Круна
    edges = apply_krone_operator(image)
    
    # Вычисление HOG на полученных границах
    features, hog_image = hog(
        edges,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True,
        block_norm='L2-Hys'
    )
    
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return features, hog_image

def save_histogram(image, filename):
    """Сохранение гистограммы изображения"""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.figure()
    plt.plot(hist)
    plt.xlabel("Значение яркости")
    plt.ylabel("Частота")
    plt.savefig(filename)
    plt.close()

def save_results(original_rgb, modified_hsl, original_hog, modified_hog, original_gray, modified_gray):
    """Сохранение всех результатов в файлы"""
    # Сохранение изображений
    cv2.imwrite('output/original_image.jpg', cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite('output/modified_image.jpg', cv2.cvtColor(cv2.cvtColor(modified_hsl, cv2.COLOR_HLS2RGB), cv2.COLOR_RGB2BGR))
    
    # Сохранение HOG изображений
    cv2.imwrite('output/original_hog.jpg', (original_hog * 255).astype(np.uint8))
    cv2.imwrite('output/modified_hog.jpg', (modified_hog * 255).astype(np.uint8))
    
    # Сохранение гистограмм
    save_histogram(original_gray, 'output/original_histogram.png')
    save_histogram(modified_gray, 'output/modified_histogram.png')
    
    # Сохранение полутоновых изображений
    cv2.imwrite('output/original_gray.jpg', original_gray)
    cv2.imwrite('output/modified_gray.jpg', modified_gray)

def main(image_path):
    create_output_directory()
    
    # 1. Загрузка изображения
    original_rgb, original_hsl = load_image(image_path)
    
    # 2. Применение линейного преобразования яркости
    modified_hsl = apply_linear_brightness(original_hsl)
    
    # 3. Получение полутоновых версий
    original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    modified_gray = cv2.cvtColor(cv2.cvtColor(modified_hsl, cv2.COLOR_HLS2RGB), cv2.COLOR_RGB2GRAY)
    
    # 4. Вычисление HOG-признаков
    original_features, original_hog_img = compute_hog_features(original_rgb)
    modified_features, modified_hog_img = compute_hog_features(cv2.cvtColor(modified_hsl, cv2.COLOR_HLS2RGB))
    
    # 5. Нормализованные HOG-признаки (Hnorm)
    H_norm_original = np.linalg.norm(original_features)
    H_norm_modified = np.linalg.norm(modified_features)
    
    # Сохранение результатов в файл
    with open('output/results.txt', 'w') as f:
        f.write(f"Hnorm исходного изображения: {H_norm_original:.2f}\n")
        f.write(f"Hnorm после преобразования: {H_norm_modified:.2f}\n")
        f.write(f"Разница: {abs(H_norm_original - H_norm_modified):.2f}\n")
    
    # 6. Сохранение всех изображений и графиков
    save_results(original_rgb, modified_hsl, original_hog_img, modified_hog_img, original_gray, modified_gray)
    
    print("Обработка завершена. Результаты сохранены в папке 'output'.")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    image_path = "2.jpg"  # Укажите путь к вашему изображению
    main(image_path)
