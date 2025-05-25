import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

SRC_PATH = Path("pictures_src/")
ALPHABET_DIR = Path("alphabet")
DST_DIR = Path("pictures_results")
PHRASE_GT = "UN JUGOSO"
SIZE = (64, 64)

ALPHABET = list("ABCDEFGHIJKLMNÑOPQRSTUVWXYZ")

os.makedirs(DST_DIR, exist_ok=True)

def to_binary(img_or_path) -> np.ndarray:
    """Возвращает бинарное изображение 0/1 (1 — чёрный)."""
    img = Image.open(img_or_path) if isinstance(img_or_path, (str, Path)) else img_or_path
    img = img.convert("L")  # градации серого
    arr = np.array(img)
    return (arr < 128).astype(np.uint8)

def normalize_bin(arr: np.ndarray, size: tuple[int, int] = SIZE) -> np.ndarray:
    """Плотно обрезает символ, центрирует на квадратном холсте, масштабирует."""
    ys, xs = np.nonzero(arr)
    if ys.size == 0:
        return np.zeros(size, dtype=np.uint8)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    crop = arr[y0:y1 + 1, x0:x1 + 1]

    h, w = crop.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = crop

    pil = Image.fromarray(canvas * 255)  # обратно 0/255
    pil = pil.resize(size, Image.LANCZOS)
    res = np.array(pil)
    return (res < 128).astype(np.uint8)

def calculate_profiles(binary_img):
    """Вычисляет горизонтальный и вертикальный профили изображения."""
    h_profile = np.sum(binary_img, axis=1)  # Горизонтальный профиль (по строкам)
    v_profile = np.sum(binary_img, axis=0)  # Вертикальный профиль (по столбцам)
    return h_profile, v_profile

def segment_lines(binary_img, h_profile, line_threshold=5):
    """Сегментирует изображение на строки текста."""
    lines = []
    in_line = False
    start_row = 0
    
    for i, value in enumerate(h_profile):
        if not in_line and value > line_threshold:
            in_line = True
            start_row = i
        elif in_line and value <= line_threshold:
            lines.append((start_row, i-1))
            in_line = False
    
    if in_line:
        lines.append((start_row, len(h_profile)-1))

    line_images = []
    for start, end in lines:
        line_img = binary_img[start:end+1, :]
        line_images.append((line_img, (0, start)))  # (image, (offset_x, offset_y))
    
    return line_images, lines

def segment_characters(line_img, v_profile, char_threshold=2):
    """Сегментирует строку на отдельные символы."""
    chars = []
    in_char = False
    start_col = 0
    
    for i, value in enumerate(v_profile):
        if not in_char and value > char_threshold:
            in_char = True
            start_col = i
        elif in_char and value <= char_threshold:
            # Проверяем, не является ли это просто промежутком внутри символа
            lookahead = min(i + 5, len(v_profile) - 1)  # Смотрим вперед на 5 пикселей
            if any(v_profile[i+1:lookahead+1] > char_threshold):
                continue
            chars.append((start_col, i-1))
            in_char = False
    
    if in_char:
        chars.append((start_col, len(v_profile)-1))

    return chars

def segment_text_to_characters(binary_img):
    """Основная функция сегментации текста на символы."""
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

def split_wide_boxes(boxes, bin_img, factor: float = 1.8, min_cut_width: int = 3):
    """Разделяет слишком широкие боксы на несколько символов."""
    widths = [x1 - x0 + 1 for x0, _, x1, _ in boxes]
    if not widths:
        return boxes
    avg_w = sum(widths) / len(widths)

    out = []
    for (x0, y0, x1, y1), w in zip(boxes, widths):
        if w > avg_w * factor:
            sub = bin_img[y0:y1 + 1, x0:x1 + 1]
            vert = sub.sum(axis=0)
            
            m = max(w // 4, min_cut_width)
            local = vert[m:-m]
            
            if local.size:
                smoothed = np.convolve(local, np.ones(5)/5, mode='valid')
                if len(smoothed) > 0:
                    cut_rel = np.argmin(smoothed) + 1 
                    cut_off = cut_rel + m
                    
                    left_width = cut_off
                    right_width = w - cut_off - 1
                    if left_width >= min_cut_width and right_width >= min_cut_width:
                        cut = x0 + cut_off
                        out += [(x0, y0, cut, y1), (cut + 1, y0, x1, y1)]
                        continue
        
        out.append((x0, y0, x1, y1))
    return out

def gap_is_space(prev_box, curr_box, ratio=1.5):
    """Определяет, является ли промежуток между символами пробелом."""
    if prev_box is None:
        return False
    gap = curr_box[0] - prev_box[2]
    avg_char_width = (prev_box[2] - prev_box[0] + curr_box[2] - curr_box[0]) / 2
    return gap > avg_char_width * ratio

def load_templates():
    """Загружает шаблоны букв."""
    tpls = []
    for ch in ALPHABET:
        path = ALPHABET_DIR / f"{ch}.bmp"
        bin_img = to_binary(path)
        tpl = normalize_bin(bin_img).astype(bool)
        tpls.append(tpl)
    return np.stack(tpls, axis=0), ALPHABET

def compute_iou_batch(tpl_stack: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """Вычисляет IoU (Intersection over Union) между шаблонами и изображением."""
    sub_b = sub.astype(bool)
    inter = np.logical_and(tpl_stack, sub_b).sum(axis=(1, 2))
    union = np.logical_or(tpl_stack, sub_b).sum(axis=(1, 2))
    return inter / np.where(union == 0, 1, union)

def recognise_image(path: Path, tpl_stack: np.ndarray, keys: list[str]):
    """Распознает текст на изображении."""
    bin_img = to_binary(path)
    boxes = segment_text_to_characters(bin_img)
    boxes = split_wide_boxes(boxes, bin_img)
    boxes.sort(key=lambda b: b[0])

    result = ""
    letters, scores = [], []
    prev = None

    for x0, y0, x1, y1 in boxes:
        sub = bin_img[y0:y1 + 1, x0:x1 + 1]
        sub = normalize_bin(sub).astype(bool)

        ious = compute_iou_batch(tpl_stack, sub)
        best_idx = int(ious.argmax())
        best_ch = keys[best_idx]
        best_score = float(ious[best_idx])

        if gap_is_space(prev, (x0, y0, x1, y1)):
            result += " "
        result += best_ch
        letters.append(best_ch)
        scores.append(best_score)
        prev = (x0, y0, x1, y1)

    return result, boxes, letters, scores

def levenshtein(a: str, b: str) -> int:
    """Вычисляет расстояние Левенштейна между двумя строками."""
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1,      
                        dp[j - 1] + 1,   
                        prev + cost)    
            prev = cur
    return dp[-1]

def accuracy(pred: str, gt: str):
    """Вычисляет точность распознавания."""
    dist = levenshtein(pred, gt)
    max_len = max(len(pred), len(gt))
    return dist, 100 * (1 - dist / max_len) if max_len else 100.0

def main():
    print("[1] Загрузка шаблонов...")
    tpl_stack, keys = load_templates()

    print("[2] Сегментация и распознавание...")

    for img_path in SRC_PATH.glob("*.bmp"):
        print(f"\nОбрабатывается: {img_path}")
        recog, boxes, letters, scores = recognise_image(img_path, tpl_stack, keys)
        errs, pct = accuracy(recog.replace(" ", ""), PHRASE_GT.replace(" ", ""))

        print(f"\nРаспознано : {recog}")
        print(f"Эталон     : {PHRASE_GT}")
        print(f"Ошибок     : {errs}/{len(PHRASE_GT.replace(' ', ''))}  |  Точность: {pct:.2f}%")

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=1)
        img.save(DST_DIR / f"{img_path.stem}_boxes_fixed.bmp")

        hyp_path = DST_DIR / f"{img_path.stem}_best_hypotheses.txt"
        with hyp_path.open("w", encoding="utf-8") as f:
            f.write("Выводятся лучшие гипотезы\n\n")
            idx = 1
            for ch, sc in zip(letters, scores):
                if ch == " ": 
                    continue
                f.write(f"{idx:2d}: '{ch}' - {sc:.6f}\n")
                idx += 1
        print(f"[✓] Файл с гипотезами → {hyp_path}")

    print("[✓] Готово, остальные результаты — в", DST_DIR)

if __name__ == "__main__":
    main()
