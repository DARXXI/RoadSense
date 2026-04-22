# road_analyzer.py - РАБОЧАЯ ВЕРСИЯ без qai-hub-models
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import requests
from pathlib import Path

# Загрузка модели FCN ResNet50 (доступна через torchvision)
def load_model():
    """
    Загрузка модели семантической сегментации FCN ResNet50
    """
    # Загружаем предобученную модель из torchvision
    model = torch.hub.load('pytorch/vision', 'fcn_resnet50', pretrained=True)
    model.eval()
    
    print("✅ Модель FCN ResNet50 загружена успешно!")
    return model

# Классы COCO (то, что умеет распознавать FCN)
# Нас интересуют дорожные классы
COCO_CLASSES = {
    0: '__background__',
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic_light',
    11: 'fire_hydrant',
    12: 'street_sign',
    13: 'stop_sign',
    14: 'parking_meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    26: 'hat',
    27: 'backpack',
    28: 'umbrella',
    29: 'shoe',
    30: 'eye_glasses',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports_ball',
    38: 'kite',
    39: 'baseball_bat',
    40: 'baseball_glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis_racket',
    44: 'bottle',
    45: 'plate',
    46: 'wine_glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot_dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted_plant',
    65: 'bed',
    66: 'mirror',
    67: 'dining_table',
    68: 'window',
    69: 'desk',
    70: 'toilet',
    71: 'door',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell_phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    83: 'blender',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy_bear',
    89: 'hair_drier',
    90: 'toothbrush',
    91: 'hair_brush',
    92: 'banner',
    93: 'blanket',
    94: 'branch',
    95: 'bridge',
    96: 'building_other',
    97: 'bush',
    98: 'cabinet',
    99: 'cage',
    100: 'cardboard',
    101: 'carpet',
    102: 'ceiling',
    103: 'ceiling_tile',
    104: 'cloth',
    105: 'clothes',
    106: 'clouds',
    107: 'counter',
    108: 'cupboard',
    109: 'curtain',
    110: 'desk_stuff',
    111: 'dirt',
    112: 'door_stuff',
    113: 'fence',
    114: 'floor',
    115: 'floor_wood',
    116: 'flower',
    117: 'fog',
    118: 'food_stuff',
    119: 'furniture',
    120: 'grass',
    121: 'gravel',
    122: 'ground',
    123: 'hill',
    124: 'house',
    125: 'leaves',
    126: 'light',
    127: 'mat',
    128: 'metal',
    129: 'mirror_stuff',
    130: 'moss',
    131: 'mountain',
    132: 'mud',
    133: 'napkin',
    134: 'net',
    135: 'paper',
    136: 'pavement',
    137: 'pillow',
    138: 'plant',
    139: 'plastic',
    140: 'platform',
    141: 'playingfield',
    142: 'railing',
    143: 'railroad',
    144: 'river',
    145: 'road',
    146: 'rock',
    147: 'roof',
    148: 'rug',
    149: 'salad',
    150: 'sand',
    151: 'sea',
    152: 'shelf',
    153: 'sky',
    154: 'skyscraper',
    155: 'snow',
    156: 'solid',
    157: 'stairs',
    158: 'stone',
    159: 'straw',
    160: 'structural',
    161: 'table',
    162: 'tent',
    163: 'textile',
    164: 'towel',
    165: 'tree',
    166: 'vegetation',
    167: 'wall',
    168: 'wall_other',
    169: 'water',
    170: 'water_drops',
    171: 'water_vehicle',
    172: 'wheat',
    173: 'window_other',
    174: 'wood',
    175: 'wool'
}

# Интересуют нас классы дорог и покрытия
ROAD_CLASSES = [145, 146, 122, 111, 132, 150]  # road, rock, ground, dirt, mud, sand
ROAD_NAMES = {
    145: 'дорога/асфальт',
    146: 'каменистая дорога',
    122: 'грунт',
    111: 'грязь',
    132: 'грязь/земля',
    150: 'песок'
}

def preprocess_image(image_path):
    """
    Подготовка изображения для модели FCN
    """
    # Загрузка изображения
    img = Image.open(image_path).convert('RGB')
    original_size = img.size[::-1]  # (height, width)
    
    # Трансформации для FCN ResNet50
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = preprocess(img).unsqueeze(0)
    original_img = np.array(img)
    
    return img_tensor, original_size, original_img

def analyze_road_surface(image_path, model):
    """
    Анализ дорожного покрытия на изображении
    """
    # Подготовка изображения
    img_tensor, original_size, original_img = preprocess_image(image_path)
    
    # Инференс
    with torch.no_grad():
        output = model(img_tensor)['out'][0]
        
        # Ресайз обратно к оригинальному размеру
        output = torch.nn.functional.interpolate(
            output.unsqueeze(0),
            size=original_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
    
    # Получаем предсказанные классы
    predicted_classes = output.argmax(0).cpu().numpy()
    
    return predicted_classes, original_img

def calculate_road_statistics(mask, original_img):
    """
    Расчет статистики по дорожному покрытию
    """
    stats = {}
    
    for class_id, class_name in ROAD_NAMES.items():
        # Маска для текущего класса
        class_mask = (mask == class_id)
        pixel_count = np.sum(class_mask)
        total_pixels = mask.size
        
        if pixel_count > 0:
            stats[class_name] = {
                'pixel_percentage': (pixel_count / total_pixels) * 100,
                'pixel_count': pixel_count
            }
    
    # Если нет ни одного дорожного класса, ищем ближайшие
    if not stats:
        # Ищем класс "road" (145) или любой другой
        road_mask = (mask == 145)
        if np.sum(road_mask) > 0:
            stats['дорога/асфальт'] = {
                'pixel_percentage': (np.sum(road_mask) / mask.size) * 100,
                'pixel_count': np.sum(road_mask)
            }
        else:
            stats['дорожное покрытие'] = {
                'pixel_percentage': 0,
                'pixel_count': 0
            }
    
    # Определяем основной тип покрытия
    primary_surface = max(stats.items(), key=lambda x: x[1]['pixel_percentage'])
    
    return stats, primary_surface

def visualize_results(image_path, mask, original_img):
    """
    Визуализация результатов сегментации
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    # Оригинальное изображение
    axes[0].imshow(original_img)
    axes[0].set_title('Оригинальное изображение')
    axes[0].axis('off')
    
    # Создаем цветную маску
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # Цвета для разных типов покрытия
    colors = {
        145: [128, 128, 128],  # дорога/асфальт - серый
        146: [139, 69, 19],    # каменистая - коричневый
        122: [160, 82, 45],    # грунт - светло-коричневый
        111: [101, 67, 33],    # грязь - темно-коричневый
        132: [101, 67, 33],    # грязь/земля
        150: [238, 232, 170],  # песок - бежевый
    }
    
    for class_id, color in colors.items():
        colored_mask[mask == class_id] = color
    
    axes[1].imshow(colored_mask)
    axes[1].set_title('Дорожное покрытие (сегментация)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('road_analysis_result.png', dpi=150)
    plt.show()
    
    print("✅ Результат сохранен как 'road_analysis_result.png'")

# Основная функция
def main():
    print("🚀 Запуск анализа дорожного покрытия...")
    
    # Загрузка модели
    model = load_model()
    
    # Путь к вашему изображению
    image_path = input("Введите путь к изображению (или URL): ").strip()
    
    # Если URL, скачиваем
    if image_path.startswith('http'):
        print("📥 Скачиваем изображение...")
        response = requests.get(image_path)
        temp_path = 'temp_image.jpg'
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        image_path = temp_path
        print("✅ Изображение скачано")
    
    # Анализ
    print("🔍 Анализируем изображение...")
    mask, original_img = analyze_road_surface(image_path, model)
    
    # Статистика
    stats, primary_surface = calculate_road_statistics(mask, original_img)
    
    # Вывод результатов
    print("\n" + "="*50)
    print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА ДОРОЖНОГО ПОКРЫТИЯ")
    print("="*50)
    
    for surface_type, data in stats.items():
        print(f"{surface_type}: {data['pixel_percentage']:.1f}% пикселей")
    
    print(f"\n🎯 Основной тип покрытия: {primary_surface[0]}")
    print(f"   (занимает {primary_surface[1]['pixel_percentage']:.1f}% кадра)")
    
    # Визуализация
    visualize_results(image_path, mask, original_img)
    
    # Очистка временного файла
    if 'temp_path' in locals():
        import os
        os.remove(temp_path)
    
    return mask, stats, primary_surface

if __name__ == "__main__":
    mask, stats, primary_surface = main()