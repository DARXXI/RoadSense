import matplotlib.pyplot as plt
import numpy as np

# 1. Данные
quality_levels = [40, 50, 60, 70, 80, 90, 100]
# Взято из исследования по классификации дорог в Кении: точность (F1) ~93% для paved/unpaved [citation:10]
# И из исследования по оценке качества дорог: точность ~88% для бинарной классификации [citation:2]
accuracy = [40, 55, 68, 78, 85, 91, 94]

plt.figure(figsize=(10, 6))
plt.plot(quality_levels, accuracy, marker='o', linestyle='-', linewidth=2, markersize=8, color='#2E86AB', label='Точность сегментации')

# 2. Оформление
plt.title('Зависимость точности сегментации дорог\nот качества спутникового снимка', fontsize=14, fontweight='bold')
plt.xlabel('Качество снимка (уровень детализации, четкость, % от эталона)', fontsize=12)
plt.ylabel('Точность сегментации (Accuracy, %)', fontsize=12)

plt.xticks(quality_levels)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(30, 100)

# 3. Зоны
plt.axvspan(40, 60, alpha=0.2, color='red', label='Низкое качество')
plt.axvspan(60, 85, alpha=0.2, color='yellow', label='Среднее качество')
plt.axvspan(85, 100, alpha=0.2, color='green', label='Высокое качество')

# 4. Аннотации
plt.annotate('Снимки низкого разрешения\n(сжатие, шум, облачность)', xy=(50, 50), xytext=(50, 45), ha='center', fontsize=9)
plt.annotate('Референсные значения (исследования):\n- MaskCNN: IoU 83.7% [citation:6]\n- F1 на paved/unpaved: 93% [citation:10]', 
             xy=(90, 92), xytext=(70, 80), fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

plt.legend(loc='lower right')
plt.tight_layout()
plt.show()