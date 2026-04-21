import matplotlib.pyplot as plt
from geodeep import segment

# Получаем маску
mask = segment("satellite_image.tiff", "roads")

# Показываем результат
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Оригинальное изображение")
# Если хотите показать оригинал - нужно загрузить его отдельно

plt.subplot(1, 2, 2)
plt.title("Маска дорог (белое = дороги)")
plt.imshow(mask, cmap='gray')
plt.show()