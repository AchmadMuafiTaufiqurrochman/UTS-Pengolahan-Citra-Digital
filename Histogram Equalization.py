import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Membaca gambar dan mengubahnya menjadi grayscale
image = Image.open('image.jpg').convert('L')  # Pastikan gambar berada di folder yang sama dengan script
image_array = np.array(image)

# Mendapatkan ukuran gambar
height, width = image_array.shape

# Histogram
histogram = np.zeros(256)
for i in range(width):
    for j in range(height):
        pixel = image_array[j, i]
        histogram[pixel] += 1

# Fungsi Histogram Equalization
cdf = histogram.cumsum()  # Cumulative distribution function
cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())  # Normalize CDF to range 0-255
cdf_normalized = cdf_normalized.astype('uint8')  # Convert to uint8 type for image processing

# Melakukan equalization
equalized_image_array = cdf_normalized[image_array]

# Mengonversi kembali ke format gambar
equalized_image = Image.fromarray(equalized_image_array)

# Menampilkan histogram dan gambar yang telah di-equalize
plt.figure(figsize=(12, 6))

# Plot histogram asli
plt.subplot(2, 2, 1)
plt.title("Original Histogram")
plt.hist(image_array.flatten(), bins=256, range=(0, 256), color='black')

# Plot histogram yang telah di-equalize
plt.subplot(2, 2, 2)
plt.title("Equalized Histogram")
plt.hist(equalized_image_array.flatten(), bins=256, range=(0, 256), color='black')

# Gambar asli
plt.subplot(2, 2, 3)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

# Gambar setelah histogram equalization
plt.subplot(2, 2, 4)
plt.title("Equalized Image")
plt.imshow(equalized_image, cmap='gray')

plt.show()
