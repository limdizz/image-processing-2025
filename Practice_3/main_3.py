import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

image = cv2.imread('sar_3.jpg')
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image)
plt.title("Исходное изображение с ч/б фильтром")
plt.show()

# Точечная бинаризация
bin_img = copy.deepcopy(image_gray)
T = 50
bin_img[image_gray < T] = 0
bin_img[image_gray >= T] = 255
plt.imshow(bin_img, cmap="gray")
plt.title("Исходное изображение: точечная бинаризация")
plt.show()

# Бинаризация Отсу
_, th2 = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(th2, cmap="gray")
plt.title("Исходное изображение: бинаризация Отсу")
plt.show()

# Адаптивная бинаризация
th3 = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 71, 21)
plt.imshow(th3, cmap="gray")
plt.title("Исходное изображение: адаптивная бинаризация")
plt.show()

# Оператор Собеля
scale = 1
delta = 0
ddepth = cv2.CV_16S
grad_x = cv2.Sobel(image_gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(image_gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
plt.imshow((grad_x - grad_x.min()) * 255, cmap="gray")
plt.title("Исходное изображение: оператор Собеля (градиент по x)")
plt.show()

plt.imshow((grad_y - grad_y.min()) * 255, cmap="gray")
plt.title("Исходное изображение: оператор Собеля (градиент по y)")
plt.show()

grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0.0)  # mean value between
plt.imshow((grad - grad.min()) * 255, cmap="gray")
plt.title("Исходное изображение: оператор Собеля")
plt.show()

# Преобразование Хафа
image_blur = cv2.GaussianBlur(image_gray, (5,5), 0)
canny = cv2.Canny(image_blur, 50, 150, apertureSize=3)
plt.imshow(canny, cmap="gray")
plt.title("Исходное изображение: Сanny")
plt.show()

lines = cv2.HoughLines(canny, 1, np.pi / 180, 120)

if lines is not None:
    max_line = None
    max_length = 0
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        length = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
        if length > max_length:
            max_length = length
            max_line = (pt1, pt2)
    if max_line:
        cv2.line(image, max_line[0], max_line[1], (0, 0, 255), 5, cv2.LINE_AA)

plt.imshow(image)
plt.title("Исходное изображение: самый протяжённый участок")
plt.show()
