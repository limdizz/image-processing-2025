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

# Canny
edges = cv2.Canny(image_gray, 100, 200)
plt.imshow(edges, cmap="gray")
plt.title("Исходное изображение: Сanny")
plt.show()

# Преобразование Хафа
canny = cv2.Canny(image_gray, 50, 150, apertureSize=3)
lines = cv2.HoughLines(canny, 1, np.pi / 180, 175)
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1300 * (-b)), int(y0 + 800 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 560 * a))
        cv2.line(image, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
plt.imshow(image)
plt.title("Исходное изображение: преобразование Хафа")
plt.show()
