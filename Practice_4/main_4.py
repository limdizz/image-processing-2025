import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('sar_1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 1. Подберите параметры алгоритма разрастания регионов так, чтобы был выделен весь участок газона.
def homo_average(img, mask, point, T):
    av_val = img[mask > 0].sum() / np.count_nonzero(img[mask > 0])

    if abs(av_val - img[point]) <= T:
        return True

    return False


def region_growing(image, seed_point, homo_fun, r, T):
    mask = np.zeros(image_gray.shape, np.uint8)
    mask[seed_point] = 1
    count = 1
    while count > 0:
        count = 0
        local_mask = np.zeros(image_gray.shape, np.uint8)
        for i in range(r, image.shape[0] - r):
            for j in range(r, image.shape[1] - r):
                if mask[i, j] == 0 and mask[i - r:i + r, j - r: j + r].sum() > 0:
                    if homo_fun(image, mask, (i, j), T):
                        local_mask[i, j] = 1
        count = np.count_nonzero(local_mask)
        print(count)
        mask += local_mask

    return mask * 255


seed_point = (250, 250)
mask = region_growing(image_gray, seed_point, homo_average, 5, 20)

plt.imshow(mask, cmap="gray")
plt.title('Участок газона (треугольный, Average)')
plt.show()


# 2. Реализуйте вычисление критерия однородности, отличного от представленного. Сравните результаты.
def homo_median(img, mask, point, T):
    region_vals = img[mask > 0]

    med_val = np.median(region_vals)

    return abs(med_val - img[point]) <= T


seed_point = (250, 250)
mask = region_growing(image_gray, seed_point, homo_median, 5, 20)

plt.imshow(mask, cmap="gray")
plt.title('Участок газона (треугольный, Median)')
plt.show()

# 3. Применить алгоритм сегментации watershed+distance transform для задачи подсчета пальмовых деревьев.
image_palms = cv2.imread('palm_1.JPG')
plt.imshow(cv2.cvtColor(image_palms, cv2.COLOR_BGR2RGB))
plt.title('Исходное изображение')
plt.show()

image_palms_gray = cv2.cvtColor(image_palms, cv2.COLOR_BGR2GRAY)
image_palms_blurred = cv2.GaussianBlur(image_palms_gray, (13, 13), 0)

ret, thresh = cv2.threshold(image_palms_blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
plt.imshow(thresh, cmap="gray")
plt.title('Бинаризация')
plt.show()

dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, cv2.THRESH_BINARY)
sure_fg = sure_fg.astype(np.uint8)
plt.imshow(dist_transform, cmap="gray")
plt.title('Distance Transform')
plt.show()

ret, markers = cv2.connectedComponents(sure_fg)
markers = cv2.watershed(image_palms, markers)
plt.imshow(markers)
plt.title('Watershed')
plt.show()

num_palms = len(np.unique(markers)) - 1
print('Количество пальм:', num_palms)

segmented_image = image_palms.copy()
segmented_image[markers == -1] = [255, 0, 255]
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title(f'Сегментированное изображение, кол-во пальм: {num_palms}')
plt.show()
