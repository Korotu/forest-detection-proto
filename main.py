# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from PIL import Image, ImageFilter
import cv2
import numpy as np
from matplotlib import pyplot as plt


def find_outline_pillow(img):
    # Use a breakpoint in the code line below to debug your script.
    image = Image.open(img)
    image = image.filter(ImageFilter.FIND_EDGES)
    image.save('outline.png')


def find_outline_cv(img):

    # grayscale
    rgb_img = cv2.imread(img)
    height, width, _ = rgb_img.shape
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("image", gray_img)

    # gray_img.show()
    # # extra padding
    # white_padding = np.zeros((50, width, 3))
    # white_padding[:, :] = [255, 255, 255]
    # rgb_img = np.row_stack((white_padding, rgb_img))

    # invert greyscale, then add black padding
    gray_img = 255 - gray_img
    cv2.imwrite('inverted.png', gray_img)
    gray_img[gray_img <= 245] = 0
    cv2.imwrite('inverted_blacked.png', gray_img)

    gray_img[gray_img > 10] = 255
    gray_img = 255 - gray_img

    black_padding = np.zeros((50, width))
    gray_img = np.row_stack((black_padding, gray_img))
    cv2.imwrite('image_polygon.png', gray_img)

    # morphological closing
    kernel = np.ones((30, 30), np.uint8)
    closing = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(closing, 100, 200)
    cv2.imwrite('outline.png', edges)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    find_outline_cv('test_region.png')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
