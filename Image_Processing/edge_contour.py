"""This is a example of using edge detector to find contour"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random as rd

img = cv.imread("test1.png")

gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
blur_img = cv.GaussianBlur(gray_img, (5, 5), 2)

edge_img = cv.Canny(blur_img, 50, 200)

# kernel = np.ones((3,3),np.uint8)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
closing = cv.morphologyEx(edge_img, cv.MORPH_CLOSE, kernel)
contours, _ = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))

contour_rgb = cv.cvtColor(gray_img, cv.COLOR_GRAY2RGB)
for i in range(len(contours)):
    contour = contours[i]
    color = (rd.uniform(100, 255), rd.uniform(100, 255), rd.uniform(100, 255))
    area = cv.contourArea(contour)
    # print(f"{i} color {color} area {area}")
    if 50 < area < 1000:
        for point in contour:
            cv.circle(contour_rgb, (point[0][0], point[0][1]), 1, color, -1)
        M = cv.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv.circle(contour_rgb, (cX, cY), 5, color, -1)

fig, ax = plt.subplots(1, 3)

ax[0].set_title("gray_img")
ax[0].imshow(gray_img, cmap="gray")
ax[1].set_title("edge_img")
ax[1].imshow(edge_img, cmap="gray")
ax[2].set_title("closing")
ax[2].imshow(closing, cmap="gray")

fig2, ax2 = plt.subplots()

plt.title("contour")
ax2.set_title("contour")
ax2.imshow(contour_rgb)

plt.show()
