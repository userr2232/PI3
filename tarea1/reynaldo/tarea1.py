import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

img = cv.imread('parrot.jpg', cv.IMREAD_GRAYSCALE)
plt.axis("off")
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()

plt.axis("on")
plt.hist(img.ravel(),256,[0,256]);
plt.show()

c = 1
c = 255 / np.log1p(np.max(img)) 

log_img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, dtype = cv.CV_8U)
log_img = np.uint8(c * (np.log1p(log_img)))


plt.axis("off")
plt.imshow(cv.cvtColor(log_img, cv.COLOR_BGR2RGB))
plt.show()

plt.axis("on")
plt.hist(log_img.ravel(), 256, [0,256])
plt.show()

eq_img = cv.equalizeHist(img)
plt.imshow(cv.cvtColor(eq_img, cv.COLOR_BGR2RGB))
plt.show()

plt.axis("on")
plt.hist(eq_img.ravel(), 256, [0,256])
plt.show()

log_img_inverse = np.uint8(np.exp(log_img/c) - 1)
plt.imshow(cv.cvtColor(log_img_inverse, cv.COLOR_BGR2RGB))
plt.show()