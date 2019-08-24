import cv2
import matplotlib.pyplot as plt

fname = 'test_images/straight_lines1.jpg'
img = cv2.imread(fname)

plt.imshow(img)
plt.show()