import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fname = 'test_images/straight_lines2.jpg'
image = mpimg.imread(fname)

plt.imshow(image)
plt.show()