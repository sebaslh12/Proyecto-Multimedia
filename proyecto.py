from PIL import Image
import numpy as np

img= np.asarray(Image.open('lena.jpg').convert('L'))
print type(img)
for x in np.nditer(img):
    print x
