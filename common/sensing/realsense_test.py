import numpy as np
from PIL import Image
im = Image.open("MP6/calibration/depth_0001.png")
A = np.asarray(im.getdata())
print(A.dtype)
print(A.shape)
print(A.min(),A.max())