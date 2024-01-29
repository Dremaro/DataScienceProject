#Started the 29/01/2024 by Shayan and Dremaro



############################## Import ##########################################
################################################################################
import re
import random
from pandas import DataFrame, merge,concat
import numpy as np

from skimage import segmentation,filters,morphology,measure,feature
from skimage.color import label2rgb
from scipy.ndimage.morphology import binary_fill_holes, binary_closing
from skimage.util import img_as_ubyte,img_as_float
from skimage.exposure import rescale_intensity

from scipy.stats import kurtosis, skew

import matplotlib.pyplot as plt

import PIL



def getOneImage(image_path):
    # Open the image file
    im = PIL.Image.open(image_path)
    
    # Convert the image to a numpy array
    I = np.array(im)
    
    return I

"""
TIP :
In Python strings, the backslash \ is an escape character, which is used to introduce special character sequences. For example, \n is a newline, and \t is a tab.

If you want to include a literal backslash in a string, you need to escape it by using two backslashes \\.

Alternatively, you can use raw strings, where backslashes are treated as literal characters. You can create a raw string by prefixing the string with r.
"""


image = getOneImage(r"photos\01\t000.tif")
plt.imshow(image)
plt.show()



