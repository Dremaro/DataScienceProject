#Started the 29/01/2024 by Shayan and Dremaro

# Important methods run by the program:
#  - Otsu's trhesholding method
#  - Watershed segmentation




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


############################## Functions ######################################################
###############################################################################################


def getOneImage(image_path):
    # Open the image file
    im = PIL.Image.open(image_path)
    
    # Convert the image to a numpy array
    I = np.array(im)
    
    return I





############################## Main Code ######################################################
###############################################################################################
"""
TIP :
In Python strings, the backslash \ is an escape character, which is used to introduce special character sequences. For example, \n is a newline, and \t is a tab.

If you want to include a literal backslash in a string, you need to escape it by using two backslashes \\.

Alternatively, you can use raw strings, where backslashes are treated as literal characters. You can create a raw string by prefixing the string with r.
"""

image = getOneImage(r"photos\01\t000.tif")

t_image = image > filters.threshold_otsu(image)
ft_image = morphology.closing(binary_fill_holes(t_image), morphology.square(3)) # closing holes and removing small objects
ft_image = measure.label(ft_image) # labeling the objects to turn it into integers
dico_notes = measure.regionprops_table(ft_image,properties=('label', 'bbox','area','centroid')) # dictionary containing the properties
'''
'label': return the label of the region.
'bbox': return the bounding box of the region. The bounding box is represented as a tuple of the form (min_row, min_col, max_row, max_col).
'area': return the total area of the region.
'centroid': return the centroid of the region, represented as (row_centroid, col_centroid).
'''
nucs=DataFrame(dico_notes) # convert the dictionary into a dataframe
rp=measure.regionprops(ft_image) # c'est une liste d'objet (représentant chaque cellule identifiée) contenant les propriétés 

label=[]
bbox=[]
center=[]
area=[]
for r in rp: #on parcour la liste d'objet
    try:
        label.append(r.label) #on ajoute les propriétés de chaque cellule dans une liste
        bbox.append(r.bbox)
        center.append(r.centroid)
        area.append(r.area)
    except:
        print('whut')
        continue

nucs=DataFrame({'labels':label,
                'nucleus_bbox':bbox,
                'nucleus_center':center,
                'nucleus_area':area})  # on crée un dataframe avec les listes
nucs=nucs[nucs.nucleus_area>400]


nrow=np.random.choice(len(nucs))
bb=nucs['nucleus_bbox'].iloc[nrow]
fig, axes = plt.subplots(2, 2)
axes[0,0].imshow(image)
axes[0,1].imshow(t_image)
axes[1,0].imshow(ft_image)
axes[1,1].imshow(image[bb[0]:bb[2],bb[1]:bb[3]])
axes[0,0].set_title('Original')
axes[0,1].set_title('Tresholded')
axes[1,0].set_title('Filled and cleaned')
axes[1,1].set_title('Zoom Boxe')
plt.show()










