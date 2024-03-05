## So here we start the code for fingerprint recognition ##
import matplotlib.pyplot as plt
#import SOCO_our_work.fingerprint_pipline as fp
import numpy as np
import cv2 as cv
from glob import glob


files = './Z_fingerprint2mach/*'
files = glob(files)
print(len(files))
if len(files) == 2:
        tocheckFP_path = files[0]
        tocheckFP = cv.imread(tocheckFP_path,0)
else:
        print('There are mutliple images in the input directory, please provide a single image to check.')
        exit(1)


compare_dtb = './Zd_fingerprint_dtb/*'

plt.imshow(tocheckFP, cmap='gray')
plt.show()




