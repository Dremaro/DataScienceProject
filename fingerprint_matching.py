## So here we start the code for fingerprint recognition ##
import matplotlib.pyplot as plt
#import SOCO_our_work.fingerprint_pipline as fp
import numpy as np
import cv2 as cv
from glob import glob
import ast  # ast is used to parse strings and find python expressions in them




datalink = {'input':0,
            'normalized':1,
            'segmented':2,
            'orientation':3,
            'gabor':4,
            'thin':5,
            'minutias':6,
            'singularities':7}

file_2match = './Z_fingerprint2mach/*'
files_2match = glob(file_2match)
compare_dtb = './Zd_fingerprint_dtb/*'
folders_dtb = glob(compare_dtb)
n = 2   # number of files per fingerprint folder

def extract_data(img_folder, l_content=[0]):
        l_input = []
        l_normalized = []
        l_segmented = []
        l_orientation = []
        l_gabor = []
        l_thin = []
        l_minutias = []
        l_singularities = []
        l_l = [l_input, l_normalized, l_segmented, l_orientation, l_gabor, l_thin, l_minutias, l_singularities]

        l_output = []
        files_pathnames = glob(img_folder+'/*')
        for i in range(8):
                if i in l_content:
                        f = open(files_pathnames[i], 'r')
                        l = ast.literal_eval(f.readline())
                        l_l[i] = l
                        f.close()
                        l_output.append(l_l[i])
        return l_output

print(extract_data(folders_dtb[0])[0][10])








