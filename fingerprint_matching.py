## So here we start the code for fingerprint recognition ##
import matplotlib.pyplot as plt
#import SOCO_our_work.fingerprint_pipline as fp
import numpy as np
import cv2 as cv
from glob import glob
from math import *
import ast  # ast is used to parse strings and find python expressions in them




#####################################    FUNCTIONS     #######################################################################################################
##############################################################################################################################################################

def extract_data(img_folder, l_content=[3, 5, 6]):
        l_input = []
        l_normalized = []
        l_segmented = []
        l_orientation = []
        l_gabor = []
        l_thin = []
        l_minutias = []
        l_singularities = []
        l_outputs = [l_input, l_normalized, l_segmented, l_orientation, l_gabor, l_thin, l_minutias, l_singularities]

        files_pathnames = glob(img_folder+'/*')
        print(files_pathnames)
        for i in range(len((l_outputs))):
                if i in l_content:
                        k = l_content.index(i)
                        # f = open(files_pathnames[i], 'r')
                        # l = ast.literal_eval(f.readline())
                        # l_l[i] = l
                        # f.close()
                        l = np.loadtxt(files_pathnames[k])
                        l_outputs[i] = l
        return l_outputs

def distance(p1, p2):
        return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def remove_border_minutiae(minutiae, thin):
        y_m = minutiae[:][0]  # y is the row in order to be the ordinate    (i)
        x_m = minutiae[:][1]  # x is the colomn in order to be the abscissa (j)

        # compute the center of the fingerprint with thin image
        i_sum, j_sum = 0, 0
        n_i, n_j = 0, 0
        for i in range(len(thin)):
                for j in range(len(thin[0])):
                        if thin[i][j] == 0:
                                i_sum += i
                                j_sum += j
                                n_i += 1
                                n_j += 1
        i_center = int(i_sum/n_i)
        j_center = int(j_sum/n_j)
        x_m = x_m - j_center    # centering the minutiae
        y_m = y_m - i_center


        l_theta = np.linspace(0,360,8)[:-1]
        delta = theta[1]/2
        l_minutiae_inside_cone = []
        for theta in l_theta:
                coeff_minus_delta = sin(theta-delta)/cos(theta-delta)
                coeff_plus_delta = sin(theta+delta)/cos(theta+delta)

                for i,j in zip(x_m, y_m):
                        coeff = i/j
                        if coeff_minus_delta < coeff < coeff_plus_delta:
                                l_points_inside_cone.append([i,j])












###################################    GLOBAL VARIABLES     ##################################################################################################
##############################################################################################################################################################

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
n = 3   # number of files per fingerprint folder




#########################################    MAIN     ########################################################################################################
##############################################################################################################################################################


[l_input, l_normalized, l_segmented, l_orientation, l_gabor, l_thin, l_minutias, l_singularities] = extract_data(folders_dtb[0])
print(l_thin)
remove_border_minutiae(l_minutias, l_thin)









