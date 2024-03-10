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

def is_point_in_sector(x, y, theta1, theta2):
    """
    Determines if a point is inside a sector defined by two angles.

    The function converts the point to polar coordinates and checks if its angle is between the two provided angles.
    The angles are normalized to be between 0 and 2*pi. The sector is assumed to be 
    defined in the counterclockwise direction from theta1 to theta2.

    Args:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        theta1 (float): The angle defining the start of the sector, in radians.
        theta2 (float): The angle defining the end of the sector, in radians.

    Returns:
        bool: True if the point is in the sector, False otherwise.
    """
    # Convert the point to polar coordinates
    r, theta = sqrt(x**2 + y**2), atan2(y, x)

    # Normalize the angles to be between 0 and 2*pi
    theta = (theta + 2*pi) % (2*pi)
    theta1 = (theta1 + 2*pi) % (2*pi)
    theta2 = (theta2 + 2*pi) % (2*pi)

    # Check if the point's angle is between theta1 and theta2
    if theta1 < theta2:
        return theta1 <= theta <= theta2
    else:  # The sector crosses the x-axis
        return theta >= theta1 or theta <= theta2

def plot_line(x1, y1, x2, y2, color='k'):
        plt.plot([x1, x2], [y1, y2], color)

def distance(p1, p2):
        return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def remove_border_minutiae(minutiae, thin, pourcentage = 4/5, plot = True):
        """
        Removes minutiae that are too close to the border of a fingerprint image.

        - minutiae (numpy.ndarray) : A 2D array where each row represents a minutia (x, y, type).
        - thin (numpy.ndarray) : A 2D binary array representing the skeletonized fingerprint image.
        - pourcentage (float, optional) : A threshold value used to determine which minutiae are too close to the border. 
                           Minutiae that are closer to the border than this percentage of the maximum distance 
                           in a sector are removed. Defaults to 4/5.
        - plot (bool, optional) : Whether to plot the minutiae before and after removal. Defaults to True.

        Return
        - (numpy.ndarray) A 2D array of the remaining minutiae after border minutiae have been removed.
        """
        minutiae_without_border = []

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
        # Centering the data around the center of the fingerprint
        i_center = int(i_sum/n_i)
        j_center = int(j_sum/n_j)
        minutiae[:, 0] = minutiae[:, 0] - i_center
        minutiae[:, 1] = minutiae[:, 1] - j_center
        
        # define the number of sectors and the half angle of each sector
        l_theta = np.linspace(0,360,20)[:-1]*pi/180
        delta = l_theta[1]/2

        if plot:
                x = minutiae[:,0]
                y = minutiae[:,1]
                plt.plot(x, y, 'ro')
                plt.gca().invert_yaxis()
                r = 200
        
        # running through each sector to find the minutiae inside the sector
        for theta in l_theta:
                l_minutiae_inside_cone = []
                angle_min = theta-delta
                angle_sup = theta+delta
                if plot:
                       plt.plot([0,r*cos(angle_min)], [0, r*sin(angle_min)])
                       plt.plot([0,r*cos(angle_sup)], [0, r*sin(angle_sup)])

                # finding the minutiae inside the sector
                for k in range(len(minutiae)):
                        [i, j] = minutiae[k][:2]
                        if is_point_in_sector(i, j, angle_min, angle_sup):
                                l_minutiae_inside_cone.append(minutiae[k])
                l_minutiae_inside_cone = np.array(l_minutiae_inside_cone)
                
                # remove minutiae too close to the border
                if len(l_minutiae_inside_cone) > 0:
                        distances_in_cone = np.array([distance([0,0], [i,j]) for [i,j] in l_minutiae_inside_cone[:, :2]])
                        distance_max = max(distances_in_cone)
                        threshold = pourcentage * distance_max
                        l_minutiae_inside_cone = l_minutiae_inside_cone[distances_in_cone < threshold] # remove minutiae too close to the border
                
                # add the minutiae inside the sector to the list of minutiae without border
                l_minutiae_inside_cone = list(l_minutiae_inside_cone)
                minutiae_without_border = minutiae_without_border + l_minutiae_inside_cone

        if plot:
                x_borderless = [i[0] for i in minutiae_without_border]
                y_borderless = [i[1] for i in minutiae_without_border]
                plt.plot(x_borderless, y_borderless, 'bo')
                plt.show()

        return np.array(minutiae_without_border)





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


[l_input, l_normalized, l_segmented, l_orientation, l_gabor, l_thin, l_minutias, l_singularities] = extract_data(folders_dtb[2])
remove_border_minutiae(l_minutias, l_thin, pourcentage = 4/5)









