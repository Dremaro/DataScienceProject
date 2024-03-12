## So here we start the code for fingerprint recognition ##
import matplotlib.pyplot as plt
#import SOCO_our_work.fingerprint_pipline as fp
import numpy as np
import cv2 as cv
from glob import glob
from math import *
import random as rd
import ast  # ast is used to parse strings and find python expressions in them




#####################################    FUNCTIONS     #######################################################################################################
##############################################################################################################################################################

def extract_data(img_folder, l_content=[3, 5, 6, 7, 8]):
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
                        l = np.loadtxt(files_pathnames[k])
                        if i == 6 or i == 7:  # minutias are composed of two files... then singularities is number 8 in folder, but still 7 in l_outputs
                                l_minutias.append(l)
                        else:
                                l_outputs[i] = l
        return l_outputs



def rotation_minutiae(minutiae, angle, rotation_center, show = False):
        x = minutiae[1][:, 0]
        y = minutiae[1][:, 1]
        if show:
                plt.plot(x, y, 'ro')
                plt.gca().invert_yaxis()
        x1 = x - rotation_center[0]
        y1 = y - rotation_center[1]
        # multplication by the rotation matrix
        x_rot = x1*cos(angle) - y1*sin(angle) + rotation_center[0]
        y_rot = x1*sin(angle) + y1*cos(angle) + rotation_center[1]
        x_pixel_coord = np.array([int(i) for i in x_rot])
        y_pixel_coord = np.array([int(i) for i in y_rot])
        type_minutiae = minutiae[0][:, 2]
        if show:
                plt.plot(x_pixel_coord, y_pixel_coord, 'bo')
                plt.show()
        
        return np.column_stack((x_pixel_coord, y_pixel_coord, type_minutiae)), np.column_stack((x_rot, y_rot))

def translation_minutiae(minutiae, translation):
        x = minutiae[1][:, 0]
        y = minutiae[1][:, 1]
        x_pixel_coord = x + translation[:,0]
        y_pixel_coord = y + translation[:,1]
        type_minutiae = minutiae[0][:, 2]
        return np.column_stack((x_pixel_coord, y_pixel_coord, type_minutiae)), np.column_stack((x_pixel_coord, y_pixel_coord))




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

def distance(p1, p2):
        return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def remove_border_minutiae(minutiae, thin, pourcentage = 4/5, n_sectors = 20, plot = False):
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
        l_theta = np.linspace(0,360,n_sectors)[:-1]*pi/180
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



def ressemblance(cloud1,cloud2,threshold, sensibility = 5):
        """Gives a pourcentage (between 0&1) of ressemblance between two clouds

        Args:
            cloud1 (array): cloud of points 1, the reference
            cloud2 (array): cloud of points 2
            threshold (float): distance under which points are considered to match
            sensibility (int, optional): number of match above which fingerprint is considered to match . Defaults to 5.

        Returns:
            _type_: _description_
        """
        points_proches = []
        for point in cloud1:
                k_point, dist = find_closest_point(point, cloud2)
                if dist < threshold:
                        points_proches.append(k_point)
        
        n_res = len(points_proches)
        n_points = min(len(cloud1),len(cloud2))
        ratio = (n_res/n_points)*sensibility

        if ratio > 1:
                ratio = 1
        l_matches = [0,1,2,3,4,5]
        return ratio

def find_closest_point(point, cloud):
        distance_min = distance(point, cloud[0])
        k_min = 0
        for k, point_1 in enumerate(cloud):
                distance_1 = distance(point, point_1)
                if distance_1 < distance_min:
                        distance_min = distance_1
                        k_min = k
                elif distance_1 == distance_min:
                        k_min = rd.choice([k, k_min])
        return k_min, distance_min

def compare_minutiaes(cloud_ref, cloud1, method):
        # Initialisation of variables and choice of reference cloud point.
        if len(cloud_ref) < len(cloud1):
                cloud_ref, cloud1 = cloud1, cloud_ref
        cloud_ref = np.array(cloud_ref)
        cloud1 = np.array(cloud1)
        points_ref = cloud_ref[:, :2]
        points_1 = cloud1[:, :2]
        centre = np.mean(points_ref, axis = 0)

        # find the closest point in cloud1 for each point in cloud_ref
        l_closest_points = []
        for point_ref in points_ref:
                k_1, distance_min = find_closest_point(point_ref, points_1)
                l_closest_points.append([points_1[k_1], point_ref, distance_min, k_1])
        l_closest_points = np.array(l_closest_points)

        # create the translation that reduces the minimal distances
        if method == 'translation':
                translation = np.mean(l_closest_points[:, 0] - l_closest_points[:, 1], axis = 0)

        # create the rotation the reduces the minimal distances
        elif method == 'rotation':
                l_angles_rotation = []
                for points in l_closest_points[:, :2]:
                        l = distance(points[0], centre)
                        norme = points[2]
                        c1 = points[0][0]
                        c2 = points[0][1]
                        l_ortho = [-c1/l, c2/l]
                        vecteur = [points[1][0]-points[0][0], points[1][1]-points[0][1]]
                        sp = vecteur[0]*l_ortho[0] + vecteur[1]*l_ortho[1]
                        l_angles_rotation.append(atan2(sp,l))
                l_angles_rotation = np.array(l_angles_rotation)
                angle = np.mean(l_angles_rotation)
        else :
                print("erreur, this method doesn't exist, ask for 'translation' or 'rotation' please")
        
        return translation, angle, l_closest_points







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


[l_input, l_normalized, l_segmented, l_orientation, l_gabor, l_thin, l_minutias, l_singularities] = extract_data(folders_dtb[5])
minutiae = [remove_border_minutiae(l_minutias[0], l_thin, pourcentage = 4/5, n_sectors=30, plot = True), remove_border_minutiae(l_minutias[1], l_thin, pourcentage = 4/5, n_sectors=30)]

print(minutiae)
print(rotation_minutiae(minutiae, pi/20, [0,0]))





