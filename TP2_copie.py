
# **Aims** Apply the concepts and algorithms seen to segment the cells from he screen of interest, to obtain masks with each individual cell outlined

# <img src="orig.png" width="600" />
# <img src="segmented_lab.png" width="600" />
# <img src="segmented.png" width="600" />
# 


import re
import random
import omero
from pandas import DataFrame, merge,concat
import numpy as np

from skimage import segmentation,filters,morphology,measure,feature
from skimage.color import label2rgb
from scipy.ndimage.morphology import binary_fill_holes, binary_closing
from skimage.util import img_as_ubyte,img_as_float
from skimage.exposure import rescale_intensity

from scipy.stats import kurtosis, skew

import omero
from omero.gateway import BlitzGateway

import matplotlib.pyplot as plt





# Connect to the server
def connect(hostname, username, password):
    conn = BlitzGateway(username, password,
                        host=hostname, secure=True)
    print("Connected: %s" % conn.connect())
    conn.c.enableKeepAlive(60)
    return conn

def getBulkAnnotationAsDf(screenID, conn):
    ofId = None
    sc = conn.getObject('Screen', screenID)
    for ann in sc.listAnnotations():
        if isinstance(ann, omero.gateway.FileAnnotationWrapper):
            if (ann.getFile().getName() == 'bulk_annotations'):
                if (ann.getFile().getSize()> 1476250900): #about 140Mb?
                    print("that's a big file...")
                    return None
                ofId = ann.getFile().getId()
                break

    if ofId is None:
        return None

    original_file = omero.model.OriginalFileI(ofId, False)

    table = conn.c.sf.sharedResources().openTable(original_file)
    count = 0
    try:
        rowCount = table.getNumberOfRows()

        column_names = []
        pattern = re.compile("Phenotype \d+$")
        for col in table.getHeaders():
            column_names.append(col.name)
            if pattern.match(col.name) is not None:
                count = count + 1

        black_list = []
        column_indices = []
        for column_name in column_names:
            if column_name in black_list:
                continue
            column_indices.append(column_names.index(column_name))

        table_data = table.slice(column_indices, None)
    finally:
        table.close()

    data = []
    for index in range(rowCount):
        row_values = [column.values[index] for column in table_data.columns]
        data.append(row_values)

    dfAnn = DataFrame(data)
    dfAnn.columns = column_names
    return dfAnn, count

def getOneImage(conn,weid,pos=None):
    we = conn.getObject('Well',weid)
    if pos==None:
        pos=random.choice(range(we.countWellSample()))  # one random field of that well
        
    im = we.getImage(pos)   
    pix = im.getPrimaryPixels()
    I1=pix.getPlane(0,0,0)
    I2=pix.getPlane(0,1,0)
    I3=pix.getPlane(0,2,0)
    I4=pix.getPlane(0,3,0)
    I5=pix.getPlane(0,4,0)
    I=np.stack([I1,I2,I3,I4,I5],2)
    return I




host = "ws://idr.openmicroscopy.org/omero-ws"
username = "public"
password = "public"
screenId = 1751 #idr0033-rohban-pathways/screenA

# Connect to the server
conn = connect(host, username, password)

# Downloading the annotation file for the whole screen
# as a panda DataFrame
dfAnn, phenotype_count = getBulkAnnotationAsDf(screenId, conn)
dfAnn.head()

weid=dfAnn[dfAnn['Control Type']=='negative control'].sample()['Well'].values
I=getOneImage(conn,weid)





plt.set_cmap('gray')

zoomx=[100,500]
zoomy=[100,500]
fig,ax=plt.subplots(1,5,figsize=(30,10))
ax[0].imshow(I[:,:,0][zoomx[0]:zoomx[1],zoomy[0]:zoomy[1]])
ax[1].imshow(I[:,:,1][zoomx[0]:zoomx[1],zoomy[0]:zoomy[1]])
ax[2].imshow(I[:,:,2][zoomx[0]:zoomx[1],zoomy[0]:zoomy[1]])
ax[3].imshow(I[:,:,3][zoomx[0]:zoomx[1],zoomy[0]:zoomy[1]])
ax[4].imshow(I[:,:,4][zoomx[0]:zoomx[1],zoomy[0]:zoomy[1]])

for x in ax.ravel():
    x.axis("off")
    




I[:,:,0]


# ## Segment the nucleis
# 
# 1. Starting from the DAPI image (the first, index 0), buils a binary image with nucleis only.
# 
# 2. Use skimage.measure.label and skimage.measure.regionprops to extract a data frame of nucleis objects with preliminary geometric measurement. Use simple manual threshold to remove outliers objects, for example those with too small areas
# 

#region : what I tried during the TP

# Survival notes !!!!
# this TP is to link with the python code sections in the bio583_2_slides course
I_s = np.copy(I)
treshold = 1000
fig,ax=plt.subplots(1,5,figsize=(30,10))

for i in range(0,5):
    I_view = I_s[:,:,i]
    for l in range(len(I_view)):
        ligne = I_view[l]
        for c in ligne:
            el = ligne[c]
            if el > treshold:
                el = 5000
            else:
                el = 0
    ax[i].imshow(I_s[:,:,i][zoomx[0]:zoomx[1],zoomy[0]:zoomy[1]])

noyaux_I = I[:,:,0] #tableaux des noyaux
#plt.plot c'est pour les graphs
#plt.imshow() c'est pour les images

def treshold_on_image(treshold, image):
    M = max(max(ligne_pixel) for ligne_pixel in image)
    return [[M if pix > treshold else 0 for pix in ligne] for ligne in image]
    
noyaux_I = treshold_on_image(900, noyaux_I)
            
            
print("ça calcule l'image...")
plt.imshow(noyaux_I, cmap='gray')
plt.colorbar()

#1080*1080
#endregion


#22
#initial seg: plain thresholding of dapi channel
otsu=filters.threshold_otsu(I[:,:,0])
dapiSeg=measure.label(I[:,:,0]>otsu)

plt.figure(figsize=(15,15))
plt.imshow(dapiSeg)

#23
rp=measure.regionprops_table(dapiSeg,properties=('label', 'bbox','area','moments_hu','centroid','bbox'))
nucs=DataFrame(rp)

#24
nucs.head()

#25
rp=measure.regionprops(dapiSeg)
#Also measure.regionprops_table()

label=[]
bbox=[]
center=[]
area=[]
eccentricity=[]
equivalent_diameter=[]
moments_hu=[]
perimeter=[]
solidity=[]
for r in rp:
    try:
        label.append(r.label)
        bbox.append(r.bbox)
        center.append(r.centroid)
        area.append(r.area)
        eccentricity.append(r.eccentricity)
        equivalent_diameter.append(r.equivalent_diameter)
        moments_hu.append(r.moments_hu)
        perimeter.append(r.perimeter)
        solidity.append(r.solidity)
    except:
        print('whut')
        continue

nucs=DataFrame({'labels':label,'nucleus_bbox':bbox,'nucleus_center':center,'nucleus_area':area,'nucleus_eccentricity':eccentricity,'nucleus_equivalent_diameter':equivalent_diameter,'nucleus_moments_hu':moments_hu,'nucleus_perimeter':perimeter,'nucleus_solidity':solidity})



# 
# ## Segment cells
# 
# 3. Similarly, try to segment the cell body using the third channel (index 2)
# 
# 3. With nucleis as seed, use the watershed segmentation algorithm (skimage.morphology.watershed) to segment single cells. (use the 'mask' option.)
# 
# 4. Again use regionprops to extract objects and geometric features and clean them.
# 
# 3. use skimage.segmentation.clear_border to remove object touching the border (optional)
# 
# 4. Manually look at segmented objects for selected phenotypes to check that the segmentation is robust.
# 
# 


