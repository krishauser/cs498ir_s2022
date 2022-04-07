import numpy as np
from klampt.io import loader
import random
from PIL import Image
import csv
import os

DEPTH_SCALE = 8000

ALL_GRASP_ATTRIBUTES = ['score', 'axis_heading','axis_elevation','opening']

def load_images_dataset(folder,attributes=ALL_GRASP_ATTRIBUTES):
    """Loads an image dataset from a folder.  Result is a list of
    (color,depth,camera_transform,grasp_attr) tuples.
    """
    metadatafn = os.path.join(folder,'metadata.csv')
    rows = []
    with open(metadatafn, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row)
    print("Read",len(rows)-1,"training images")
    if len(rows)<=1:
        raise RuntimeError("Hey, no rows read from metadata file?")
    cols = dict((v,i) for (i,v) in enumerate(rows[0]))
    dataset = []
    for i in range(1,len(rows)):
        color = np.asanyarray(Image.open(os.path.join(folder,rows[i][cols['color_fn']])))
        depth = np.asanyarray(Image.open(os.path.join(folder,rows[i][cols['depth_fn']])))*(1.0/DEPTH_SCALE)
        assert len(color.shape)==3 and color.shape[2]==3
        gripper_base_fn = os.path.join(folder,rows[i][cols['grasp_fn']])
        base,ext = os.path.splitext(gripper_base_fn)
        grasp_attrs = dict()
        for attr in attributes:
            grasp_channel = np.asanyarray(Image.open(base + '_' + attr + ext))*(1.0/255.0)
            grasp_attrs[attr] = grasp_channel
        dataset.append((color,depth,loader.read('RigidTransform',rows[i][cols['view_transform']]),grasp_attrs))
    return dataset

def get_region_of_interest(image,roi,fill_value='average'):
    """
    Retrieves a subset of an image, being friendly to image boundaries.

    Args
        image (np.ndarray): has at least 2 dimensions
        roi (tuple): values (i1,i2,j1,j2) defining the patch boundaries
            [i1:i2,j1:j2] (note: non-inclusive of i2 and j2).
        fillvalue: a value to fill in when the roi expands beyond the
            boundaries. Can also be 'average'
    """
    i1,i2,j1,j2 = roi
    if i1 < 0 or i2 > image.shape[0] or j1 < 0 or j2 > image.shape[1]:
        subset = image[max(i1,0):min(i2,image.shape[0]),
                    max(j1,0):min(j2,image.shape[1])]
        paddings = [(max(-i1,0),max(i2-image.shape[0],0)),
                    (max(-j1,0),max(j2-image.shape[1],0))]
        if len(image.shape) > 2:
            paddings += [(0,0)]*(len(image.shape)-2)
        if fill_value == 'average':
            if len(subset)==0:
                fill_value = 0
            else:
                fill_value = np.average(subset)
        res = np.pad(subset,tuple(paddings),mode='constant',constant_values=(fill_value,))
        assert res.shape[0] == i2-i1,"Uh... mismatch? {} vs {} (roi {})".format(res.shape[0],i2-i1,roi)
        assert res.shape[1] == j2-j1,"Uh... mismatch? {} vs {} (roi {})".format(res.shape[1],j2-j1,roi)
        return res
    else:
        return image[i1:i2,j1:j2]

def set_region_of_interest(image,roi,value):
    """Sets a patch of an image to some value, being tolerant to the
    boundaries of the image.
    """
    i1,i2,j1,j2 = roi
    image[max(i1,0):min(i2,image.shape[0]),max(j1,0):min(j2,image.shape[1])] = value


def make_patch_dataset(dataset,predicted_attr='score',patch_size=30):
    """Create a matrix (X,y) consisting of flattened feature vectors
    from the image dataset.
    """
    #TODO: tune me / fill me in for Problem 1a
    samples_per_image = 100
    patch_radius = patch_size//2
    A = []
    b = []
    for image in dataset:
        color,depth,transform,grasp_attrs = image
        #you might want these?
        color_gradient_x = np.linalg.norm(color[1:,:,:]-color[:-1,:,:],axis=2)
        color_gradient_y = np.linalg.norm(color[:,1:,:]-color[:,:-1,:],axis=2)
        depth_gradient_x = depth[1:,:]-depth[:-1,:]
        depth_gradient_y = depth[:,1:]-depth[:,:-1]
        output = grasp_attrs[predicted_attr]
        scores = []
        for i in range(samples_per_image):
            x,y = random.randint(patch_radius,color.shape[1]-1-patch_radius),random.randint(patch_radius,color.shape[0]-1-patch_radius)
            
            roi = (y-patch_radius,y+patch_radius,x-patch_radius,x+patch_radius)
            patch1 = get_region_of_interest(color,roi).flatten()
            patch2 = get_region_of_interest(depth,roi).flatten()
            A.append(np.hstack((patch1,patch2)))
            assert len(A[-1].shape)==1
            assert A[-1].shape == A[0].shape
            b.append(output[y,x])
    return np.vstack(A),np.array(b)
