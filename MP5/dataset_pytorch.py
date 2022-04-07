import numpy as np
import os
import sys
import csv
import random
import pickle
from dataset import load_images_dataset,get_region_of_interest

def sample_patch_dataset(dataset,N,patch_size=30):
    """Sample a bunch of (index,x,y) tuples over the dataset
    """
    #TODO: tune me / fill me in for Problem 2b
    retained_per_image = N//len(dataset)
    patch_radius = patch_size//2
    samples = []
    for image_idx,image in enumerate(dataset):
        color,depth,transform,grasp_attrs = image
        grasp_score = grasp_attrs['score']
        for i in range(retained_per_image):
            x,y = random.randint(patch_radius,color.shape[1]-1-patch_radius),random.randint(patch_radius,color.shape[0]-1-patch_radius)
            samples.append((image_idx,x,y))
    return samples


def generate_patch_dataset(N=10000,train_test_split=0.75):
    dataset = load_images_dataset('image_dataset')
    samples = sample_patch_dataset(dataset,N)
    try:
        os.mkdir('patch_dataset')
    except Exception:
        pass
    random.shuffle(samples)
    Ntrain = int(len(samples)*train_test_split)
    train_samples = samples[:Ntrain]
    test_samples = samples[Ntrain:]
    #write to patch_dataset folder
    metadata_fn = 'patch_dataset/train.csv'
    f_meta = open(metadata_fn,'w')
    f_meta.write(','.join(['id','image_index','x','y'])+'\n')
    for i,(img,x,y) in enumerate(train_samples):
        f_meta.write("%04d,%d,%d,%d\n"%(i,img,x,y))
    f_meta.close()
    metadata_fn = 'patch_dataset/test.csv'
    f_meta = open(metadata_fn,'w')
    f_meta.write(','.join(['id','image_index','x','y'])+'\n')
    for i,(img,x,y) in enumerate(test_samples):
        f_meta.write("%04d,%d,%d,%d\n"%(i,img,x,y))
    f_meta.close()
    print("Saved samples to patch_dataset/")
    

class PatchDataset:
    """Compatible with PyTorch dataloaders.  Extracts patches.""" 
    def __init__(self,dataset='image_dataset',samples='patch_dataset/train.csv',
        output_attrs=['score','opening','axis_heading','axis_elevation'],
        patch_size=30,downsample=1):
        if isinstance(dataset,str):
            dataset = load_images_dataset(dataset)
        if isinstance(samples,str):
            with open(samples,'r') as f:
                reader = csv.reader(f)
                header = None
                samples = []
                for row in reader:
                    if header is None:
                        header = row
                        assert len(header)==4
                    else:
                        samples.append([int(v) for v in row[1:4]])
                for (img,x,y) in samples:
                    assert img >= 0 and img < len(dataset),"Invalid image index?"
        self.dataset = dataset
        self.samples = samples
        self.output_attrs = output_attrs
        self.patch_size = patch_size
        self.downsample = downsample
        self.best_patch = None

    def __getitem__(self,idx):
        image_idx,x,y = self.samples[idx]
        patch_radius = self.patch_size//2
        color,depth,transform,grasp_attrs = self.dataset[image_idx]
        if self.downsample is not None:
            color = color[::self.downsample,::self.downsample,:]
            depth = depth[::self.downsample,::self.downsample]
            x = x//self.downsample
            y = y//self.downsample
            grasp_attrs = dict((k,v[::self.downsample,::self.downsample]) for (k,v) in grasp_attrs.items())
    
        roi = (y-patch_radius,y+patch_radius,x-patch_radius,x+patch_radius)
        patch1 = get_region_of_interest(color,roi)
        patch2 = get_region_of_interest(depth,roi)
        if self.best_patch is None or grasp_attrs['score'][y,x]  > self.best_patch[0]:
            self.best_patch = (grasp_attrs['score'][y,x],image_idx,x,y,patch1,patch2)
        #Note: Pytorch expects the color patch to have dimension 3xwxh
        return {'color':patch1,'depth':patch2,'output':np.array([grasp_attrs[attr][y,x] for attr in self.output_attrs])}
    
    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    generate_patch_dataset()
    