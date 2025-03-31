
import numpy as np 
import pandas as pd
import warnings
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
import torchio as tio
import torch
import nibabel as nib
import SimpleITK as sitk
import os

from src.dataset.acdc import ACDC

warnings.filterwarnings("ignore")

class ACDCProcessed(ACDC): 
    def __init__(self, data_path,is_validation_set, is_testset): 
        self.data_path = data_path
        self.is_testset = is_testset
        self.is_validation_set = is_validation_set
        self.df = self.__make_dataframe() 

    def __getitem__(self, index):
        img_path = self.df.iloc[index, 0]
        gt_path = self.df.iloc[index, 1]

        obj_ids = torch.tensor([1,2,3]).to(torch.long)
        
        img = sitk.ReadImage(img_path)
        gt = sitk.ReadImage(gt_path)
        
        img = sitk.GetArrayFromImage(img)
        gt = sitk.GetArrayFromImage(gt)
        
        img = torch.tensor(img).unsqueeze(0)
        gt = torch.tensor(gt).unsqueeze(0)
        gt = (gt == obj_ids[:,None,None,None])

        
        if (self.is_testset): 
            return img, gt, str(img_path)

        if (self.is_validation_set): 
            num_layers = img.size(1)

            transform = tio.Compose([
                tio.Resize((num_layers, 352,352)), 
                tio.CropOrPad((10, 224, 224)),
                tio.ZNormalization()
            ])
            
            gt_transform = tio.Compose([
                tio.Resize((num_layers, 352,352)), 
                tio.CropOrPad((10, 224, 224)),
            ])
            
            img = transform(img)
            gt = gt_transform(gt).to(torch.long)
            return img, gt
            
            
        num_layers = img.size(1)

        transform = tio.Compose([
            tio.Resize((num_layers, 352,352)), 
            tio.CropOrPad((10, 224, 224)),
            tio.ZNormalization()
        ])
        
        gt_transform = tio.Compose([
            tio.Resize((num_layers, 352,352)), 
            tio.CropOrPad((10, 224, 224)),
        ])
        
        img = transform(img)
        gt = gt_transform(gt).to(torch.long)
        
        subject = tio.Subject(
            image = tio.ScalarImage(tensor = img), 
            mask = tio.LabelMap(tensor = gt)
        )

        augment = tio.Compose([
            tio.RandomFlip(axes=(1,2)), 
            tio.RandomElasticDeformation(max_displacement=1), 
            tio.RandomGamma()
        ])
        subject = augment(subject)
        return subject.image.tensor, subject.mask.tensor
        

    
    def __make_dataframe(self): 
        img_list = []
        gt_list = []
                
        for root, _, files in os.walk(self.data_path): 
            files = [os.path.join(root, file) for file in files]

            if (len(files) == 0): 
                continue

            patient_id = os.path.basename(root)

            ed_path = os.path.join(root, patient_id + "_ED"  + "_processed.nii.gz")
            ed_gt_path = os.path.join(root, patient_id + "_ED_gt" + "_processed.nii.gz")

            es_path = os.path.join(root, patient_id + "_ES" + "_processed.nii.gz")
            es_gt_path = os.path.join(root, patient_id + "_ES_gt" + "_processed.nii.gz")

            img_list.append(ed_path)
            img_list.append(es_path)
            gt_list.append(ed_gt_path)
            gt_list.append(es_gt_path)

        df = pd.DataFrame({
            "img": img_list, 
            "gt": gt_list
        }).sort_values(by="img")
        return df