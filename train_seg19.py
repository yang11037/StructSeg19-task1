#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import SimpleITK as sitk
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms

import vnet


# class DataLoader:
#     def __init__(self, dataset_dir):
#         self.dataset_dir = dataset_dir
def ImageResample(sitk_image, new_spacing=[1., 1., 1.], is_label = False):
    '''
    sitk_image:
    new_spacing: x,y,z
    is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
    '''
    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    new_spacing = np.array(new_spacing)
    new_size = size * spacing / new_spacing
    new_spacing_refine = size * spacing / new_size
    new_spacing_refine = [float(s) for s in new_spacing_refine]
    new_size = [int(s) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        #resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetInterpolator(sitk.sitkLinear)

    newimage = resample.Execute(sitk_image)
    return newimage


def test_load():
    img_path = r"/Users/qinyang.lu/PycharmProjects/4Fun/data.nii.gz"
    label_path = r"/Users/qinyang.lu/PycharmProjects/4Fun/label.nii.gz"
    # img_path = r"D:\git\data\StructSeg 2019\Task1_HaN_OAR\HaN_OAR\1\data.nii.gz"
    # label_path = r"D:\git\data\StructSeg 2019\Task1_HaN_OAR\HaN_OAR\1\label.nii.gz"
    ds_im = sitk.ReadImage(img_path)
    ds_label = sitk.ReadImage(label_path)


    # print(np.shape(img_array))
    # print(np.shape(label_array))

    # print(ds_im.GetOrigin())
    # print("voxel spacing:{}".format(ds_im.GetSpacing()))
    # print("origin size:{}".format(ds_im.GetSize()))
    # print(ds_im.GetDirection())

    new_ds_im = ImageResample(ds_im, new_spacing=[3., 3., 3.])
    # print("size after resample 3mm:{}".format(new_ds_im.GetSize()))
    new_ds_label = ImageResample(ds_label, new_spacing=[3., 3., 3.])


    img_array = sitk.GetArrayFromImage(new_ds_im)
    label_array = sitk.GetArrayFromImage(new_ds_label)

    sub_volume_im = img_array[0:96, 0:96, 0:96]
    sub_volume_label = label_array[0:96, 0:96, 0:96]
    sub_volume_im = np.transpose(sub_volume_im, (2, 0, 1))
    sub_volume_label = np.transpose(sub_volume_label, (2, 0, 1))
    sub_volume_label = np.reshape(sub_volume_label, -1)
    one_hot = np.eye(23)[sub_volume_label].T


    trainTransform = transforms.Compose([
        transforms.ToTensor(),
    ])

    sub_volume_im = trainTransform(sub_volume_im)
    sub_volume_im = torch.reshape(sub_volume_im, (1, 1, 96, 96, 96))
    sub_volume_im = sub_volume_im.type(torch.FloatTensor)

    sub_volume_label = trainTransform(one_hot)
    sub_volume_label = torch.reshape(sub_volume_label, (1, 23, -1))
    sub_volume_label = sub_volume_label.type(torch.FloatTensor)

    model = vnet.VNet(classes=23, batch_size=1).cpu()
    output = model(sub_volume_im)
    print(output.shape)



if __name__ == '__main__':
    test_load()