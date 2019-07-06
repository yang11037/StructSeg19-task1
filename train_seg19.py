#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torch.utils.data as data

import vnet
np.random.seed(0)


class CtDataset(data.Dataset):
    def __init__(self, dataset_dir, classes, mode, patch_size, new_space, winw, winl):
        self.root = dataset_dir
        self.mode = mode
        self.datalist, self.targetlist = self._get_filelist()
        self.patch_size = patch_size  # [x,y,z]
        self.new_space = new_space
        self.winw = winw
        self.winl = winl
        self.classes = classes

    def _get_filelist(self):
        """get the all paths of nii.gz"""
        if self.mode == "train" or self.mode == "test":
            trainlist = []
            targetlist = []
            for patient_id in os.listdir(self.root):
                current_path = os.path.join(self.root, patient_id)
                trainlist.append(os.path.join(current_path, "data.nii.gz"))
                targetlist.append(os.path.join(current_path, "label.nii.gz"))
            return trainlist, targetlist
        elif self.mode == "inference":
            trainlist = []
            for patient_id in os.listdir(self.root):
                current_path = os.path.join(self.root, patient_id)
                trainlist.append(os.path.join(current_path, "data.nii.gz"))
            return trainlist, None

    def __getitem__(self, index):
        """ get the img and mask, the index is the patient_id here"""
        patient_id = index % 49
        img_path = self.datalist[patient_id]
        if self.mode == "train" or self.mode == "test":
            mask_path = self.targetlist[patient_id]
            img_patch, mask_patch = self._load_data(img_path, mask_path)
            return img_patch, mask_patch
        elif self.mode == "inference":
            img_patch = self._load_data(img_path)
            return img_patch

    def _load_data(self, img_path, label_path = None):
        """ load, resample, normalize, and ramdon crop a sub volume from the image"""
        ds_im = sitk.ReadImage(img_path)
        if label_path:
            ds_label = sitk.ReadImage(label_path)

        new_ds_im = self.ImageResample(ds_im)
        if label_path:
            new_ds_label = self.ImageResample(ds_label)

        # this function will transify the img shape to [depth, width, height]
        img_array = sitk.GetArrayFromImage(new_ds_im)

        # convert the width and level
        imin = self.winl - self.winw / 2
        imax = self.winl + self.winw / 2
        img_array[img_array < imin] = imin
        img_array[img_array > imax] = imax

        # get the ramdon crop point
        x_start = int(np.random.rand() * (img_array.shape[2] - self.patch_size[0]))
        y_start = int(np.random.rand() * (img_array.shape[1] - self.patch_size[1]))
        z_start = int(np.random.rand() * (img_array.shape[0] - self.patch_size[2]))
        print(z_start)
        print(y_start)
        print(x_start)


        sub_volume_im = img_array[z_start:z_start+self.patch_size[2], \
                        y_start:y_start+self.patch_size[1], \
                        x_start:x_start+self.patch_size[0]]

        #TODO: 归一化时，是不是应该原来大于0的现在也大于0
        # normal to [-1, 1]
        for depth in range(sub_volume_im.shape[0]):
            for width in range(sub_volume_im.shape[1]):
                for height in range(sub_volume_im.shape[2]):
                    if sub_volume_im[depth, width, height] < imin:
                        sub_volume_im[depth, width, height] = -1
                    elif sub_volume_im[depth, width, height] < imax:
                        sub_volume_im[depth, width, height] = \
                            2 * (sub_volume_im[depth, width, height] - imin) / self.winw - 1
                    else:
                        sub_volume_im[depth, width, height] = 1

        # sub_volume_im = np.transpose(sub_volume_im, (2, 0, 1))  # depth,height,width in Conv3D

        trainTransform = transforms.Compose([
            transforms.ToTensor(),
        ])
        sub_volume_im = trainTransform(sub_volume_im)
        sub_volume_im = sub_volume_im.permute(1, 2, 0).contiguous()
        sub_volume_im = torch.reshape(sub_volume_im, (1,) + tuple(self.patch_size))
        sub_volume_im = sub_volume_im.type(torch.FloatTensor)


        if label_path:
            label_array = sitk.GetArrayFromImage(new_ds_label)
            sub_volume_label = label_array[z_start:z_start+self.patch_size[2], \
                               y_start:y_start+self.patch_size[1], \
                               x_start:x_start+self.patch_size[0]]
            # sub_volume_label = np.transpose(sub_volume_label, (2, 0, 1))
            sub_volume_label = np.reshape(sub_volume_label, -1)
            one_hot = np.eye(self.classes)[sub_volume_label].T
            sub_volume_label = trainTransform(one_hot)
            sub_volume_label = torch.reshape(sub_volume_label, (self.classes,) +
                                             tuple(self.patch_size))
            sub_volume_label = sub_volume_label.type(torch.FloatTensor)
        if label_path:
            return sub_volume_im, sub_volume_label
        else:
            return sub_volume_im

    def ImageResample(self, sitk_image, is_label=False):
        '''
        sitk_image:
        new_spacing: x,y,z
        is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
        '''
        size = np.array(sitk_image.GetSize())
        spacing = np.array(sitk_image.GetSpacing())
        new_spacing = np.array(self.new_space)
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
            # resample.SetInterpolator(sitk.sitkBSpline)
            resample.SetInterpolator(sitk.sitkLinear)

        newimage = resample.Execute(sitk_image)
        return newimage

    def __len__(self):
        #TODO: 返回数据的总长度
        return len(self.datalist)



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

    winl = 50  # window level
    winw = 350  # window width
    imin = winl - winw / 2
    imax = winl + winw / 2
    img_array[img_array < imin] = imin
    img_array[img_array > imax] = imax

    sub_volume_im = img_array[28:28+96, 75:75+96, 57:57+96]
    sub_volume_label = label_array[28:28+96, 75:75+96, 57:57+96]
    _, axs = plt.subplots(2, 2)
    axs[0][0].imshow(sub_volume_im[50, :, :], cmap='gray')
    axs[0][1].imshow(sub_volume_label[50, :, :], cmap='gray')
    axs[1][0].imshow(img_array[78, :, :], cmap='gray')
    axs[1][1].imshow(label_array[78, :, :], cmap='gray')
    plt.show()
    # # sub_volume_im = np.transpose(sub_volume_im, (2, 0, 1))  # depth,height,width in Conv3D
    # # sub_volume_label = np.transpose(sub_volume_label, (2, 0, 1))
    # sub_volume_label = np.reshape(sub_volume_label, -1)
    # one_hot = np.eye(23)[sub_volume_label].T
    #
    #
    # trainTransform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    #
    # sub_volume_im = trainTransform(sub_volume_im)
    # sub_volume_im = torch.reshape(sub_volume_im, (1, 1, 96, 96, 96))
    # sub_volume_im = sub_volume_im.type(torch.FloatTensor)
    #
    # sub_volume_label = trainTransform(one_hot)
    # sub_volume_label = torch.reshape(sub_volume_label, (1, 23, -1))
    # sub_volume_label = sub_volume_label.type(torch.FloatTensor)
    # print(sub_volume_im.shape)
    # print(sub_volume_label.shape)
    # exit()
    #
    # model = vnet.VNet(classes=23, batch_size=1).cpu()
    # output = model(sub_volume_im)
    # print(output.shape)



if __name__ == '__main__':
    trainSet = CtDataset("./data/test", 23, "train", [96, 96, 96], [3, 3, 3], 350, 50)
    trainLoader = data.DataLoader(trainSet, batch_size=1, shuffle=False)
    dataiter = iter(trainLoader)
    img, mask = dataiter.next()
    _, axs = plt.subplots(2, 1)
    axs[0].imshow(img[0, 0, 0, :, :], cmap='gray')
    axs[1].imshow(mask[0, 20, 0, :, :], cmap='gray')
    plt.show()
    model = vnet.VNet(classes=23, batch_size=1).cpu()
    output = model(img)
    print(output.shape)
    # test_load()