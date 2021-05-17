import re

import SimpleITK as sitk
import numpy as np
import os
img_path =r"G:\Spine_Compitition\Spine\MR"
mask_path = r"G:\Spine_Compitition\Spine\Mask"
#label_path = "/home/ubuntu/liuyiyao/3D_breast_Seg/Dataset/great_data_large"
match_txt_dir = r"G:\pre_Data\Spine"

mask_names = os.listdir(mask_path)
j=0
for name in mask_names:
    print(name)

    if not os.path.isdir(match_txt_dir):
        os.makedirs(match_txt_dir)
    match_txt = match_txt_dir+'/match.txt'
    open(match_txt, 'a').write(
        "Name: {} >> No: {:0>5d}\n".format(name,j))
    dir = r"G:\pre_Data\Spine\case_%05.0d"%j
    if not os.path.isdir(dir):
        os.makedirs(dir)
    id = re.findall("\d+",name)[0]
    img_name = "Case%d.nii.gz"%int(id)
    mask_data = sitk.ReadImage(mask_path + '/' + name)
    img_data = sitk.ReadImage(img_path + '/' + img_name)
    sitk.WriteImage(img_data,dir+'/imaging.nii.gz')
    sitk.WriteImage(mask_data,dir+'/segmentation.nii.gz')
    #source = open(label_path + '/' + name.split('.')[0] + '/label.txt')  # 打开源文件
    #indate = source.read()  # 显示所有源文件内容
    #target = open(dir + '/label.txt', 'w')  # 打开目的文件
    #target.write(indate)
    j+=1
