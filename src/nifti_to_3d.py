import os
import torch
import nibabel as nib
import numpy as np
from src.utils import config_loader


data_path = []
directory = config_loader.train_data
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    data_path.append(f)


def label_rescale(image_label, w_ori, h_ori, z_ori, flag):
    w_ori, h_ori, z_ori = int(w_ori), int(h_ori), int(z_ori)
    # resize label map (int)
    if flag == 'trilinear':
        teeth_ids = np.unique(image_label)
        image_label_ori = torch.zeros((w_ori, h_ori, z_ori)).cuda(0)
        image_label = torch.from_numpy(image_label).cuda(0)
        for label_id in range(len(teeth_ids)):
            image_label_bn = (image_label == teeth_ids[label_id]).float()
            # image_label_bn = torch.from_numpy(image_label_bn.astype(float))
            image_label_bn = image_label_bn[None, None, :, :, :]
            image_label_bn = torch.nn.functional.interpolate(image_label_bn, size=(w_ori, h_ori, z_ori),
                                                             mode='trilinear')
            image_label_bn = image_label_bn[0, 0, :, :, :]
            image_label_ori[image_label_bn > 0.5] = teeth_ids[label_id]
        image_label = image_label_ori.cpu().data.numpy()

    if flag == 'nearest':
        image_label = torch.from_numpy(image_label).cuda(0)
        image_label = image_label[None, None, :, :, :].float()
        image_label = torch.nn.functional.interpolate(image_label, size=(w_ori, h_ori, z_ori), mode='nearest')
        image_label = image_label[0, 0, :, :, :].cpu().data.numpy()
    return image_label


def read_data(data_patch):
    # src_data_file = os.path.join(data_patch)
    src_data_vol = nib.load(data_patch)
    image = src_data_vol.get_fdata()

    spacing = src_data_vol.header['pixdim'][1:4]
    print(spacing)
    w, h, d = image.shape
    # rescale if needed ??
    # image = label_rescale(image, w * (spacing[0] / 0.2), h * (spacing[0] / 0.2), d * (spacing[0] / 0.2), 'nearest')
    # if image[image > -1000].mean() < -100:
    #     intensity_scale = (-60 + 1000) / (image[image > -1000].mean() + 1000)
    #     image = (image + 1000) * intensity_scale - 1000
    #
    # image[image < 500] = 500
    # image[image > 2500] = 2500
    # image = (image - 500) / (2500 - 500)
    low_bound = np.percentile(image, 5)
    up_bound = np.percentile(image, 99.9)
    return image, low_bound, up_bound, w, h, d


if __name__ == '__main__':
    # net_roi, net_cnt, net_skl, ins_net = load_model()

    for data_id in range(len(data_path)):
        print('process the data:', data_id)
        print(data_path[data_id])
        # images, labels = read_data(image_list[data_id], image_list1[data_id])

        image, low_bound, up_bound, w_o, h_o, d_o = read_data(data_path[data_id])
        print(image.shape)
        print(type(image))
        print(low_bound, up_bound, w_o, h_o, d_o)
        unique, counts = np.unique(image, return_counts=True)
        print(dict(zip(unique, counts)))

        # tooth_label = inference(image, net_roi, net_cnt, net_skl, ins_net, low_bound, up_bound, w_o, h_o, d_o)
