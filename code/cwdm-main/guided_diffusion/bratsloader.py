import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import os
import os.path
import nibabel


class BRATSVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, mode='train', gen_type=None):
        '''
        directory is expected to contain some folder structure:
                if some subfolder contains only files, all of these
                files are assumed to have a name like
                brats_train_NNN_XXX_123_w.nii.gz
                where XXX is one of t1n, t1c, t2w, t2f, seg
                we assume these five files belong to the same image
                seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.mode = mode
        self.directory = os.path.expanduser(directory)
        self.gentype = gen_type
        self.seqtypes = ['t1n', 't1c', 't2w', 't2f', 'seg']
        self.seqtypes_set = set(self.seqtypes)

        self.database = []

        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have a datadir
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('-')[4].split('.')[0]
                    datapoint[seqtype] = os.path.join(root, f)
                self.database.append(datapoint)

    # ★★★ 修正点1 ★★★
    def __len__(self):
        return len(self.database)

    # ★★★ 修正点2 ★★★
    def __getitem__(self, idx):
        filedict = self.database[idx]

        # Load T1n (Input/Conditioning Modality)
        if 't1n' in filedict:
            t1n_np = nibabel.load(filedict['t1n']).get_fdata()
            t1n_np_clipnorm = clip_and_normalize(t1n_np)
            t1n = torch.zeros(1, 240, 240, 160)
            t1n[:, :, :, :155] = torch.tensor(t1n_np_clipnorm)
            t1n = t1n[:, 8:-8, 8:-8, :]
        else:
            # This case should ideally not happen if t1n is the input
            t1n = torch.zeros(1, 224, 224, 160)

        # Load T1c (Output/Target Modality)
        if 't1c' in filedict:
            t1c_np = nibabel.load(filedict['t1c']).get_fdata()
            t1c_np_clipnorm = clip_and_normalize(t1c_np)
            t1c = torch.zeros(1, 240, 240, 160)
            t1c[:, :, :, :155] = torch.tensor(t1c_np_clipnorm)
            t1c = t1c[:, 8:-8, 8:-8, :]
        else:
            # This case should ideally not happen if t1c is the target
            t1c = torch.zeros(1, 224, 224, 160)

        # Set target and conditioning modalities
        target = t1c
        cond = {'cond_1': t1n}

        return target, cond


def clip_and_normalize(img):
    img_clipped = np.clip(img, np.quantile(img, 0.001), np.quantile(img, 0.999))
    img_normalized = (img_clipped - np.min(img_clipped)) / (np.max(img_clipped) - np.min(img_clipped))

    return img_normalized