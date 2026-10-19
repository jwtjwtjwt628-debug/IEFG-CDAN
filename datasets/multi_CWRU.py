import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm

#Digital data was collected at 12,000 samples per second
signal_size = 1024
dataname= {0:["97.mat","105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat","234.mat"],  # 1797rpm
           1:["98.mat","106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat","235.mat"],  # 1772rpm
           2:["99.mat","107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat","236.mat"],  # 1750rpm
           3:["100.mat","108.mat", "121.mat","133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat","237.mat"]}  # 1730rpm

datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data",
               "Normal Baseline Data"]
axis = ["_DE_time", "_FE_time", "_BA_time"]

label = [i for i in range(0, 10)]

def get_files(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab =[]
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            if n==0:
               path1 =os.path.join(root,datasetname[3], dataname[N[k]][n])
            else:
                path1 = os.path.join(root,datasetname[0], dataname[N[k]][n])
            data1, lab1 = data_load(path1,dataname[N[k]][n],label=label[n])
            lab1=k*100+np.array(lab1)#k是域标签,lab1代表的是类标签
            lab1=lab1.tolist()
            data += data1
            lab +=lab1
    return [data, lab]


def data_load(filename, axisname, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    # === 修复 CWRU 数据集内部变量名混乱的问题 ===
    mat_dict = loadmat(filename)
    fl = None
    
    # 1. 优先寻找包含 'DE_time' (驱动端) 的键名
    for key in mat_dict.keys():
        if 'DE_time' in key:
            fl = mat_dict[key]
            break
            
    # 2. 如果没找到，就随便找一个不是 '__' 开头的系统变量（避开MATLAB自带的隐藏变量）
    if fl is None:
        for key in mat_dict.keys():
            if not key.startswith('__'):
                fl = mat_dict[key]
                break
                
    # 3. 终极容错：如果实在找不到，打印出里面到底有什么
    if fl is None:
        raise ValueError(f"在文件 {filename} 中找不到振动数据！当前文件包含的变量有: {mat_dict.keys()}")
    # ============================================

    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab

#--------------------------------------------------------------------------------------------------------------------
class CWRU(object):
    num_classes = 10
    inputchannel = 1
    def __init__(self, data_dir, transfer_task, normlizetype="0-1",
                 source_sample_ratio=1.0, noise_sigma=0.0):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        print('self.source_N',self.source_N)
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.source_sample_ratio = source_sample_ratio
        self.noise_sigma = noise_sigma

        train_transforms = [Reshape(), Normalize(self.normlizetype)]
        if noise_sigma > 0:
            train_transforms.append(AddGaussian(sigma=noise_sigma))
        train_transforms.append(Retype())

        self.data_transforms = {
            'train': Compose(train_transforms),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
            ])
        }

    def data_split(self, transfer_learning=True):
        if transfer_learning:
            # get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])

            if self.source_sample_ratio < 1.0:
                train_pd, _ = train_test_split(train_pd,
                    train_size=self.source_sample_ratio,
                    random_state=42, stratify=train_pd["label"])
                print(f'[Source sample reduction] Using {len(train_pd)} samples ({self.source_sample_ratio*100:.0f}%)')

            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_train, target_val
        else:
            #get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_val


"""
    def data_split(self):

"""