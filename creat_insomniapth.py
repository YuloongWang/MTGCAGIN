import os
import numpy as np
from torch import tensor, float32, save, load
import pandas as pd
from collections import Counter
import random

# D:\ProgramData\BrainGNN\stagin-main\data\behavioral\insomnia_aal.csv
csv_data = pd.read_csv('D:\\ProgramData\\BrainGNN\\stagin-main\\data\\behavioral\\insomnia_exp.csv')
path = "D:\\ProgramData\\BrainGNN\\stagin-main\\data\\insomnia\\insomniaData_cc200\\"
datanames = [f'{str(num)}.npy' for num in list(csv_data['Subject'])]
# datanames = os.listdir(path)
# random.shuffle(datanames)
namelist = []  # 文件名
subjectID = []  # 从文件名中拆分出的ID，size:871
data_shape = []  # 时间点长度
data_dict = {}  # 构造pth文件所用的字典，key为ID，value为data
SUB_ID = list(csv_data['Subject'])  # csv文件SUB_ID列，ID
ID_label = list(csv_data['label'])  # csv文件DX_GROUP列，label
for f in datanames:
    if f.endswith('.npy'):
        data = np.load(path + f).T
        data_dict[f.split('.')[0]] = data
        namelist.append(f)
        data_shape.append(data.shape[0])
        subjectID.append(f.split('.')[0])

print(data_dict.__len__())
print(Counter(data_shape))
save(data_dict, os.path.join('D:\\ProgramData\\BrainGNN\\stagin-main\\data\\insomnia_cc200.pth'))  # ###

# results_int = list(map(int, subjectID))
# index_ID = []  # 871个ID在1035个ID中的位置
# for i in results_int:
#     index_ID.append(SUB_ID.index(i))
#
# subject_label = []  # 871个受试者的label
# for i in index_ID:
#     subject_label.append(ID_label[i])
#
# subject_label_list = list(zip(results_int, subject_label))
# ABIDE_list = pd.DataFrame(data=subject_label_list, columns=['Subject', 'label'])
# ABIDE_list.to_csv('D:\\ProgramData\\BrainGNN\\stagin-main\\data\\behavioral\\insomnia_eFC.csv', index=False)  # ###
