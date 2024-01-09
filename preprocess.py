import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch
import random
import pickle

# Load preprocessed data
emg_data = []
strain_data = []

fileList = os.listdir(path="C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\raw_data")
for i in fileList:
    filePath = "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\raw_data\\"+i
    with open(file=filePath,mode='r') as f:
        emg = []
        strain = []
        lines = f.readlines()
        for line in lines:
            words = line.split(sep=",")
            words = [float(word) for word in words]
            emg.append(words[0:8])
            strain.append(0.512*words[8]/2048)
        emg = np.asarray(emg)                                                   # emg shape: [8,L]
        strain = np.asarray(strain)
        emg_data.append(emg.T)
        strain_data.append(strain)
random_emg = emg_data.pop(10)
random_strain = strain_data.pop(10)
actionless_emg = emg_data.pop(28)
actionless_strain = strain_data.pop(28)

# Trimming & Parsing
valid_emg = []
valid_strain= []
valid_pre_strain = []
for i in range(39):
    strain_peaks,others = scipy.signal.find_peaks(strain_data[i],height=0.225,distance=2000,prominence=0.095)
    if i==30:
        strain_peaks = strain_peaks[1:]
    elif i==31:
        strain_peaks = strain_peaks[5:]
    elif i==36:
        strain_peaks = strain_peaks[1:]
    left_min = np.argmin(strain_data[i][strain_peaks[0]-4000:strain_peaks[0]])+strain_peaks[0]-4000
    right_min = np.argmin(strain_data[i][strain_peaks[-1]:strain_peaks[-1]+4000])+strain_peaks[-1]+1
    index = np.arange(left_min,right_min)
    emg = []
    for j in range(8):
        emg.append(emg_data[i][j][np.subtract(index,200)])
    emg = torch.Tensor(np.array(emg))
    valid_emg.append(emg.T.unfold(dimension=0,size=400,step=100).transpose(1,2))            # emg shape: [N,400,8]
    valid_pre_strain.append(torch.Tensor(strain_data[i][np.subtract(index,200)]))           
    valid_pre_strain[i] = valid_pre_strain[i].unfold(dimension=0,size=400,step=100)         # pre_strain shape: [N,400]
    valid_strain.append(torch.Tensor(strain_data[i][index]))
    valid_strain[i] = valid_strain[i].unfold(dimension=0,size=400,step=100)
    valid_strain[i] = torch.mean(valid_strain[i][:,359:399],dim=1)                          # strain shape: [N]
###########################################################################################################################
#############################################     Validation, Random     ##################################################
###########################################################################################################################
validation_emg_1019 = valid_emg.pop(10)
validation_emg_1026 = valid_emg.pop(27)
validation_emg_1031 = valid_emg.pop(36)
validation_emg = torch.cat((validation_emg_1019,validation_emg_1026,validation_emg_1031),dim=0)
validation_pre_strain_1019 = valid_pre_strain.pop(10)
validation_pre_strain_1026 = valid_pre_strain.pop(27)
validation_pre_strain_1031 = valid_pre_strain.pop(36)
validation_pre_strain = torch.cat((validation_pre_strain_1019,validation_pre_strain_1026,validation_pre_strain_1031),dim=0)
validation_strain_1019 = valid_strain.pop(10)
validation_strain_1026 = valid_strain.pop(27)
validation_strain_1031 = valid_strain.pop(36)
validation_strain = torch.cat((validation_strain_1019,validation_strain_1026,validation_strain_1031),dim=0)
random_emg = torch.Tensor(random_emg[:,:-200])
random_pre_strain = torch.Tensor(random_strain[:-200])
random_strain = torch.Tensor(random_strain[200:])
random_emg = random_emg.T
random_emg = random_emg.unfold(dimension=0,size=400,step=100).transpose(1,2)
random_pre_strain = random_pre_strain.unfold(dimension=0,size=400,step=100)
random_strain = random_strain.unfold(dimension=0,size=400,step=100)
random_strain = torch.mean(random_strain[:,359:399],dim=1)
actionless_emg = torch.Tensor(actionless_emg[:,:-200])
actionless_pre_strain = torch.Tensor(actionless_strain[:-200])
actionless_strain = torch.Tensor(actionless_strain[200:])
actionless_emg = actionless_emg.T
actionless_emg = actionless_emg.unfold(dimension=0,size=400,step=100).transpose(1,2)
actionless_pre_strain = actionless_pre_strain.unfold(dimension=0,size=400,step=100)
actionless_strain = actionless_strain.unfold(dimension=0,size=400,step=100)
actionless_strain = torch.mean(actionless_strain[:,359:399],dim=1)
###########################################################################################################################
#####################################################     Calibration    ##################################################
###########################################################################################################################
whole_strain_old = []
for i in ["1","2","3","4","5","6","7","10","validation"]:
    strain_filePath = "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\raw_data\\1026_set_"+\
        i+".txt"
    strains = []
    with open(file=strain_filePath,mode='r') as f:
        lines = f.readlines()
        for line in lines:
            words = line.split(sep=",")
            words = [float(i) for i in words]
            strain = words[-1]
            strains.append(0.512*strain/2048)
    strains = np.asarray(strains)
    whole_strain_old.append(strains)
for i in ["1","2","3"]:
    strain_filePath = "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\raw_data\\1031_set_"+\
        i+".txt"
    strains = []
    with open(file=strain_filePath,mode='r') as f:
        lines = f.readlines()
        for line in lines:
            words = line.split(sep=",")
            words = [float(i) for i in words]
            strain = words[-1]
            strains.append(0.512*strain/2048)
    strains = np.asarray(strains)
    whole_strain_old.append(strains)

angle_filePath = "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\calculated_angle_1026.xlsx"
angles = pd.read_excel(io=angle_filePath,sheet_name="Sheet1")
def untilnan(arr):
    i = 0
    while i < len(arr):
        if math.isnan(arr[i]): break
        i += 1
    new_arr = arr[:i]
    return new_arr
angle_filePath_2 = "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\calculated_angle_1031.xlsx"
angles_2 = pd.read_excel(io=angle_filePath_2,sheet_name="Sheet1")

whole_angle_old = []
for i in angles.columns[1:]:
    angle = angles[i]
    angle = angle.to_numpy()
    angle = untilnan(angle)
    whole_angle_old.append(angle)
whole_angle_old.pop(7)
whole_angle_old.pop(7)
for i in angles_2.columns[1:4]:
    angle = angles_2[i]
    angle = angle.to_numpy()
    angle = untilnan(angle)
    whole_angle_old.append(angle)

fit_strains_old = []
fit_angles_old = []
for i in range(len(whole_strain_old)):
    angle_length = len(whole_angle_old[i])
    strain_peaks, strain_proms = scipy.signal.find_peaks(whole_strain_old[i],height=0.225,distance=2000,\
                                                         prominence=0.095)
    if i==10:
        strain_peaks = strain_peaks[5:]
    angle_peaks, angle_proms = scipy.signal.find_peaks(whole_angle_old[i],height=75,distance=60,prominence=40)
    data_point = np.arange(0,angle_length)
    f = scipy.interpolate.interp1d(data_point,whole_angle_old[i])
    start = np.arange(0,angle_peaks[0],0.03)
    wavelength = []
    for j in range(len(strain_peaks)-1):
        wavelength.append(strain_peaks[j+1]-strain_peaks[j]+1)
    for k in range(len(strain_peaks)-1):
        mid = np.linspace(angle_peaks[k],angle_peaks[k+1],num=wavelength[k])[:-1]
        start = np.concatenate((start,mid))
    end = np.arange(angle_peaks[-1],angle_length-1,0.03)
    extended_data_point = np.concatenate((start,end))
    extended_angle = f(extended_data_point)
    angle_peaks,angle_proms = scipy.signal.find_peaks(extended_angle,height=75,distance=60,prominence=40)
    head_dif = len(whole_strain_old[i][:strain_peaks[0]])-len(extended_angle[:angle_peaks[0]])
    if head_dif > 0:
        whole_strain_old[i] = whole_strain_old[i][head_dif:]
        strain_peaks = strain_peaks - head_dif
    tail_dif = len(whole_strain_old[i][strain_peaks[-1]:]) - len(extended_angle[angle_peaks[-1]:])
    if tail_dif > 0:
        tail_dif = -1*tail_dif
        whole_strain_old[i] = whole_strain_old[i][:tail_dif]
    whole_strain_old[i] = whole_strain_old[i][strain_peaks[0]-3000:]
    extended_angle = extended_angle[angle_peaks[0]-3000:]
    fit_strains_old.append(whole_strain_old[i])
    fit_angles_old.append(extended_angle)

models_old = []
for i in range(10):
    models_old.append([])
    
def get_rsquared(x,y,model):
    y_model = model(x)
    y_bar = np.mean(y)
    r_squared = np.sum((y_model-y_bar)**2)/np.sum((y-y_bar)**2)
    return r_squared

class strain_angle_pair:
    def __init__(self, strain, angle, piece):
        self.strain = strain
        self.angle = angle
        self.piece = piece
    def pair(self):
        pairs = []
        for i in range(len(self.strain)):
            pairs.append([self.strain[i],self.angle[i]])
        min_strain = np.min(self.strain)
        from_max_to_min = np.max(self.strain)-min_strain
        even_ten = []
        for i in range(2*self.piece):
            even_ten.append([])
        for pair in pairs:
            index = math.floor(self.piece*((pair[0]-min_strain)/(from_max_to_min+1e-05)))
            even_ten[2*index].append(pair[0])
            even_ten[2*index+1].append(pair[1])
        for i in range(len(even_ten)):
            even_ten[i] = np.asarray(even_ten[i])
        return even_ten
    
k = 0
for i in [0,1,2,3,4,7,8,9,10,11]:
    min_strain = np.min(fit_strains_old[i])
    models_old[k].append(min_strain)
    pair = strain_angle_pair(fit_strains_old[i],fit_angles_old[i],10)
    piecewise = pair.pair()
    for j in range(pair.piece):
        linear_model_params = np.polyfit(piecewise[2*j], piecewise[2*j+1], 1)
        linear_model = np.poly1d(linear_model_params)
        linear_rsquared = get_rsquared(piecewise[2*j], piecewise[2*j+1], linear_model)
        quadratice_model_params = np.polyfit(piecewise[2*j], piecewise[2*j+1], 2)
        quadratic_model = np.poly1d(quadratice_model_params)
        quadratice_squared = get_rsquared(piecewise[2*j], piecewise[2*j+1], quadratic_model)
        if linear_rsquared <= quadratice_squared:
            models_old[k].append(linear_model_params)
            models_old[k].append(np.max(piecewise[2*j]))
        else:
            models_old[k].append(quadratice_model_params)
            models_old[k].append(np.max(piecewise[2*j]))
    k = k + 1
    
def get_angles(strains,models):
    strains = np.array(strains)
    angles = np.zeros_like(strains)
    min_strain = np.min(strains)
    index = 0
    prev_difference = 10000
    curves = []
    for i in range(len(models)):
        difference = abs(min_strain-models[i][0])
        if difference < prev_difference:
            index = i
            prev_difference = difference
    for i in range(len(models[index])//2):
        curves.append(np.poly1d(models[index][2*i+1]))
    for i in range(len(strains)):
        for j in range(len(models[index])//2):
            if strains[i] <= models[index][2*j+2]:
                angle = curves[j](strains[i])
                break
            elif strains[i] > models[index][-1]:
                angle = curves[-1](strains[i])
                break
        angles[i] = angle
    angles = torch.from_numpy(angles)
    
    return angles

whole_strain_new = []
for i in ["5","6","7","8","9","10","validation"]:
    strain_filePath = "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\raw_data\\1031_set_"+\
        i+".txt"
    strains = []
    with open(file=strain_filePath,mode='r') as f:
        lines = f.readlines()
        for line in lines:
            words = line.split(sep=",")
            words = [float(i) for i in words]
            strain = words[-1]
            strains.append(0.512*strain/2048)
    strains = np.asarray(strains)
    whole_strain_new.append(strains)

angle_filePath = "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\calculated_angle_1031.xlsx"
angles = pd.read_excel(io=angle_filePath,sheet_name="Sheet1")
whole_angle_new = []
for i in angles.columns[5:]:
    angle = angles[i]
    angle = angle.to_numpy()
    angle = untilnan(angle)
    whole_angle_new.append(angle)

fit_strains_new = []
fit_angles_new = []
for i in range(len(whole_strain_new)):
    angle_length = len(whole_angle_new[i])
    strain_peaks, strain_proms = scipy.signal.find_peaks(whole_strain_new[i],height=0.225,distance=2000,\
                                                         prominence=0.095)
    if i==3:
        strain_peaks = strain_peaks[1:]
    elif i==5:
        strain_peaks = strain_peaks[1:]
    angle_peaks, angle_proms = scipy.signal.find_peaks(whole_angle_new[i],height=75,distance=60,prominence=40)
    data_point = np.arange(0,angle_length)
    f = scipy.interpolate.interp1d(data_point,whole_angle_new[i])
    start = np.arange(0,angle_peaks[0],0.03)
    wavelength = []
    for j in range(len(strain_peaks)-1):
        wavelength.append(strain_peaks[j+1]-strain_peaks[j]+1)
    for k in range(len(strain_peaks)-1):
        mid = np.linspace(angle_peaks[k],angle_peaks[k+1],num=wavelength[k])[:-1]
        start = np.concatenate((start,mid))
    end = np.arange(angle_peaks[-1],angle_length-1,0.03)
    extended_data_point = np.concatenate((start,end))
    extended_angle = f(extended_data_point)
    angle_peaks,angle_proms = scipy.signal.find_peaks(extended_angle,height=75,distance=60,prominence=40)
    head_dif = len(whole_strain_new[i][:strain_peaks[0]])-len(extended_angle[:angle_peaks[0]])
    if head_dif < 0:
        head_dif = -1*(head_dif)
        extended_angle = extended_angle[head_dif:]
        angle_peaks,angle_proms = scipy.signal.find_peaks(extended_angle,height=75,distance=60,prominence=40)
    elif head_dif > 0:
        whole_strain_new[i] = whole_strain_new[i][head_dif:]
        strain_peaks = strain_peaks - head_dif
    tail_dif = len(whole_strain_new[i][strain_peaks[-1]:]) - len(extended_angle[angle_peaks[-1]:])
    if tail_dif > 0:
        tail_dif = -1*tail_dif
        whole_strain_new[i] = whole_strain_new[i][:tail_dif]
    elif tail_dif < 0:
        extended_angle = extended_angle[:tail_dif]
        angle_peaks,angle_proms = scipy.signal.find_peaks(extended_angle,height=75,distance=60,prominence=40)
    whole_strain_new[i] = whole_strain_new[i][strain_peaks[0]-3000:]
    extended_angle = extended_angle[angle_peaks[0]-3000:]
    fit_strains_new.append(whole_strain_new[i])
    fit_angles_new.append(extended_angle)
    
models_new = [[],[],[],[],[],[],[]]
for i in range(len(fit_strains_new)):
    min_strain = np.min(fit_strains_new[i])
    models_new[i].append(min_strain)
    pair = strain_angle_pair(fit_strains_new[i],fit_angles_new[i],10)
    piecewise = pair.pair()
    for j in range(pair.piece):
        linear_model_params = np.polyfit(piecewise[2*j], piecewise[2*j+1], 1)
        linear_model = np.poly1d(linear_model_params)
        linear_rsquared = get_rsquared(piecewise[2*j], piecewise[2*j+1], linear_model)
        quadratice_model_params = np.polyfit(piecewise[2*j], piecewise[2*j+1], 2)
        quadratic_model = np.poly1d(quadratice_model_params)
        quadratice_squared = get_rsquared(piecewise[2*j], piecewise[2*j+1], quadratic_model)
        if linear_rsquared <= quadratice_squared:
            models_new[i].append(linear_model_params)
            models_new[i].append(np.max(piecewise[2*j]))
        else:
            models_new[i].append(quadratice_model_params)
            models_new[i].append(np.max(piecewise[2*j]))

####################################################################################################
####################################################################################################
####################################################################################################

valid_angle = []
for i in range(28):
    angle = get_angles(valid_strain[i],models_old)
    valid_angle.append(angle)
valid_angle.append(get_angles(valid_strain[28], models_new))
valid_angle.append(get_angles(valid_strain[29], models_old))
valid_angle.append(get_angles(valid_strain[30], models_old))
for i in range(31,36):
    valid_angle.append(get_angles(valid_strain[i], models_new))
validation_angle_1019 = get_angles(validation_strain_1019, models_old)
validation_angle_1026 = get_angles(validation_strain_1026, models_old)
validation_angle_1031 = get_angles(validation_strain_1031, models_new)
actionless_angle = get_angles(actionless_strain, models_old)
random_angle = get_angles(random_strain, models_old)

# Data permutation
total_emg = valid_emg[0]
total_pre_strain = valid_pre_strain[0]
total_strain = valid_strain[0]
total_angle = valid_angle[0]
for i in range(len(valid_emg)-1):
    total_emg = torch.cat((total_emg,valid_emg[i+1]),dim=0)
    total_pre_strain = torch.cat((total_pre_strain,valid_pre_strain[i+1]),dim=0)
    total_strain = torch.cat((total_strain,valid_strain[i+1]),dim=0)
    total_angle = torch.cat((total_angle,valid_angle[i+1]),dim=0)
total_emg = torch.cat((total_emg,actionless_emg),dim=0)
total_pre_strain = torch.cat((total_pre_strain,actionless_pre_strain),dim=0)
total_strain = torch.cat((total_strain,actionless_strain),dim=0)
total_angle = torch.cat((total_angle,actionless_angle),dim=0)
validation_angle = torch.cat((validation_angle_1019,validation_angle_1026,validation_angle_1031),dim=0)

def by_strain(strain):
    max_strain = torch.max(strain)
    min_strain = torch.min(strain)
    scope = max_strain-min_strain
    index_container = []
    train_index = []
    test_index = []
    scope = 2048*scope.item()/0.512
    scope = round(scope)
    scope = scope//20
    for i in range(scope):
        index_container.append([])
    for i in range(strain.size(0)):
        if (((strain[i]-min_strain)*(2048/0.512))//20)==scope:
            index = scope-1
        else:
            index = round(((strain[i]-min_strain)*(2048/0.512)).item())//20
        index_container[index].append(i)
    for i in range(len(index_container)):
        eighty = round(len(index_container[i])*0.8)
        train = random.sample(index_container[i],eighty)
        test = set(index_container[i])-set(train)
        test = list(test)
        train_index.append(train)
        test_index.append(test)
    for i in range(1,len(index_container)):
        train_index[0] = train_index[0] + train_index[i]
        test_index[0] = test_index[0] + test_index[i]
    train_index[0] = np.asarray(train_index[0])
    train_index[0] = np.random.permutation(train_index[0])
    train_index[0] = torch.from_numpy(train_index[0]).to(torch.long)
    test_index[0] = np.asarray(test_index[0])
    test_index[0] = np.random.permutation(test_index[0])
    test_index[0] = torch.from_numpy(test_index[0]).to(torch.long)
    return train_index[0],test_index[0]
train_index,test_index = by_strain(total_strain)
delta_strain = torch.mean(total_pre_strain[:,359:399],dim=1)
delta_strain = total_strain - delta_strain
validation_delta_strain = torch.mean(validation_pre_strain[:,359:399],dim=1)
validation_delta_strain = validation_strain - validation_delta_strain
random_delta_strain = torch.mean(random_pre_strain[:,359:399],dim=1)
random_delta_strain = random_strain - random_delta_strain

total_pre_strain = torch.mul(total_pre_strain,0.1)
random_pre_strain = torch.mul(random_pre_strain,0.1)
validation_pre_strain = torch.mul(validation_pre_strain,0.1)

train_emg = total_emg[train_index]
test_emg = total_emg[test_index]
train_pre_strain = total_pre_strain[train_index]
test_pre_strain = total_pre_strain[test_index]
train_strain = total_strain[train_index]
test_strain = total_strain[test_index]
train_angle = total_angle[train_index]
test_angle = total_angle[test_index]
train_delta_strain = delta_strain[train_index]
test_delta_strain = delta_strain[test_index]

train_emg_filepath = \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\train_emg_all.p"
train_pre_strain_filepath = \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\train_pre_strain_all.p"
train_strain_filepath = \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\train_strain_all.p"
test_emg_filepath = \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\test_emg_all.p"
test_pre_strain_filepath = \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\test_pre_strain_all.p"
test_strain_filepath = \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\test_strain_all.p"
train_angle_filepath = \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\train_angle_all.p"
test_angle_filepath= \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\test_angle_all.p"
train_delta_strain_filepath = \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\train_delta_strain_all.p"
test_delta_strain_filepath = \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\test_delta_strain_all.p"
with open(train_emg_filepath,'wb') as f:
    pickle.dump(train_emg,f)
with open(test_emg_filepath,'wb') as f:
    pickle.dump(test_emg,f)
with open(train_pre_strain_filepath,'wb') as f:
    pickle.dump(train_pre_strain,f)
with open(test_pre_strain_filepath,'wb') as f:
    pickle.dump(test_pre_strain,f)
with open(train_strain_filepath,'wb') as f:
    pickle.dump(train_strain,f)
with open(test_strain_filepath,'wb') as f:
    pickle.dump(test_strain,f)
with open(train_angle_filepath,'wb') as f:
    pickle.dump(train_angle,f)
with open(test_angle_filepath,'wb') as f:
    pickle.dump(test_angle,f)
with open(train_delta_strain_filepath,'wb') as f:
    pickle.dump(train_delta_strain,f)
with open(test_delta_strain_filepath,'wb') as f:
    pickle.dump(test_delta_strain,f)
random_emg_filepath = \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\random_emg_all.p"
random_pre_strain_filepath = \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\random_pre_strain_all.p"
random_strain_filepath = \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\random_strain_all.p"
random_angle_filepath = \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\random_angle_all.p"
random_delta_strain_filepath = \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\random_delta_strain_all.p"
with open(random_emg_filepath,'wb') as f:
    pickle.dump(random_emg,f)
with open(random_pre_strain_filepath,'wb') as f:
    pickle.dump(random_pre_strain,f)
with open(random_strain_filepath,'wb') as f:
    pickle.dump(random_strain,f)
with open(random_angle_filepath,'wb') as f:
    pickle.dump(random_angle,f)
with open(random_delta_strain_filepath,'wb') as f:
    pickle.dump(random_delta_strain,f)
validation_emg_filepath = \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\validation_emg_all.p"
validation_pre_strain_filepath = \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\validation_pre_strain_all.p"
validation_strain_filepath = \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\validation_strain_all.p"
validation_angle_filepath = \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\validation_angle_all.p"
validation_delta_strain_filepath = \
    "C:\\Users\\srbl\\Desktop\\SRBL\\Hyundai\\Data\\All_data\\processed_with_actionless\\validation_delta_strain_all.p"
with open(validation_emg_filepath,'wb') as f:
    pickle.dump(validation_emg,f)
with open(validation_pre_strain_filepath,'wb') as f:
    pickle.dump(validation_pre_strain,f)
with open(validation_strain_filepath,'wb') as f:
    pickle.dump(validation_strain,f)
with open(validation_angle_filepath,'wb') as f:
    pickle.dump(validation_angle,f)
with open(validation_delta_strain_filepath,'wb') as f:
    pickle.dump(validation_delta_strain,f)