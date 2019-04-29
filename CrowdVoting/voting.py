# This is the code for analyzing detection results.
'''
@file_name: the txt file contains the detection results. Format: patient_id, frame_id, Ymin, Xmin,Ymax,Xmax,Score, IoU
@test_id: the tested patient
'''

import numpy as np

file_name = "faster.txt"
test_id = "18"

with open(file_name,'r') as f:
    lines = f.readlines()
    frame_id_18=list()
    info_18=list()

    for line in lines:
        sp_1=line.split("_")
        p_id=sp_1[0]
        sp_2=sp_1[1].split(".jpg")
        frame_id = sp_2[0]        
        cur_info = sp_2[1].split(",")
        #print(cur_info)
        cur_info_mat=np.zeros(shape=(8,1))
        cur_info_mat[0]=int(p_id)
        cur_info_mat[1]=int(frame_id)
        for i in range(6):
            cur_info_mat[i+2]=float(cur_info[i+1].replace("\n",""))
        if(p_id==test_id and cur_info_mat[7]!=0):
            #print(cur_info_mat)
            frame_id_18.append(frame_id)
            info_18.append(cur_info_mat)

#bumber of correctly detected bounding box        
num_bounding_box= len(set(frame_id_18))
print(num_bounding_box)

#further anaysis
sele_18=list()
m_18=np.array(info_18)
for every in set(frame_id_18):
    cur_m_18=list()
    for i in range(m_18.shape[0]):        
        if(m_18[i,1]==int(every)):
            cur_m_18.append(m_18[i,:])
    #cur_m_18=np.array(cur_m_18)
    cur_max=-1
    for j in range(len(cur_m_18)):
        if cur_m_18[j][6]>cur_max:
            cur_max_b=cur_m_18[j]
            cur_max=cur_m_18[j][6]
    sele_18.append(cur_max_b)
iou_18=np.zeros(shape=(len(sele_18),2))
for i in range(len(sele_18)):
    iou_18[i,1]=sele_18[i][7]
    iou_18[i,0]=sele_18[i][1]    

print(iou_18.mean(0))
print(iou_18.var(0))
print(iou_18)
