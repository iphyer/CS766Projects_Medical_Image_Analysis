import os

def getImageList(imageTXT):
    """
    Function to loop the testing images for test
    :param imageTXT: the txt that stores the
    :return: imageFileList: the list contains all the original test image list
    """
    imageFileList = list()
    with open(imageTXT,'r') as f:
        lines = f.readlines()
        for line in lines:
            imageFileList.append(line.strip())
    return imageFileList


file_names = getImageList('trains.txt')
print(len(file_names))
count = 0
sum_x_diff=0
sum_y_diff=0
output = 'x_leng     y_leng'
for file in file_names:
    with open(file) as f:
        cur_content = f.read().strip('\n')
    currLines = cur_content.split(" ")
    cur_x_dif = float(currLines[2])-float(currLines[0])
    cur_y_dif = float(currLines[3])-float(currLines[1])
    cur_content=str(cur_x_dif)+"    "+str(cur_y_dif)
    output += cur_content + '\n' 
    count=count+1
    sum_x_diff=sum_x_diff+cur_x_dif
    sum_y_diff=sum_y_diff+cur_y_dif

mean_x_diff=sum_x_diff/count
mean_y_diff=sum_y_diff/count
suma = str(mean_x_diff)+"    "+str(mean_y_diff)
output +=suma+'\n' 
with open('dif.txt', 'w') as f:
    f.write(output)
