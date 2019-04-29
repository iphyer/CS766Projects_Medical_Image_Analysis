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
count = 0
output = ''
for file in file_names:
    with open(file) as f:
        cur_content = f.read().strip('\n')
    currLines = cur_content.split(" ")
    cur_l = currLines[0]
    currLines[0]=currLines[1]
    currLines[1]=cur_l
    cur_l = currLines[2]
    currLines[2]=currLines[3]
    currLines[3]=cur_l
    cur_content=""
    for i in range(4):  
        temp = int(float(currLines[i]))
        if i == 0:
            cur_content = str(temp)
        else:
            cur_content = cur_content + "," + str(temp)

    file=file.replace(".txt",".jpg")
    print(file)
    print(cur_content)
    content = './MedData/images/'+ file+ ' '+cur_content + ',0'
    output += content + '\n' 
    count =count+1

with open('output.txt', 'w') as f:
    f.write(output)
