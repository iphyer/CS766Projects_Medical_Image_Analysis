# CS766Projects_Medical_Image_Analysis
CS766  Projects Medical Image Analysis

Yiqin Pan, Zhiyi Chen, Mingren Shen

## Table of Content
### 1. Proposal

![Project Pipeline](Archive/project_structure.jpg)

#### Motivation:
Arteriograms are the X-ray images for arteries in the heart, brain and other parts of the body by injecting a special dye into the arteries to enhance the blood vessel. And in real-world practice, radiologists are required to identify the bleeding sites in the image to provide information for future medical intervention. However, analyzing those image results requires manual inspection and discussion of experience radiologists which is slow, labor-intensive and error-prone. So we want to use deep learning based object detection methods to help identify those bleeding sites. 

#### Why important:
Blood bleeding in the brain or other organs is dangerous and can be detected by radiology experiments. And traditionally radiologist can only do this by hand and it would be good if we can design an automatic object detection algorithm to help solve this problem. Now, one member of our team is working with UW-Hospital to solve this problem and we want to try some new ideas for this problem.

#### State-of-Art research:
A two-stage method was used to solve the extravasation detection problem, where the first stage was used to classify whether a bleed was present and the second stage where an object detector was trained to identify the site of bleeding. ResNet-152 was used as the first stage classifier and Faster R-CNN was used as the second stage object detector. Eighty percent of the data was used for training and twenty percent was used to validate the first stage of the algorithm. Ten unique positive arteriogram images from the new series were used to test the second stage of the algorithm.  
This two stages methods are done by one of our team members currently and they just submitted this method as an abstract for CIRSE 2019, Cardiovascular and Interventional Radiology Society Meeting.

#### Reason for developing our own approach:
The reason we are developing our own approach is that the dataset we are using, the Active Extravasation on Arteriograms image datasets, may change significantly according to different patient situations which impedes the performance of common object detection algorithms e.g. Faster R-CNN, YOLO. So we plan to borrow the idea of ensemble learning and try to using crowd voting to get a better performance of the model. Currently, Faster R-CNN model can be 50% accurate, and if we can combine different algorithms together, theoretically we can get better results as shown in the figure below. 
