from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
import csv
from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

def bbox_iou(a, b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        a: (list of 4 numbers) [y1,x1,y2,x2]
        b: (list of 4 numbers) [y1,x1,y2,x2]
    Returns:
        iou: the value of the IoU of two bboxes

    """
    # (float) Small value to prevent division by zero
    epsilon = 1e-5
    # COORDINATES OF THE INTERSECTION BOX
    # print(a)
    # print(b)
    y1 = max(a[0], b[0])
    x1 = max(a[1], b[1])
    y2 = min(a[2], b[2])
    x2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou

img_height = 512 # Height of the model input images
img_width = 512 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 1 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_coco
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True

# 1: Build the Keras model.

K.clear_session() # Clear previous models from memory.

#model = ssd_300(image_size=(img_height, img_width, img_channels),
#                n_classes=n_classes,
#                mode='training',
#                l2_regularization=0.0005,
#                scales=scales,
#                aspect_ratios_per_layer=aspect_ratios,
#                two_boxes_for_ar1=two_boxes_for_ar1,
#                steps=steps,
#                offsets=offsets,
#                clip_boxes=clip_boxes,
#                variances=variances,
#                normalize_coords=normalize_coords,
#                subtract_mean=mean_color,
#                swap_channels=swap_channels)

model_path = './ssd512_mine.h5'

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'compute_loss': ssd_loss.compute_loss})
testimages_dir = './data/testImages'
testlabels = './data/testlabels.csv'

convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

input_format = ['image_name', 'class_id', 'ymin', 'xmin', 'ymax', 'xmax']

val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

val_dataset.parse_csv(images_dir=testimages_dir,
                  labels_filename=testlabels,
                  input_format=input_format,
                  include_classes='all',
                  random_sample=False,
                  ret=False,
                  verbose=True)

val_dataset.create_hdf5_dataset(file_path='dataset_pascal_voc_07_test.h5',
                                resize=False,
                                variable_image_size=True,
                                verbose=True)

predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=False,
                                         transformations=[convert_to_3_channels,
                                                          resize],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'filenames',
                                                  'inverse_transform',
                                                  'original_images',
                                                  'original_labels'},
                                         keep_images_without_gt=False)

np.set_printoptions(precision = 2, suppress = True, linewidth = 90)

#hits = 0
#total = 33

rescsv1 = open('result1.csv', 'w', newline = '')
rescsv2 = open('result2.csv', 'w', newline = '')
rescsv3 = open('result3.csv', 'w', newline = '')
rescsv4 = open('result4.csv', 'w', newline = '')
rescsv5 = open('result5.csv', 'w', newline = '')
rescsv6 = open('result6.csv', 'w', newline = '')
rescsv7 = open('result7.csv', 'w', newline = '')
rescsv8 = open('result8.csv', 'w', newline = '')
rescsv9 = open('result9.csv', 'w', newline = '')
rescsv10 = open('result10.csv', 'w', newline = '')
writer1 = csv.writer(rescsv1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer2 = csv.writer(rescsv2, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer3 = csv.writer(rescsv3, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer4 = csv.writer(rescsv4, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer5 = csv.writer(rescsv5, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer6 = csv.writer(rescsv6, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer7 = csv.writer(rescsv7, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer8 = csv.writer(rescsv8, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer9 = csv.writer(rescsv9, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer10 = csv.writer(rescsv10, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
labcsv = open('label.csv', 'w', newline = '')
writer11 = csv.writer(labcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = predict_generator

y_pred = model.predict(batch_images)

y_pred_decoded = decode_detections(y_pred,
                                confidence_thresh=0.0005,
                                iou_threshold=0.0001,
                                top_k=10,
                                normalize_coords=normalize_coords,
                                img_height=img_height,
                                img_width=img_width)

y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)

writer1.writerow([batch_filenames[0], y_pred_decoded_inv[0][0][2], y_pred_decoded_inv[0][0][3], y_pred_decoded_inv[0][0][4],y_pred_decoded_inv[0][0][5], y_pred_decoded_inv[0][0][1]])
if(len(y_pred_decoded_inv[0]) >= 2):
	writer2.writerow([batch_filenames[0], y_pred_decoded_inv[0][1][2], y_pred_decoded_inv[0][1][3], y_pred_decoded_inv[0][1][4],y_pred_decoded_inv[0][1][5], y_pred_decoded_inv[0][1][1]])
else:
	writer2.writerow([batch_filenames[0], 0, 0, 0, 0, 0])

if(len(y_pred_decoded_inv[0]) >= 3):
	writer3.writerow([batch_filenames[0], y_pred_decoded_inv[0][2][2], y_pred_decoded_inv[0][2][3], y_pred_decoded_inv[0][2][4],y_pred_decoded_inv[0][2][5], y_pred_decoded_inv[0][2][1]])
else:
	writer3.writerow([batch_filenames[0], 0, 0, 0, 0, 0])
  
if(len(y_pred_decoded_inv[0]) >= 4):
		writer4.writerow([batch_filenames[0], y_pred_decoded_inv[0][3][2], y_pred_decoded_inv[0][3][3], y_pred_decoded_inv[0][3][4],y_pred_decoded_inv[0][3][5], y_pred_decoded_inv[0][3][1]])
else:
	writer4.writerow([batch_filenames[0], 0, 0, 0, 0, 0])
  
if(len(y_pred_decoded_inv[0]) >= 5):
	writer5.writerow([batch_filenames[0], y_pred_decoded_inv[0][4][2], y_pred_decoded_inv[0][4][3], y_pred_decoded_inv[0][4][4],y_pred_decoded_inv[0][4][5], y_pred_decoded_inv[0][4][1]])
else:
	writer5.writerow([batch_filenames[0], 0, 0, 0, 0, 0])
if(len(y_pred_decoded_inv[0]) >= 2):
	writer6.writerow([batch_filenames[0], y_pred_decoded_inv[0][5][2], y_pred_decoded_inv[0][5][3], y_pred_decoded_inv[0][5][4],y_pred_decoded_inv[0][5][5], y_pred_decoded_inv[0][5][1]])
else:
	writer6.writerow([batch_filenames[0], 0, 0, 0, 0, 0])

if(len(y_pred_decoded_inv[0]) >= 3):
	writer7.writerow([batch_filenames[0], y_pred_decoded_inv[0][6][2], y_pred_decoded_inv[0][6][3], y_pred_decoded_inv[0][6][4],y_pred_decoded_inv[0][6][5], y_pred_decoded_inv[0][6][1]])
else:
	writer7.writerow([batch_filenames[0], 0, 0, 0, 0, 0])
  
if(len(y_pred_decoded_inv[0]) >= 4):
	writer8.writerow([batch_filenames[0], y_pred_decoded_inv[0][7][2], y_pred_decoded_inv[0][7][3], y_pred_decoded_inv[0][7][4],y_pred_decoded_inv[0][7][5], y_pred_decoded_inv[0][7][1]])
else:
	writer8.writerow([batch_filenames[0], 0, 0, 0, 0, 0])
  
if(len(y_pred_decoded_inv[0]) >= 5):
	writer9.writerow([batch_filenames[0], y_pred_decoded_inv[0][8][2], y_pred_decoded_inv[0][8][3], y_pred_decoded_inv[0][8][4],y_pred_decoded_inv[0][8][5], y_pred_decoded_inv[0][8][1]])
else:
	writer9.writerow([batch_filenames[0], 0, 0, 0, 0, 0])

if(len(y_pred_decoded_inv[0]) >= 5):
	writer10.writerow([batch_filenames[0], y_pred_decoded_inv[0][9][2], y_pred_decoded_inv[0][9][3], y_pred_decoded_inv[0][9][4],y_pred_decoded_inv[0][9][5], y_pred_decoded_inv[0][9][1]])
else:
	writer10.writerow([batch_filenames[0], 0, 0, 0, 0, 0])

#result_bbox = [y_pred_decoded_inv[0][0][2], y_pred_decoded_inv[0][0][3], y_pred_decoded_inv[0][0][4],y_pred_decoded_inv[0][0][5]]
#print (result_bbox)

writer11.writerow([batch_original_labels[0][0][0], batch_original_labels[0][0][1], batch_original_labels[0][0][2], batch_original_labels[0][0][3], batch_original_labels[0][0][4]])



for i in range(33):
	batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)

	y_pred = model.predict(batch_images)

	y_pred_decoded = decode_detections(y_pred,
                                   	confidence_thresh=0.0005,
                                   	iou_threshold=0.0001,
                                   	top_k=10,
                                   	normalize_coords=normalize_coords,
                                   	img_height=img_height,
                                   	img_width=img_width)

	y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)

	writer1.writerow([batch_filenames[0], y_pred_decoded_inv[0][0][2], y_pred_decoded_inv[0][0][3], y_pred_decoded_inv[0][0][4],y_pred_decoded_inv[0][0][5], y_pred_decoded_inv[0][0][1]])
	if(len(y_pred_decoded_inv[0]) >= 2):
		writer2.writerow([batch_filenames[0], y_pred_decoded_inv[0][1][2], y_pred_decoded_inv[0][1][3], y_pred_decoded_inv[0][1][4],y_pred_decoded_inv[0][1][5], y_pred_decoded_inv[0][1][1]])
	else:
		writer2.writerow([batch_filenames[0], 0, 0, 0, 0, 0])

	if(len(y_pred_decoded_inv[0]) >= 3):
		writer3.writerow([batch_filenames[0], y_pred_decoded_inv[0][2][2], y_pred_decoded_inv[0][2][3], y_pred_decoded_inv[0][2][4],y_pred_decoded_inv[0][2][5], y_pred_decoded_inv[0][2][1]])
	else:
		writer3.writerow([batch_filenames[0], 0, 0, 0, 0, 0])
  
	if(len(y_pred_decoded_inv[0]) >= 4):
		writer4.writerow([batch_filenames[0], y_pred_decoded_inv[0][3][2], y_pred_decoded_inv[0][3][3], y_pred_decoded_inv[0][3][4],y_pred_decoded_inv[0][3][5], y_pred_decoded_inv[0][3][1]])
	else:
		writer4.writerow([batch_filenames[0], 0, 0, 0, 0, 0])
  
	if(len(y_pred_decoded_inv[0]) >= 5):
		writer5.writerow([batch_filenames[0], y_pred_decoded_inv[0][4][2], y_pred_decoded_inv[0][4][3], y_pred_decoded_inv[0][4][4],y_pred_decoded_inv[0][4][5], y_pred_decoded_inv[0][4][1]])
	else:
		writer5.writerow([batch_filenames[0], 0, 0, 0, 0, 0])
	if(len(y_pred_decoded_inv[0]) >= 2):
		writer6.writerow([batch_filenames[0], y_pred_decoded_inv[0][5][2], y_pred_decoded_inv[0][5][3], y_pred_decoded_inv[0][5][4],y_pred_decoded_inv[0][5][5], y_pred_decoded_inv[0][5][1]])
	else:
		writer6.writerow([batch_filenames[0], 0, 0, 0, 0, 0])

	if(len(y_pred_decoded_inv[0]) >= 3):
		writer7.writerow([batch_filenames[0], y_pred_decoded_inv[0][6][2], y_pred_decoded_inv[0][6][3], y_pred_decoded_inv[0][6][4],y_pred_decoded_inv[0][6][5], y_pred_decoded_inv[0][6][1]])
	else:
		writer7.writerow([batch_filenames[0], 0, 0, 0, 0, 0])
  
	if(len(y_pred_decoded_inv[0]) >= 4):
		writer8.writerow([batch_filenames[0], y_pred_decoded_inv[0][7][2], y_pred_decoded_inv[0][7][3], y_pred_decoded_inv[0][7][4],y_pred_decoded_inv[0][7][5], y_pred_decoded_inv[0][7][1]])
	else:
		writer8.writerow([batch_filenames[0], 0, 0, 0, 0, 0])
  
	if(len(y_pred_decoded_inv[0]) >= 5):
		writer9.writerow([batch_filenames[0], y_pred_decoded_inv[0][8][2], y_pred_decoded_inv[0][8][3], y_pred_decoded_inv[0][8][4],y_pred_decoded_inv[0][8][5], y_pred_decoded_inv[0][8][1]])
	else:
		writer9.writerow([batch_filenames[0], 0, 0, 0, 0, 0])

	if(len(y_pred_decoded_inv[0]) >= 5):
		writer10.writerow([batch_filenames[0], y_pred_decoded_inv[0][9][2], y_pred_decoded_inv[0][9][3], y_pred_decoded_inv[0][9][4],y_pred_decoded_inv[0][9][5], y_pred_decoded_inv[0][9][1]])
	else:
		writer10.writerow([batch_filenames[0], 0, 0, 0, 0, 0])

	#result_bbox = [y_pred_decoded_inv[0][0][2], y_pred_decoded_inv[0][0][3], y_pred_decoded_inv[0][0][4],y_pred_decoded_inv[0][0][5]]
	#print (result_bbox)

	writer11.writerow([batch_original_labels[0][0][0], batch_original_labels[0][0][1], batch_original_labels[0][0][2], batch_original_labels[0][0][3], batch_original_labels[0][0][4]])

	#label_bbox = [batch_original_labels[0][0][1], batch_original_labels[0][0][2], batch_original_labels[0][0][3], batch_original_labels[0][0][4]]
	#print (label_bbox)

	#if(bbox_iou(result_bbox, label_bbox) > 0) :
		#hits = hits + 1

#precision = hits / total
#print(precision)
