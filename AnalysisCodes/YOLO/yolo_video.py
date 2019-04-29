import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image

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



def detect_img(yolo):
    file_names = getImageList('testimages.txt')
    box_sum=""
    iou_out=""
    print(len(file_names))
    for file in file_names:
            file_label = file.replace(".jpg",".txt")
            with open(file_label) as f:
                cur_content = f.read().strip('\n')
                ls_label_box = cur_content.split(" ")
                label_box=[9999,9999,9999,9999]
                for i in range(4):  
                  label_box[i]=float(ls_label_box[i])

        #summer
        #img = input('Input image filename:')
            try:
                #summer:img--file
                image = Image.open(file)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image,iou_out,box_sum,label_box,file)
                r_image[0].show()  
                box_sum=r_image[1]
                iou_out=r_image[2]         
    with open('loc.txt', 'w') as f:
        f.write(box_sum)
    with open('iou.txt', 'w') as f:
        f.write(iou_out)
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=True, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
