import os
import numpy as np
import PIL.Image
import siammask
import cv2
import os
from PIL import Image
path='/mnt/DATA/Abhay/Downloads/tf-siammask/drone'


video_folder = 'drone-20'
frame_annotations_folder = "/mnt/DATA/Abhay/Downloads/tf-siammask/drone/"+video_folder+"/gt"
#jpeg_images_folder = "/media/cs18s504/DATA/Arun/siamban-master/testing_dataset/LaSOT/train/Train/"+video_folder+"/JPEGImages1"
#annotations_folder = "/media/cs18s504/DATA/Arun/siamban-master/testing_dataset/LaSOT/train/Train/"+video_folder+"/Annotations1"
ground_truth_path = "/mnt/DATA/Abhay/Downloads/tf-siammask/drone/"+video_folder+"/groundtruth.txt"
#mask_folder="/mnt/DATA/Abhay/Downloads/tf-siammask/drone/"+video_folder+"/masks"
os.mkdir(frame_annotations_folder)
#os.mkdir(mask_folder)

with open(ground_truth_path) as f:
    lines=f.readlines()
    for i in range(len(lines)):
        with open(frame_annotations_folder+'/'+str(i+1).zfill(8)+'.txt','w') as file:
            file.write(' '.join([str(18)]+ lines[i].strip().split(',')))
            file.close()
path1="/mnt/DATA/Abhay/Downloads/tf-siammask/drone/"+video_folder+"/gt"
dir_list1=os.listdir(path1)
count=len(dir_list1)
a1=np.arange(1,count).tolist()
for i in range(len(a1)-1):
    d=(str(a1[i]).zfill(8))
    d1=(str(a1[i+1]).zfill(8))
    b1="{}.jpg".format(d)
    b2="{}.jpg".format(d1)
    b3="{}.txt".format(d)
    with open("/mnt/DATA/Abhay/Downloads/tf-siammask/drone/"+video_folder+'/gt/'+b3) as f:
        lines=f.readlines()
    a = lines[0].strip().split()
    x1, y1, x2, y2 = int(a[1]), int(a[2]), (int(a[1])+int(a[3])), (int(a[2])+int(a[4])) 

    sm = siammask.SiamMask()

    # Weight files are automatically retrieved from GitHub Releases
    sm.load_weights()

    # Adjust this parameter for the better mask prediction
    sm.box_offset_ratio = 1.5

    img_prev = np.array(PIL.Image.open("/mnt/DATA/Abhay/Downloads/tf-siammask/drone/"+video_folder+'/img/'+b1))[..., ::-1]
    box_prev = np.array([[x1, y1], [x2, y2]])
    img_next = np.array(PIL.Image.open("/mnt/DATA/Abhay/Downloads/tf-siammask/drone/"+video_folder+'/img/'+b2))[..., ::-1]

    # Predicted box and mask images is created if `debug=True`
    #box, mask = sm.predict(img_prev, box_prev, img_next, debug=True)
    #img=cv2.imread('predicted_mask.png')
    #cv2.imwrite('/mnt/DATA/Abhay/Downloads/tf-siammask/drone/'+video_folder+'/masks/'+str(i+2).zfill(8)+'.jpg', img)
    
    # img_prev = np.array(PIL.Image.open('/mnt/DATA/Abhay/Downloads/tf-siammask/drone/drone-14/img/00000001.jpg'))[..., ::-1]
    # box_prev = np.array([[x1, y1], [x2, y2]])
    # img_next = np.array(PIL.Image.open('/mnt/DATA/Abhay/Downloads/tf-siammask/drone/drone-14/img/00000002.jpg'))[..., ::-1]

   # Predicted box and mask images is created if `debug=True`
    box, mask = sm.predict(img_prev, box_prev, img_next, debug=True)
    print((box))
    mask_img=Image.open('predicted_mask.png')
    box_img = Image.open('predicted_box.png')
    mask_img.save('/mnt/DATA/Abhay/Downloads/tf-siammask/drone_masks/' +video_folder+ '/mask/'+str(i+2).zfill(8)+'.jpg') 
    box_img.save('/mnt/DATA/Abhay/Downloads/tf-siammask/drone_masks/' +video_folder+'/img/'+str(i+2).zfill(8)+'.jpg') 

    #Cropped Image & Mask

    bbox = list(box)
    n_bbox = []
    for arr in bbox:
        arr =list(arr)
        n_bbox.append(arr)

    [[cx1,cy1],[cx2,cy2]] = n_bbox
    cx1,cy1,cx2,cy2 = int(cx1),int(cy1), int(cx2), int(cy2)
    cropped_mask = mask_img.crop((cx1,cy1,cx2,cy2))
    cropped_img = box_img.crop((cx1,cy1,cx2,cy2))
    cropped_mask.save('/mnt/DATA/Abhay/Downloads/tf-siammask/drone_masks/'+video_folder+'/crop_mask/'+str(i+2).zfill(8)+'.jpg') 
    cropped_img.save('/mnt/DATA/Abhay/Downloads/tf-siammask/drone_masks/' +video_folder+'/crop_img/'+str(i+2).zfill(8)+'.jpg') 
