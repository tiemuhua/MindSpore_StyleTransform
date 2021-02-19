from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

def get_pic_paths_in_folder(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.jpg':  
                L.append(os.path.join(root, file))
    return L
def load_np_image(file_path):
  '''input:a 3-D image
     Returns:numpy array ranging from 0 to 255
     dtype=unit8
  '''
  im = np.array(Image.open(file_path),dtype='float32')
  im=np.swapaxes(im,1,2)
  im=np.swapaxes(im,0,1)
  im=np.reshape(im,(1,im.shape[0],im.shape[1],im.shape[2]))
  return im

def Tensor_to_image(image,file_path):
    image=image.asnumpy()
    image=np.swapaxes(image,1,2)
    image=np.swapaxes(image,2,3)
    image=image.squeeze()
    plt.imshow(image)
    plt.savefig(file_path)


import numpy as np
import cv2


def crop_and_resize(file_path, size=256):
    image = resize_image(file_path)
    h, w = image.shape[:2]
    y = (h-size)//2
    x = (w-size)//2
    if len(image.shape)==2 :
        image=np.expand_dims(image,2).repeat(3,axis=2)
        print("??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????")
        print(image.shape)
    im = image[y:y+size, x:x+size, :]
    im=np.swapaxes(im,1,2)
    im=np.swapaxes(im,0,1)
    im=np.reshape(im,(1,im.shape[0],im.shape[1],im.shape[2]))
    return im


def resize_image(file_path, size=256, bias=5):
    image = np.array(Image.open(file_path),dtype='float32')
    image_shape = image.shape
    size_min = np.min(image_shape[:2])
    size_max = np.max(image_shape[:2])
    scale = 256 / float(size_min)
    image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)
    return image


x=crop_and_resize("/data/imagenet/train/n01491361/n01491361_7487.JPEG")
print(x.shape)
