from PIL import Image
import numpy as np
import os


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
  im = np.array(Image.open(file_path))
  im=np.reshape(im,(1,im.shape[0],im.shape[1],im.shape[2]))
  return im
'''
def resize():
  return 0
im=load_np_image('eiffel_tower.jpg')
print(type(im))

print(im.dtype)

print(im.shape)

print(im)
'''
