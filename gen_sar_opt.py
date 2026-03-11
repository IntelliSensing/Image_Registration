
from tqdm import tqdm
import cv2
import numpy as np
from collections import Counter
import math
import os
import time
import random
from PIL import Image


def gen_patch(data_root,tar_path,train_or_val='train'):
    opt_img = [f'{data_root}/{train_or_val}/{x}' for x in os.listdir(f"{data_root}/{train_or_val}") if 'opt' in x]
    dep_img = [f'{x.replace("opt", "sar")}' for x in opt_img]

    with open(f'OS_{train_or_val}.txt', 'w') as train_txt:
        for opt_img_path, dep_img_path in zip(opt_img, dep_img):
            opt_img = cv2.imread(opt_img_path)
            h,w,c = opt_img.shape
            #h0 = random.sample([x for x in range(h-512)], 1)[0]
            #w0 = random.sample([x for x in range(w-512)], 1)[0]
            #opt_img = opt_img[h0:h0+512,w0:w0+512]
            print(opt_img_path, dep_img_path)
            opt_img = cv2.cvtColor(opt_img,cv2.COLOR_RGB2GRAY)
            dep_img = cv2.imread(dep_img_path)
            dep_img = cv2.cvtColor(dep_img,cv2.COLOR_RGB2GRAY)

            dep_patch = np.reshape(dep_img,(8,64,8,64)).transpose(0,2,1,3).reshape(64,64,64)[:,np.newaxis,...]
            opt_patch = np.reshape(opt_img,(8,64,8,64)).transpose(0,2,1,3).reshape(64,64,64)[:,np.newaxis,...]
            for i in random.sample([x for x in range(dep_patch.shape[0])], 20):
                opt_name = os.path.splitext(os.path.basename(opt_img_path))
                dep_name = os.path.splitext(os.path.basename(dep_img_path))
                os.makedirs(os.path.join(tar_path, train_or_val), exist_ok=True)
                dep_tar_path = os.path.join(tar_path, train_or_val, dep_name[0]+'_'+str(i)+'_sar.png')
                opt_tar_path = os.path.join(tar_path,train_or_val, opt_name[0]+'_'+str(i)+'_opt.png')
                #cv2.imwrite(opt_tar_path,opt_patch[i].squeeze())
                Image.fromarray(dep_patch[i].squeeze()).save(dep_tar_path, 'png', optimize=True)
                Image.fromarray(opt_patch[i].squeeze()).save(opt_tar_path, 'png', optimize=True)
                train_txt.write(' '.join([dep_tar_path, opt_tar_path, '\n']))


if __name__ == '__main__':

    data_root = 'E:/Storage/Pycharm/image_Registration/OSdataset/512/'
    patch_root = 'E:/Storage/Pycharm/image_Registration/OSdataset/patch/'
    #gen_patch(embede_model,test_path,data_root,test_tar_dir)
    gen_patch(data_root, patch_root, 'train')
    gen_patch(data_root,patch_root ,'val')
    gen_patch(data_root, patch_root,'test')