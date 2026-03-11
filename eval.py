import torch
from torch.utils.data import DataLoader
#from networks_l2_sos import EmbeddingNet,LDMNet
# from networks_angle_classification import EmbeddingNet,LDMNet
from networks_l2_sos import EmbeddingNet,LDMNet
from tqdm import tqdm
import cv2
import numpy as np
from collections import Counter
import math
import os
import time
from scipy import stats
from skimage import io
from PIL import Image

crop_value =32

def mse(src_pts, dst_pts, pix=5):
    """
    computer xmse, ymse, xymse
    """
    if src_pts.shape[0] == 0 :
        print("[WARNING] Empty point sets.")
        return float('nan'), float('nan'), float('nan'), 0.0

    srcxs = src_pts[:, 0]
    srcys = src_pts[:, 1]

    dstxs = dst_pts[:, 0]
    dstys = dst_pts[:, 1]


    pixel = np.multiply(np.abs(srcxs-dstxs) < pix , np.abs(srcys-dstys) < pix)
    print("\n", len(srcxs[pixel]), len(srcxs))
    rate = len(srcxs[pixel])/len(srcxs)

    srcxs = srcxs[pixel]
    srcys = srcys[pixel]
    dstxs = dstxs[pixel]
    dstys = dstys[pixel]

    xmse = np.sqrt(np.sum((dstxs-srcxs)*(dstxs-srcxs)) / len(srcxs))
    ymse = np.sqrt(np.sum((dstys-srcys)*(dstys-srcys)) / len(srcys))
    xymse = np.sqrt(xmse*xmse+ymse*ymse)

    return xmse, ymse ,xymse, rate

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
global IM_POS
IM_POS = 0

def load_model(model,model_path):
    ## load model from path
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return

def kpts2descriptors(kpts,img,model,batch_size=512,patch_size=64,use_gpu=True):
    """
    convert keypoints to descriptors"""
    descrs = []
    length = len(kpts)
    shards = int(np.ceil(length/batch_size))
    h,w = img.shape[:2]
    sx,sy,ex,ey = 0,0,0,0
    ltoffsets = patch_size//2
    rboffsets = patch_size-ltoffsets
    for i in range(shards):
        patches = []
        batch_kpts = kpts[i*batch_size:min((i+1)*batch_size,length)]
        for kp in batch_kpts:
            x, y = kp.pt
            x = int(x)
            y = int(y)
            if x <= ltoffsets:
                sx = 0
                ex = sx + patch_size
            else:
                ex = min(x+rboffsets,w)
                sx = ex-patch_size
            if y <= ltoffsets:
                sy = 0
                ey = sy + patch_size
            else:
                ey = min(y+rboffsets,h)
                sy = ey-patch_size
            patch = img[int(sy):int(ey),int(sx):int(ex)]
            assert patch.shape[0] == patch_size and patch.shape[1] == patch_size,str(patch.shape)+'  '+str(x)+' '+str(y)
            patches.append(patch)

        patches = torch.from_numpy(np.asarray(patches)).float()
        patches = torch.unsqueeze(patches, 1)
        if use_gpu:
            patches = patches.cuda()
        descrs.append(model(patches).detach().cpu().numpy())
    descrs = np.concatenate(descrs,axis=0)
    return descrs

def match_images(img1,img2,model,fp_detector,thresh=1.20,knn=1):
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) # 转换图片为灰度图
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) # 转换图片为灰度图
    normalize_img1 = cv2.normalize(img1,dst=None,alpha=450,beta=10,norm_type=cv2.NORM_MINMAX) # 归一化处理
    normalize_img2 = cv2.normalize(img2,dst=None,alpha=450,beta=10,norm_type=cv2.NORM_MINMAX) # 归一化处理
    kp1 = fp_detector.detect(normalize_img1, None) # 检测特征点
    kp2 = fp_detector.detect(normalize_img2, None) # 检测特征点
    desc_tfeat1 = kpts2descriptors(kp1,img1,model) # 提取特征点描述子
    desc_tfeat2 = kpts2descriptors(kp2,img2,model) #    提取特征点描述子
    #print('query.shape:',desc_tfeat1.shape)
    #print('train.shape:',desc_tfeat2.shape)
    bf = cv2.BFMatcher(cv2.NORM_L2) #   创建暴力匹配器
    matches = bf.knnMatch(desc_tfeat1,desc_tfeat2, k=knn) # 进行knn匹配
    good = []
    
    for m in matches:
        for mm in m:
            if mm.distance < thresh:
                good.append(mm)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2).squeeze()
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2).squeeze()
    ## will filter unhomography point
    H,mask = cv2.findHomography(src_pts,dst_pts,cv2.USAC_MAGSAC,ransacReprojThreshold=0.25,
                                           confidence=0.99999, maxIters=10000) # 计算单应性矩阵
    src_pts = src_pts[mask.squeeze().astype(bool)] # 过滤掉不符合单应性矩阵的点
    dst_pts = dst_pts[mask.squeeze().astype(bool)] #    过滤掉不符合单应性矩阵的点

    return  src_pts, dst_pts


def validate(model,fp_detector,data_path,save_path):
    """
    validate the model on the test set"""
    xmses = []
    ymses = []
    xymses = []
    rates = []
    test_targets = [(f"{data_path}/{x}", x) for x in os.listdir(data_path)][:90]
    for data_dir, x in tqdm(test_targets):

        sar_img = np.array(Image.open(f"{data_dir}/{x}.png").convert('RGB')) # sar image
        opt_img = np.array(Image.open(f"{data_dir}/{x.replace('sar', 'opt')}.png").convert('RGB')) # opt image
        src_pts, dst_pts = match_images(sar_img, opt_img,model,fp_detector,thresh=7,knn=2) # use opt as queryImg, sar as trainImg
        xmse, ymse, xymse, rate = mse(src_pts, dst_pts + crop_value,10) # compute mse
        if np.isnan(xmse):
            continue
        #print(round(xmse, 4), round(ymse, 4), round(xymse, 4))
        xmses.append(xmse)
        ymses.append(ymse)
        xymses.append(xymse)
        rates.append(rate)
    #correct_sa = np.mean(
    #        [[float(dist <= t) for t in thres] for dist in dists_sa], axis=0)
    print("mse:", round(np.mean(xmses), 4), round(np.mean(ymses), 4), round(np.mean(xymses), 4), "rate", round(np.mean(rates), 4))

if __name__ == '__main__':
    #model_path = './weights_angle_classification_b512_d0.5_p128_bi/rockets_33_3946.00_365.00.tar'
    _model_base_path = os.getcwd()  # 模型路径

    eval_path = 'E:/Storage/Pycharm/image_Registration/OS_crop' # 测试数据集路径

    embede_model = EmbeddingNet() # 特征提取网络
    embede_model.cuda() # 加载到GPU
    ldm_model = LDMNet(embede_model) # LDM网络
    ldm_model.cuda().eval() # 加载到GPU并设置为评估模式

    model_base_path = f'{_model_base_path}/weights/' # 模型路径

    best_acu_sa = [0,0,0,0] #   最优模型的准确率
    best_mma = []
    for model_name in os.listdir(model_base_path): # 遍历模型文件
        if model_name != 'rockets_60_0.0067_0.0015_1.23.tar':
            continue
        print(f"load model: {model_name}")
        load_model(ldm_model,model_base_path + model_name) # 加载模型
        fp_detector = cv2.FastFeatureDetector_create() # 特征检测器
        start = time.time()
        # test(embede_model,fp_detector,data_path=hard_test_path,data_root=hard_test_root,save_path='./eval_os')
        validate(embede_model,fp_detector,eval_path, save_path='./eval_results_most_common_anglesos_eval_15_knn2_bi') # 验证模型