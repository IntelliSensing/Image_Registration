import torch
from torch.utils.data import DataLoader
from networks_l2_sos import EmbeddingNet,LDMNet
import cv2
import numpy as np
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
global IM_POS
IM_POS = 0

def load_model(model,model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict']) 
    return 

def kpts2descriptors(kpts,img,model,batch_size=512,patch_size=64,use_gpu=True):
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

def match_images(img1,img2,model,fp_detector, thresh=1.20,knn=1):
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    normalize_img1 = cv2.normalize(img1,dst=None,alpha=450,beta=10,norm_type=cv2.NORM_MINMAX)
    normalize_img2 = cv2.normalize(img2,dst=None,alpha=450,beta=10,norm_type=cv2.NORM_MINMAX)
    kp1 = fp_detector.detect(normalize_img1, None)
    kp2 = fp_detector.detect(normalize_img2, None)
    desc_tfeat1 = kpts2descriptors(kp1,img1,model)
    desc_tfeat2 = kpts2descriptors(kp2,img2,model)
    #print('query.shape:',desc_tfeat1.shape)
    #print('train.shape:',desc_tfeat2.shape)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc_tfeat1,desc_tfeat2, k=knn)
    good = []
    for m in matches:
        for mm in m:
            if mm.distance < thresh:
                good.append(mm)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2).squeeze()
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2).squeeze()
    return src_pts, dst_pts

def init_model():
    fp_detector = cv2.FastFeatureDetector_create()
    embede_model = EmbeddingNet()
    embede_model.cuda()
    ldm_model = LDMNet(embede_model)
    ldm_model.cuda().eval()
    # load_model(ldm_model, "weights/rockets_180_0.009522625559423173_0.0018907563025210084_1.23.tar")
    load_model(ldm_model, "weights/rockets_60_0.0067_0.0015_1.23.tar")
    return embede_model, ldm_model, fp_detector

def reco(sar_img, opt_img):
    fp_detector = cv2.FastFeatureDetector_create()
    embede_model = EmbeddingNet()
    embede_model.cuda()
    ldm_model = LDMNet(embede_model)
    ldm_model.cuda().eval()
    load_model(ldm_model, "D:/myself/image_Registration/weights1/rockets_180_0.009522625559423173_0.0018907563025210084_1.23.tar")
    src_pts, dst_pts = match_images(sar_img, opt_img, embede_model, fp_detector, thresh=7,knn=2)
    return src_pts, dst_pts

if __name__ == '__main__':
    from PIL import Image
    sar_img = np.array(Image.open("D:/myself/image_Registration/OS_crop/sar2/sar2.png").convert('RGB'))
    opt_img = np.array(Image.open("D:/myself/image_Registration/OS_crop/sar2/opt2.png").convert('RGB'))
    opt_img = cv2.copyMakeBorder(opt_img, 0, 32, 0, 32, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    out_img = np.concatenate((sar_img, opt_img), axis=1)
    src_pts, dst_pts = reco(sar_img, opt_img )
    H,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,ransacReprojThreshold=16)
    for i in range(src_pts.shape[0]):
        if int(mask[i]):
            cv2.line(out_img,(int(src_pts[i,0]),int(src_pts[i,1])),(int(dst_pts[i,0])+512,int(dst_pts[i,1])),(0,255,0),1)
    cv2.imshow('img', out_img)
    cv2.waitKey(0)



    
        
