from networks_l2_sos import EmbeddingNet, LDMNet
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
class Config():
    pass

cfg = Config()
cfg.w_dict = {"1": 0.4, "2": 0.4, "3": 0.4,
              "4": 0.6, "5": 0.6, "6": 0.6, "7": 0.8, "8": 0.8}
cfg.device = 'cuda:0'
cfg.batch_size = 1024
cfg.epochs = 180
cfg.train_data = 'OS_train.txt'
cfg.test_data = 'OS_val.txt'
cfg.weights_dir = 'weights/'
if not os.path.exists(cfg.weights_dir):
    os.mkdir(cfg.weights_dir)

class RGBDDataset(Dataset):
    def __init__(self, label_file,transform=None,mode='train'):
        super(RGBDDataset, self).__init__()
        with open(label_file,'r') as f:
            self.samples = f.readlines()
        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'train':
            sar_path, opt_path, _ = self.samples[index].strip('\n').split(' ')
            optical_img = cv2.imread(opt_path, 0)
            sar_img = cv2.imread(sar_path, 0)
            flip = np.random.randint(0,2)
            rotation = np.random.randint(0,4)
            if flip:
                optical_img = cv2.flip(optical_img,1)
                sar_img = cv2.flip(sar_img,1)
            if rotation > 0:
                optical_img = cv2.rotate(optical_img,int(rotation-1))
                sar_img = cv2.rotate(sar_img,int(rotation-1))

            sample = {'optical':optical_img[np.newaxis,...], 'sar':sar_img[np.newaxis,...]}
            if self.transform:
                sample = self.transform(sample)
            return sample
        elif self.mode == 'eval':
            sar_path, opt_path,_ = self.samples[index].strip('\n').split(' ')
            optical_img = cv2.imread(opt_path, 0)
            sar_img = cv2.imread(sar_path, 0)

            sample = {'optical':optical_img[np.newaxis,...], 'sar':sar_img[np.newaxis,...]}
            if self.transform:
                sample = self.transform(sample)
            return sample
        elif self.mode == 'test':
            line = self.samples[index].strip('\n')
            sar_path,opt_path,label = line.strip('\n').split(' ')
            opt_img = cv2.imread(opt_path,0)[np.newaxis,...]
            sar_img = cv2.imread(sar_path,0)[np.newaxis,...]
            label = float(label)
            sample = {'optical':opt_img, 'sar':sar_img, 'label':label}
            if self.transform:
                sample = self.transform(sample)
            return sample
        else:
            return {}

    def __len__(self):
        return len(self.samples)


def train_one_epoch(model, data_loader, optimizer):
    model.train()
    tbar = tqdm(data_loader)
    total_train_loss = 0
    tacc = 0
    n_sample = 0
    for i, sample in enumerate(tbar):
        optical, sar, = sample['optical'].cuda().float(), sample['sar'].cuda().float()
        optimizer.zero_grad()
        loss, diff = model(sar, optical, sos=False)
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        diff_val = diff.detach().cpu().numpy()
        diff_val -= 0.7
        if len(diff_val.shape) == 2:
            diff_val = (np.max(diff_val,axis=-1) <= 0) * 1.0
        else:
            diff_val = (diff_val <=0) * 1.0
        loss_val = loss.item()
        total_train_loss += loss_val*diff_val.shape[0]
        tacc += np.sum(diff_val)
        n_sample += diff_val.shape[0]
        tbar.set_description('Train loss: %.3f' % (loss_val))
    return total_train_loss/n_sample, tacc/ n_sample


def validate(model, data_loader):
    model.eval()
    tbar = tqdm(data_loader)
    tacc = 0
    n_sample = 0
    t_loss = 0
    for i, sample in enumerate(tbar):
        optical, sar, = sample['optical'].cuda().float(), sample['sar'].cuda().float()

        # labels = torch.ones(sample.shape[0])
        labels = torch.ones(len(sample))

        with torch.no_grad():

            loss, diff = model(sar, optical)
        diff_val = diff.detach().cpu().numpy()
        diff_val -= 0.7

        # if i == (len(tbar)-1):
        #     print(diff_val)
        #     print(diff_val.shape)

        if len(diff_val.shape) == 2:
            diff_val = (np.max(diff_val,axis=-1) <= 0) * 1.0
        else:
            diff_val = (diff_val <=0) * 1.0
        #print(diff_val)
        loss_val = loss.item()
        t_loss += loss_val*diff_val.shape[0]
        n_sample += diff_val.shape[0]
        tacc += np.sum(diff_val)
        tbar.set_description('eval loss: %.3f' % (loss_val))
    return t_loss/n_sample,tacc/n_sample


def save_weights(model, optimizer,lr_scheduler, name, epoch, tacc,eacc,eval_loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict':lr_scheduler.state_dict()
    }
    path = "{}/{}_{}_{}_{}_{:.2f}.tar".format(
        cfg.weights_dir, name, epoch, round(tacc, 4), round(eacc, 4), round(eval_loss, 4))
    torch.save(checkpoint, path)


def load_weights(model, optimizer, lr_scheduler, model_path=None):
    if not os.path.exists(model_path):
        return -1
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    epoch = checkpoint['epoch']
    print('restore from %s'%(model_path))
    return epoch



if __name__ == '__main__':

    torch.cuda.set_device(cfg.device)
    embed_model = EmbeddingNet()
    embed_model.cuda().float()
    model = LDMNet(embed_model)
    model.cuda().float()

    # transform=torchvision.transforms.Compose([ToTensor()])
    train_loader = DataLoader(RGBDDataset(cfg.train_data, transform=None, mode='train'),
                              batch_size=cfg.batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(RGBDDataset(cfg.test_data, transform=None, mode='eval'),
                             batch_size=cfg.batch_size, shuffle=False, num_workers=1)
    #optimizer = torch.optim.SGD(model.parameters(),lr=1e-2,momentum=0.9,weight_decay=5e-4)
    warming_epoch = 5
    warming_optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.5)
    current_epoch = 0
    # current_epoch = load_weights(model, optimizer,lr_scheduler, 'weights_HardNet_SARptical/rockets_21_3015.0_0.0_1.15.tar') + 1
    eval_loss,acc = validate(model, test_loader)
    print('eval, epoch:{}, acc:{}, eval_loss:{:.2f}'.format(current_epoch, acc,eval_loss))

    min_eval_loss = 10000
    maxmemory = 20
    memory = 0
    for epoch in range(current_epoch, cfg.epochs):
        if epoch < warming_epoch:
            train_loss, train_acc = train_one_epoch(model, train_loader, warming_optimizer)
        else:
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
            lr_scheduler.step()
        print('train, epoch:{}, loss:{:.2f}, acc:{}'.format(epoch, train_loss, train_acc))
        eval_loss, eval_acc = validate(model, test_loader)
        print('eval, epoch:{}, loss:{:.2f}, acc:{}'.format(epoch, eval_loss,eval_acc))
        """
        if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            memory = 0
            save_weights(model, optimizer,lr_scheduler, 'rockets', epoch, train_acc, eval_acc,eval_loss)

        else:
            memory += 1
            if memory == maxmemory:
                save_weights(model, optimizer,lr_scheduler, 'rockets', epoch, train_acc, eval_acc,eval_loss)
                break
        """
        if epoch % 10 == 0 and epoch > 1:
            save_weights(model, optimizer,lr_scheduler, 'rockets', epoch, train_acc, eval_acc, eval_loss)