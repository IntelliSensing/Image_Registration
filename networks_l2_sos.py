import torch
import torch.nn as nn
import torch.nn.functional as f
import math

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s,padding=k//2, dilation=d, groups=g,bias=False)
        self.bn = nn.BatchNorm2d(c2)
        #self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
        self.act = nn.ReLU(c2) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class EmbeddingNet(nn.Module):
    """
    transform 64x64 patch to 128 dim features
    """
    def __init__(self,dim_desc=128,drop_rate=0.5):
        super(EmbeddingNet, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.eps_fea_norm = 1e-5
        self.eps_l2_norm = 1e-10
        self.embede = nn.Sequential(
            nn.InstanceNorm2d(1,self.eps_fea_norm),
            Conv(1,32,3,2), #d2
            Conv(32,32,3,1),
            Conv(32,32,3,1),
            Conv(32,64,3,2),#d4
            Conv(64,64,3,1), 
            Conv(64,128,3,2), #d8
            Conv(128,128,3,1), 
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, 8,1,bias=False),
            nn.BatchNorm2d(self.dim_desc)
        )
        self.desc_norm = nn.Sequential(
            nn.LocalResponseNorm(2 * self.dim_desc, alpha=2 * self.dim_desc, beta=0.5, k=0)
        )
        
    def forward(self, patch):
        descr = self.desc_norm(self.embede(patch)+self.eps_l2_norm)
        return descr.view(-1,self.dim_desc)



# for sarptical dataset
class EmbeddingNet_SARptical(nn.Module):
    """
    transform 64x64 patch to 128 dim features
    """
    def __init__(self,dim_desc=128,drop_rate=0.5):
        super(EmbeddingNet, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.eps_fea_norm = 1e-5
        self.eps_l2_norm = 1e-10
        self.embede = nn.Sequential(
            nn.InstanceNorm2d(1,self.eps_fea_norm),
            Conv(1,32,3,2), #d2
            Conv(32,32,3,1),
            Conv(32,32,3,1),
            Conv(32,64,3,2),#d4
            Conv(64,64,3,1), 
            Conv(64,128,3,2), #d8
            Conv(128,128,3,1), 
            Conv(128,128,3,2), #d16
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, 8,1,bias=False),
            nn.BatchNorm2d(self.dim_desc)
        )
        self.desc_norm = nn.Sequential(
            nn.LocalResponseNorm(2 * self.dim_desc, alpha=2 * self.dim_desc, beta=0.5, k=0)
        )
        
    def forward(self, patch):
        descr = self.desc_norm(self.embede(patch)+self.eps_l2_norm)
        return descr.view(-1,self.dim_desc)

class EmbeddingNet_ROIsummer(nn.Module):
    """
    transform 64x64 patch to 128 dim features
    """
    def __init__(self,dim_desc=128,drop_rate=0.5):
        super(EmbeddingNet_ROIsummer, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.eps_fea_norm = 1e-5
        self.eps_l2_norm = 1e-10
        self.embede = nn.Sequential(
            nn.InstanceNorm2d(1,self.eps_fea_norm),
            Conv(1,32,3,2), #d2
            Conv(32,32,3,1),
            Conv(32,32,3,1),
            Conv(32,64,3,2),#d4
            Conv(64,64,3,1), 
            Conv(64,64,3,2),#d8
            Conv(64,64,3,1), 
            Conv(64,128,3,2), #d16
            Conv(128,128,3,1), 
            Conv(128,128,3,2), #d32
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, 8,1,bias=False),
            nn.BatchNorm2d(self.dim_desc)
        )
        self.desc_norm = nn.Sequential(
            nn.LocalResponseNorm(2 * self.dim_desc, alpha=2 * self.dim_desc, beta=0.5, k=0)
        )
        
    def forward(self, patch):
        descr = self.desc_norm(self.embede(patch)+self.eps_l2_norm)
        return descr.view(-1,self.dim_desc)

class LDMNet(nn.Module):
    def __init__(self,EmbeddingNet):
        super(LDMNet,self).__init__()
        self.EmbeddingNet = EmbeddingNet
        self.weight_decay = 5e-4

    def get_embedding(self,x):
        return self.EmbeddingNet(x)

    def forward(self,p,a,margin=1,sos=False): 
        pe = self.EmbeddingNet(p)
        ae = self.EmbeddingNet(a)
        loss = self.loss(pe,ae,margin,sos)
        return loss
    
    def loss(self,pe,ae,margin=1,sos=False):
        d_pa = self.distance_pa(pe,ae)
        d_n,sos_item = self.distance_n(pe,ae,sos=sos)
        diff = f.relu(d_pa-d_n+margin).squeeze()
        loss = torch.sum(torch.pow(diff,2))/pe.shape[0] + sos_item
        return loss,diff

    def distance_pa(self,pe,ae):
        dist = torch.sqrt(torch.sum(torch.pow(pe-ae,2),dim=-1,keepdim=True))
        return dist

    def distance_n(self,ae,pe,k=1,sos=False):
        b,f = pe.size()
        mask = torch.eye(b).view(-1) == 0
        expand_pe1 = (pe.expand(b,b,f).reshape(-1,f))[mask,:] # [b*(b-1),f]
        expand_pe2 = (pe.reshape(b,1,f).expand(b,b,f).reshape(-1,f))[mask,:] # [b*(b-1),f]
        expand_ae1 = (ae.expand(b,b,f).reshape(-1,f))[mask,:]
        expand_ae2 = (ae.reshape(b,1,f).expand(b,b,f).reshape(-1,f))[mask,:]
        dist_pe1_ae2 = self.distance_pa(expand_pe1,expand_ae2).reshape(b,b-1)
        dist_pe2_ae1 = self.distance_pa(expand_pe2,expand_ae1).reshape(b,b-1)
        d_n = torch.cat([dist_pe1_ae2,dist_pe2_ae1],dim=-1)
        #d_n,_ = torch.min(d_n,dim=-1,keepdim=True) # (b,)
        d_n,_ = torch.topk(d_n,dim=-1,k=1,largest=False,sorted=False)
        sos_val = 0
        if sos:
            dist_pe = self.distance_pa(expand_pe1,expand_pe2).reshape(b,b-1) # (b,b-1)
            dist_ae = self.distance_pa(expand_ae1,expand_ae2).reshape(b,b-1)
            diff = torch.pow(dist_pe-dist_ae,2) #(b,b-1)
            _,index_pe = torch.topk(dist_pe,k=k,dim=-1,largest=False,sorted=False) # (b,k)
            _,index_ae = torch.topk(dist_ae,k=k,dim=-1,largest=False,sorted=False) 
            index = torch.cat([index_pe,index_ae],dim=-1) # (b,2k)
            sos_val = 0
            for i in range(b):
                sos_index = (index[i]).unique()
                sos_val += torch.sqrt(torch.sum(diff[i,sos_index]))
            sos_val = sos_val / b
            #sos_val = sos_val *self.weight_decay
        return d_n,sos_val
if __name__ == '__main__':
    embeddingNet = EmbeddingNet_ROIsummer()
    embeddingNet.cuda()
    ldmNet = LDMNet(embeddingNet)
    ldmNet.cuda()
    # print(ldmNet)
    p = torch.rand(16,1,256,256).cuda().float()
    a = torch.rand(16,1,256,256).cuda().float()
    loss = ldmNet(p,a)
    fa = ldmNet.get_embedding(a)
    print(loss)
    print(fa.shape)