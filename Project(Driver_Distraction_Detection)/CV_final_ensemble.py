#!/usr/bin/env python
# coding: utf-8

# # Ensemble CNN's

# In[ ]:


from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.utils import shuffle
from sklearn import metrics

from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2 as cv
import time
import copy as cp
import pandas as pd
import pickle 

import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# # Data Loading

# In[ ]:


def load1(path,lbl):
    transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainloader = torch.utils.data.DataLoader(
          ImageFilelist(root=path,x=lbl, 
          transform=transform),
          batch_size=32, shuffle=False, pin_memory=True)
    
    return trainloader


# In[ ]:


def default_loader(path):
    im=Image.open(path).convert('RGB')
    return im
 
def default_flist_reader(flist,flag):
    imlist=[]
    lbl=['c0','c1', 'c2' ,'c3','c4','c5','c6','c7','c8','c9']
    if(flag=='training'):
       for i in lbl:
           p=flist+i+'\\'
           for file in glob.glob(p+'*jpg'):
              nm=lbl.index(i)
              imlist.append(cp.deepcopy((file,nm)))    
    else:
      p=flist
      for file in glob.glob(p+'*jpg'):
          nm=file.split('\\')
          nm=nm[len(nm)-1]
          imlist.append(cp.deepcopy((file,nm)))    
    print(len(imlist))
    return imlist
   
class ImageFilelist(data.Dataset):
     def __init__(self, root, x,transform=None, target_transform=None,
         flist_reader=default_flist_reader, loader=default_loader):
         self.imlist = flist_reader(root,x)
         self.transform = transform
         self.target_transform = target_transform
         self.loader = loader
   
     def __getitem__(self, index):
         impath, target = self.imlist[index]
         img = self.loader(impath)
         if self.transform is not None:
            img = self.transform(img)
         if self.target_transform is not None:
            target = self.target_transform(target)
         return img, target
   
     def __len__(self):
         return len(self.imlist)


# In[1]:


def train(trainloader,net):
    trn=[]; lbl=[]; c=0; t=time.time()
    with torch.no_grad():
         for data1 in trainloader:
             c+=1
             if(c%50==0):
                 print('Training: ',c,', Time: ',time.time()-t)
                 t=time.time()
             images, labels = data1
             images, labels = torch.tensor(images), torch.tensor(labels)
             images, labels = images.to(device), labels.to(device)
             out = net(images)
             for outputs in out:
                trn.append(cp.deepcopy(np.array(outputs.cpu(),np.double)))
             for label in labels:
                lbl.append(cp.deepcopy(np.array(label.cpu(),np.int32)))
    return np.array(trn),np.array(lbl)

def test(trainloader,net):
    trn=[]; lbl=[]; c=0; t=time.time()
    with torch.no_grad():
         for data1 in trainloader:
             c+=1
             if(c%50==0):
                 print('Testing: ',c,', Time: ',time.time()-t)
                 t=time.time()
             images, labels = data1
             images, labels = torch.tensor(images), labels
             images, labels = images.to(device), labels
             out = net(images)
             for outputs in out:
                trn.append(cp.deepcopy(np.array(outputs.cpu(),np.double)))
             for label in labels:
                lbl.append(cp.deepcopy(np.array(label,np.str)))
    return np.array(trn),np.array(lbl)


# In[ ]:


trn='C:\\Users\\arjun_000\\Desktop\\IIITD_notsync\\Dataset\\train\\'
tst='C:\\Users\\arjun_000\\Desktop\\IIITD_notsync\\Dataset\\test\\'


# In[ ]:


trainloader=load1(trn,'training')


# # CNN

# In[ ]:


net1 = torchvision.models.alexnet(True)
net2= torchvision.models.vgg16(True)
#net3=torchvision.models.densenet161(True)
#net4=torchvision.models.inception_v3(True)
net1.to(device)
net2.to(device)
net1.classifier=nn.Sequential(*list(net.classifier.children())[:-5])
net1.classifier=nn.Sequential(*list(net.classifier.children())[:-3])
#print(net)


# In[ ]:


data2,lbl2=train(trainloader,net1)
data3,lbl3=train(trainloader,net2)
print(data1.shape,lbl.shape)
print(data1.shape,lbl.shape)


# In[ ]:


data1=[]
for i in range(len(data2)):
    tmp=np.concatenate((data3[i],data3[i]))
    data1.append(tmp)
data1=np.array(data1)


# In[ ]:


Xtr=[]; Ytr=[]
Xte=[]; Yte=[]
for i in range(10):
    idx=lbl==i
    tmp_f=data1[idx]
    tmp_l=lbl[idx]
    tmp_f,tmp_l=shuffle(tmp_f,tmp_l)
    n=int(0.8*tmp_l.shape[0])              # 80 - 20 split
    for j in range(0,n):
        Xtr.append(tmp_f[j])
        Ytr.append(tmp_l[j])
    for j in range(n,tmp_l.shape[0]):
        Xte.append(tmp_f[j])
        Yte.append(tmp_l[j])
Xtr,Ytr=np.array(Xtr),np.array(Ytr)
Xte,Yte=np.array(Xte),np.array(Yte)
print(Xtr.shape,Ytr.shape)
print(Xte.shape,Yte.shape)


# # Classification and Results

# In[ ]:


clf=SVC(kernel='rbf',probability=True)
clf.fit(Xtr1,Ytr)


# In[ ]:


p_lbl=clf.predict(Xte1)
cmat=confusion_matrix(Yte,p_lbl)
sns.heatmap(cmat,annot=True,fmt='g')
print('Accuracy: %0.2f'%(clf.score(Xte1,Yte)*100))


# In[ ]:


prob=clf.predict_proba(Xte1)
scores=[]
for i in range(10):
    fpr, tpr, thresholds = metrics.roc_curve(Yte, prob[:,i], pos_label=i)
    plt.plot(fpr,tpr,label='C'+str(i))
    plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')  
plt.title('Alexnet+svm-rbf')


# # References

# In[ ]:


[1] https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
[2] https://pytorch.org/docs/stable/torchvision/models.html
[3] https://pytorch.org/docs/stable/data.html
[4] https://github.com/pytorch/vision/issues/81

