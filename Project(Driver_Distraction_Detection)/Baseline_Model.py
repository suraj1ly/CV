
# coding: utf-8

# In[51]:


import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import pickle
import numpy.linalg as linalg
from sklearn.model_selection import train_test_split
from random import shuffle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
import seaborn
from numpy.random import choice
from sklearn.tree import DecisionTreeClassifier
import math
from sklearn import svm
import seaborn as sns
import matplotlib.patheffects as PathEffects
import torchvision.models as models
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.cm as cm
import imghdr
import tensorflow as tf


# In[52]:


from sklearn.svm import SVC


# In[53]:


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


# In[54]:


import warnings
warnings.filterwarnings('ignore')


# In[55]:


def mean_cal(data):
    main_mean=[]
    for i in range(len(data[0])):
        temp=[]
        for j in range(len(data)):
            temp.append(data[j][i])
        mean1=np.mean(np.array(temp))
        main_mean.append(mean1)
    return main_mean
        


# In[56]:


def cal_eigenenergy(arr,k):
    sum=0
    for i in range(len(arr)):
        sum+=arr[i]
    initial=arr[0]
    counter=0
    while initial<(sum*k/float(100)):
        counter+=1
        initial+=arr[counter]
    return counter+1
        


# In[57]:


def pca(data,k): # k is eigen energy
    data=np.array(data)
    mean=np.array(mean_cal(data))
    normal_data=[]
    for i in range(len(data)):
        normal_data.append(np.subtract(data[i],mean))
    normal_data=np.array(normal_data)
    cov_mat=np.cov(np.transpose(normal_data))
    eig_val, eig_vect = linalg.eigh(cov_mat)
    eig_vect=np.transpose(eig_vect)
#     eig_val=abs(eig_val)
    eig_val1=copy.deepcopy(eig_val)
    eig_vect1=copy.deepcopy(eig_vect)
    for i in range(len(eig_val)):
        if eig_val[i]<0:
            eig_val1[i]=eig_val[i]*(-1)
    
    eig_vect_set=sort_list(eig_vect1.tolist(),eig_val1.tolist())
    eig_vect_set.reverse()
    
    eig_val2=sorted(eig_val1, reverse=True)
    k=cal_eigenenergy(eig_val2,k)
#     print(k)
    eig_vect_set=eig_vect_set[:k]
#     print("Eig Vl:",eig_val)
#     print("Eig Vect : ",eig_vect_set)
    
#     print(eig_vect_set)
    print("Features it took are : ",k)
    eig_vect=np.transpose(eig_vect_set)
#     dot_result=np.dot(normal_data,eig_vect)
    return eig_vect


# In[58]:


#geeks for geeks 
def sort_list(list1, list2): 
  
    zipped_pairs = zip(list2, list1) 
  
    z = [x for _, x in sorted(zipped_pairs)] 
      
    return z 


# In[59]:


def labelling(predict,true):
    h=[]
    for i in range(len(predict)):
        if predict[i]==true[i]:
            h.append(1)
        else:
            h.append(0)
    return h


# In[60]:


def accuracy(predict,true):
    count=0
    for i in range(len(predict)):
        if predict[i]==true[i]:
            count+=1
    return count/float(len(predict))


# In[61]:


def find_tpr_fpr(predict,real,checker):
    tp=0
    tn=0
    fp=0
#     print("find_tpr_fpr")
    fn=0
    voc=copy.deepcopy([0,1,2,3,4,5,6,7,8,9])
    v=voc.index(checker)
    del voc[v]
    for i in range(len(predict)):
        if predict[i]==checker and real[i]==checker:
            tp=tp+1
        if (predict[i] in voc ) and real[i]==checker:
            fn=fn+1
        if predict[i]==checker and (real[i] in voc):
            fp=fp+1
        if (predict[i] in voc) and (real[i] in voc):
            tn=tn+1
    tpr2=0
    fpr2=0
#     print("Total :",(tp+fp+tn+fn))
    tpr2=float(tp/float(tp+fn))   
    fpr2=float(fp/float(fp+tn))
    
    return tpr2,fpr2


# In[62]:


def adaboost(n,train_data1,train_label1,test_data,test_label,weights,d):
    
    alpha_k=[]
    Ck=[]
    nat=[i for i in range(len(train_data1))]
    main_data=copy.deepcopy(train_data1)
    main_label=copy.deepcopy(train_label1)
    
    for i in range(n):
        print(i)
#         print("Hello : ",i)
        sample = choice(nat, d,p=weights,replace=False)
#         sample = choice(nat,d,weights,replace=False)
        train_data=[]
        train_label=[]
        for j in range(len(sample)):
            train_data.append(train_data1[sample[j]])
            train_label.append(train_label1[sample[j]])
        
        clf=DecisionTreeClassifier(max_depth=3,max_leaf_nodes=10)
        clf.fit(np.array(train_data),np.array(train_label))
        predict1=clf.predict(np.array(main_data))
        h=labelling(predict1.tolist(),main_label)
        train_err=clf.score(np.array(train_data),np.array(train_label))
        train_err=1-train_err
        alpha=0.5*np.log((1-train_err)/float(train_err))+np.log(25)
        alpha_k.append(alpha)
        Ck.append(clf)
#         print("Hello1 : ",i)
        for j in range(len(weights)):
            
            if h[j]==1:
                
                weights[j]=weights[j]*math.exp((-1)*alpha)
            else:
                weights[j]=weights[j]*math.exp(alpha)
        w=copy.deepcopy(weights)
        total=np.sum(w)
        for j in range(len(weights)):
            weights[j]=weights[j]/float(total)
#         print("Hello2 : ",i)
 #For test set
    test_predict=[]
    for i in range(len(test_data)):
        disc_func=[[] for i in class_label]
        for j in range(k_max):
            index=Ck[j].predict(np.array(test_data[i]).reshape(1,-1)).tolist()[0]
            if disc_func[index]==[]:
                disc_func[index].append(alpha_k[j])
            else:
                disc_func[index][0]+=alpha_k[j]
        test_predict.append(disc_func.index(max(disc_func)))

    test_acc=accuracy(test_predict,test_label)   
# For train set
    train_predict=[]
#     for i in range(len(train_data1)):
#         disc_func1=[[] for i in class_label]
#         for j in range(k_max):
#             index=Ck[j].predict(np.array(train_data1[i]).reshape(1,-1)).tolist()[0]
#             if disc_func1[index]==[]:
#                 disc_func1[index].append(alpha_k[j])
#             else:
#                 disc_func1[index][0]+=alpha_k[j]
#         train_predict.append(disc_func1.index(max(disc_func1)))

#     train_acc=accuracy(train_predict,train_label1)  
    train_acc=0
    return Ck,alpha_k,train_predict,test_predict,train_acc,test_acc
            


# In[63]:


def bagging(n,train_data1,train_label1,test_data,test_label,d):
    
    alpha_k=[]
    Ck=[]
    nat=[i for i in range(len(train_data1))]

    for i in range(n):
        sample = choice(nat, d,replace=True)
        train_data=[]
        train_label=[]
        for j in range(len(sample)):
            train_data.append(train_data1[sample[j]])
            train_label.append(train_label1[sample[j]])

        clf=DecisionTreeClassifier(max_depth=2,max_leaf_nodes=5)
        clf.fit(np.array(train_data),np.array(train_label))
        Ck.append(clf)
    #For test set
    test_predict=[]

    for i in range(len(test_data)):
        disc_func=[[] for i in class_label]
        for j in range(k_max):
            index=Ck[j].predict(np.array(test_data[i]).reshape(1,-1)).tolist()[0]
            if disc_func[index]==[]:
                disc_func[index].append(1)
            else:
                disc_func[index][0]+=1
        test_predict.append(disc_func.index(max(disc_func)))

    test_acc=accuracy(test_predict,test_label)   
    train_acc=0
#     print("Accuracy in test data : ",acc)
#     #For training set
    train_predict=[]

#     for i in range(len(train_data1)):
#         disc_func=[[] for i in class_label]
#         for j in range(k_max):
#             index=Ck[j].predict(np.array(train_data1[i]).reshape(1,-1)).tolist()[0]
#             if disc_func[index]==[]:
#                 disc_func[index].append(1)
#             else:
#                 disc_func[index][0]+=1
#         train_predict.append(disc_func.index(max(disc_func)))

#     train_acc=accuracy(train_predict,train_label1)   
    

    return Ck,train_predict,test_predict,train_acc,test_acc
            


# In[64]:


def roc_design(prob_dist,testdata,checker):
    aux1=[]
    aux2=[]
    testdata1=copy.deepcopy(testdata)
    for i in range(len(testdata)):
        
        aux1.append(prob_dist[i])
        aux2.append(testdata[i])
    main1=sort_list(aux2, aux1)
#     print("Probability in incresing order : ",main1)
    
    tpr=[]
    fpr=[]
    #aux1 has prob_distribution and main1 has testlabel in sorted order
   
    main2=[]
    j=0
    for j in range(len(prob_dist)):
        main2.append(checker)
    i=0
    #Logic 
    if (checker+1)==10:
        flag=checker-1
    else:
        flag=checker+1
        
    while i <len(prob_dist):
        tpr1=0
        fpr1=0
        j=0
        
        while (j  <= i):
            main2[j]=flag
            j=j+1
#         j=i
#         while j <len(prob_dist):
#             main1[j]=2
#             j=j+1
#         print(main1)
#         m=[]
#         tpr.append(tpr1)
#         fpr.append(fpr1)
        tpr1,fpr1=find_tpr_fpr(copy.deepcopy(main2),copy.deepcopy(main1),checker)
#         e.append(testdata)
#         d.append(main1)
        fpr.append(fpr1)
        tpr.append(tpr1)
        
        i=i+50
    return tpr,fpr


# In[65]:


#Cross validation 
def cross_validation_svm(train_new,train_label2,train_size):

    fold=5
    model=[]
    score_set=[]
    if len(train_new)%5==0:
        length=int(len(train_new)/5)
    else:
        length=int(len(train_new)/5)+1
    newlength=length
    counter=0
    for q in range(fold):
        valid_test_data=[]
        valid_test_label=[]
        valid_train_data=[]
        valid_train_label=[]
        if (max(len(train_new),length)==length):
            length=len(train_new)
#         print("Counter : ",counter)
#         print("End : ",length)
        for j in range(counter,length):
            valid_test_data.append(train_new[j])
            valid_test_label.append(train_label2[j])
        counter=counter+newlength
        length=length+newlength
        valid_test_data=valid_test_data
        valid_test_label=valid_test_label
        for j in range(len(train_new)):
            if train_new[j] not in valid_test_data:
                valid_train_data.append(train_new[j])
                valid_train_label.append(train_label2[j])
#         print(q)
#         print("Data : ",valid_train_data)
#         clf = LogisticRegressionCV(random_state=0, solver='lbfgs',multi_class='multinomial').fit(copy.deepcopy(np.array(valid_train_data)), copy.deepcopy(np.array(valid_train_label)))
#         clf = GaussianNB().fit(np.array(valid_train_data), np.array(valid_train_label))
        clf = svm.SVC(gamma=0.001)
        clf.fit(np.array(valid_train_data), np.array(valid_train_label))
        a=[]
        a=clf.predict(np.array(valid_test_data))
        score=clf.score(np.array(valid_test_data),np.array(valid_test_label))
        score_set.append(score)
        model.append(clf) 
        del clf


    return model,score_set
    
    


# In[66]:


#Cross validation 
def cross_validation_gaussian(train_new,train_label2,train_size):

    fold=5
    model=[]
    score_set=[]
    if len(train_new)%5==0:
        length=int(len(train_new)/5)
    else:
        length=int(len(train_new)/5)+1
    newlength=length
    counter=0
    for q in range(fold):
        valid_test_data=[]
        valid_test_label=[]
        valid_train_data=[]
        valid_train_label=[]
        if (max(len(train_new),length)==length):
            length=len(train_new)
#         print("Counter : ",counter)
#         print("End : ",length)
        for j in range(counter,length):
            valid_test_data.append(train_new[j])
            valid_test_label.append(train_label2[j])
        counter=counter+newlength
        length=length+newlength
        valid_test_data=valid_test_data
        valid_test_label=valid_test_label
        for j in range(len(train_new)):
            if train_new[j] not in valid_test_data:
                valid_train_data.append(train_new[j])
                valid_train_label.append(train_label2[j])
#         print(q)
#         print("Data : ",valid_train_data)
#         clf = LogisticRegressionCV(random_state=0, solver='lbfgs',multi_class='multinomial').fit(copy.deepcopy(np.array(valid_train_data)), copy.deepcopy(np.array(valid_train_label)))
        clf = GaussianNB().fit(np.array(valid_train_data), np.array(valid_train_label))
        a=[]
        a=clf.predict(np.array(valid_test_data))
        score=clf.score(np.array(valid_test_data),np.array(valid_test_label))
        score_set.append(score)
        model.append(clf) 
        del clf


    return model,score_set
    
    


# In[67]:


#Cross validation 
def cross_validation_logistic(train_new,train_label2,train_size):

    fold=5
    model=[]
    score_set=[]
    if len(train_new)%5==0:
        length=int(len(train_new)/5)
    else:
        length=int(len(train_new)/5)+1
    newlength=length
    counter=0
    for q in range(fold):
        valid_test_data=[]
        valid_test_label=[]
        valid_train_data=[]
        valid_train_label=[]
        if (max(len(train_new),length)==length):
            length=len(train_new)
#         print("Counter : ",counter)
#         print("End : ",length)
        for j in range(counter,length):
            valid_test_data.append(train_new[j])
            valid_test_label.append(train_label2[j])
        counter=counter+newlength
        length=length+newlength
        valid_test_data=valid_test_data
        valid_test_label=valid_test_label
        for j in range(len(train_new)):
            if train_new[j] not in valid_test_data:
                valid_train_data.append(train_new[j])
                valid_train_label.append(train_label2[j])
#         print(q)
#         print("Data : ",valid_train_data)
        clf = LogisticRegressionCV(random_state=0, solver='lbfgs',multi_class='multinomial').fit(np.array(valid_train_data), np.array(valid_train_label))
#         clf = GaussianNB().fit(np.array(valid_train_data), np.array(valid_train_label))
        a=[]
        a=clf.predict(np.array(valid_test_data))
        score=clf.score(np.array(valid_test_data),np.array(valid_test_label))
        score_set.append(score)
        model.append(clf) 
        del clf


    return model,score_set
    
    


# In[68]:


# os.chdir('CV')


# In[71]:


os.chdir('..')


# In[72]:


pwd


# In[73]:


os.chdir('train')


# In[74]:


list_dir=os.listdir()


# In[75]:


pwd


# In[76]:


#For dumping dataset for Face data here initial 800 for each class is treated as test set and others as train set.
train_data=[]
train_label=[]
test_data=[]
test_label=[]
count1=0
for i in list_dir:
    
    print(i)
    os.chdir(i)
    list_doc=os.listdir()
    temp1=[]
    temp=[]
    count=0
    for j in list_doc:
        count+=1
        img=cv2.imread(j,0)
        img=cv2.resize(img,(48,48))
   
         
#         print(img)
        img=img.ravel()
        if count<800:
            test_data.append(list(img))
            test_label.append(count1)
        else:
 
            train_data.append(list(img))
            train_label.append(count1)
    count1+=1
    os.chdir('..')


# In[77]:


# For shuffling the data set
main_data_set=[]
for i in range(len(train_data)):
    temp=[]
    temp.append(train_data[i])
    temp.append(train_label[i])
    main_data_set.append(temp)
shuffled_data=random.sample(main_data_set, len(main_data_set))
train_data=[]
train_label=[]
for i in range(len(shuffled_data)):
    train_data.append(shuffled_data[i][0])
    train_label.append(shuffled_data[i][1])


# In[78]:


fold=5
train_size=len(train_data)
valid_size=int(train_size/float(fold))


# In[79]:


#Visualisation 
#Visualization of Data 
def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))
    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


# In[80]:


# #Dumping into train data and label.
# with open("train_data_cv.txt", 'wb') as f:
#     pickle.dump(train_data, f)
# with open("train_label_cv.txt", 'wb') as f:
#     pickle.dump(train_label, f)


# In[ ]:


os.chdir('..')


# In[ ]:


#For loading the train data and train label
train_data = pickle.load(open("train_data_cv.txt","rb"))
train_label= pickle.load(open("train_label_cv.txt","rb"))


# In[ ]:


# #Visualisation of Data 

# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# pca = PCA(n_components=30)
# train_data1=pca.fit_transform(np.array(train_data))
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(np.array(train_data1))# fashion_scatter(tsne_results, np.array(train_label))


# In[ ]:


# fashion_scatter(tsne_results, np.array(train_label))


# In[92]:


os.chdir('data')


# In[93]:


pwd


# In[94]:


# For CNN
#Writing into directory named as data
s="Image"
g=".jpg"
counter=0
for i in range(len(train_data)):
    counter+=1
    f=os.listdir()
    if str(train_label[i]) not in f:
        os.mkdir(str(train_label[i]))
        img=np.array(train_data[i],dtype=np.uint8).reshape(48,48)
        os.chdir(str(train_label[i]))
#         plt.imshow(np.array(train_data[i]).reshape(48,48))
        cv2.imwrite(s+str(counter)+g,img/np.max(img)*255)
        os.chdir('..')
    else:
        img=np.array(train_data[i],dtype=np.uint8).reshape(48,48)
        os.chdir(str(train_label[i]))
        cv2.imwrite(s+str(counter)+g,img/np.max(img)*255)
        os.chdir('..')
        


# In[96]:


os.chdir('..')


# In[97]:


os.chdir('testdata')


# In[98]:


#Writing into directory named as testdata
s="Image"
g=".jpg"
counter=0
for i in range(len(test_data)):
    counter+=1
    f=os.listdir()
    if str(test_label[i]) not in f:
        os.mkdir(str(test_label[i]))
        img=np.array(test_data[i],dtype=np.uint8).reshape(48,48)
        os.chdir(str(test_label[i]))
#         plt.imshow(np.array(train_data[i]).reshape(48,48))
        cv2.imwrite(s+str(counter)+g,img/np.max(img)*255)
        os.chdir('..')
    else:
        img=np.array(test_data[i],dtype=np.uint8).reshape(48,48)
        os.chdir(str(test_label[i]))
        cv2.imwrite(s+str(counter)+g,img/np.max(img)*255)
        os.chdir('..')
        


# In[99]:


import torch.utils.data as data

from PIL import Image
import os
import os.path

def default_loader(path):
	return Image.open(path).convert('RGB')

def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    os.chdir(flist)
    f=os.listdir()
    for i in range(len(f)):
        os.chdir(f[i])
        g=os.listdir()
        for j in range(len(g)):
            imlist.append(((os.getcwd()+str('/')+str(g[j])), int(f[i])))

        os.chdir('..')

    return imlist
class ImageFilelist(data.Dataset):
	def __init__(self, root, flist, transform=None, target_transform=None,
			flist_reader=default_flist_reader, loader=default_loader):
		self.root   = root
		self.imlist = flist_reader(flist)		
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader

	def __getitem__(self, index):
		impath, target = self.imlist[index]
		img = self.loader(os.path.join(self.root,impath))
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		
		return img, target

	def __len__(self):
		return len(self.imlist)


# In[102]:


os.chdir('..')


# In[103]:


transform = transforms.Compose(
    [transforms.Resize(224),transforms.ToTensor()])
trainloader = torch.utils.data.DataLoader(ImageFilelist(root="./data/", flist="./data/",
                                                         transform=transform),batch_size=4,
                                          shuffle=True, num_workers=2)
os.chdir('..')
testloader = torch.utils.data.DataLoader(ImageFilelist(root="./testdata/", flist="./testdata/",
                                                         transform=transform),batch_size=4,
                                          shuffle=True, num_workers=2)


# In[104]:


classes = (0,1,2,3,4,5,6,7,8,9)


# In[137]:


alexnet = models.alexnet(pretrained=True)
alexnet.to("cuda:0")
alexnet.add_module(module=nn.Linear(*list(alexnet.children())[:-3]))
# list(vgg_baseline.children())[:-1][-1]


# In[81]:


# a=[]
# for i in range(len(train_data)):
   
#     gray=np.array(train_data[1],dtype=np.uint8).reshape(48,48)
#     faces = face_cascade.detectMultiScale(gray,1.3,5)
#     a.append(faces)
#     for (x,y,w,h) in faces:
#         img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]
        


# In[117]:


#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
train_data=[]
train_label=[]
for i, data in enumerate(trainloader, 0):
    # get the inputs
    inputs,labels=data
    inputs,labels=inputs.to("cuda:0"), labels.to("cuda:0")

    outputs = resnet18(inputs)
    g=outputs.cpu().detach().numpy().tolist()
    h=labels.cpu().numpy().tolist()
    for j in range(len(g)):

        train_data.append(g[j])
        train_label.append(h[j])
#         print(h[i])
   


# In[118]:


test_data=[]
test_label=[]
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images,labels=images.to("cuda:0"), labels.to("cuda:0")
        outputs = resnet18(images)
        g=outputs.cpu().numpy().tolist()
        h=labels.cpu().numpy().tolist()
        for i in range(len(g)):
            test_data.append(g[i])
            test_label.append(h[i])


# In[119]:


train_size=len(train_data)


# In[120]:


# Cross Validation

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import csv
from xgboost import XGBClassifier


# In[112]:


#PCA 
import seaborn as sns
from sklearn.decomposition import PCA
pca = PCA(n_components=100)
train_data1=pca.fit_transform(np.array(train_data))
test_data1=pca.transform(np.array(test_data))


# In[121]:


X_train, X_test, y_train = np.array(train_data),np.array(test_data),np.array(train_label)


# In[122]:


kf=KFold(n_splits=5)
model=[]
score_set1=[]
for train_index, test_index in kf.split(X_train):
    print("TRAIN:", train_index, "TEST:", test_index)
    valid_train_data, valid_test_data = X_train[train_index], X_train[test_index]
    valid_train_label, valid_test_label = y_train[train_index], y_train[test_index]
    
    clf = LogisticRegressionCV(random_state=0, solver='lbfgs',multi_class='multinomial')
    clf.fit(np.array(valid_train_data),np.array(valid_train_label))

    b=clf.predict(np.array(valid_test_data))

    score1=clf.score(valid_test_data,valid_test_label)
    print("Accuracy ",score1)
    score_set1.append(score1)
    model.append(clf) 


# In[123]:


score_mean=np.mean(score_set1)
standard_dev=np.std(score_set1)
print("Standard deviation : ",standard_dev)
print("Mean of accuracy :",score_mean)
print("Accuracy for each validation : ",score_set1)
best=model[np.argmax(score_set1)]
best_score6=best.score(np.array(test_data),np.array(test_label))
prob_dist6=best.predict_proba(test_data)
label_pred=best.predict(test_data).tolist()
print("Test Accuracy by best model in cross validation : ",best_score6)


# # Using Feature as Intensity Value

# # Implementing PCA

# In[ ]:


transf_mat=pca(train_data,80)
train_data1=np.dot(train_data,transf_mat).real
test_data1=np.dot(test_data,transf_mat)


# In[ ]:


len(train_label)


# In[ ]:


len(test_label)


# In[ ]:


clf = LogisticRegressionCV(random_state=0, solver='lbfgs',multi_class='multinomial').fit(copy.deepcopy(np.array(train_data1)), copy.deepcopy(np.array(train_label)))
label_predict=clf.predict(np.array(test_data1)).tolist()
score=clf.score(np.array(test_data1),np.array(test_label))
print("Accuracy : ",score)


# # Using Logistic Regression Algorithm

# In[ ]:


model,score_set=cross_validation_logistic(train_data,train_label,train_size)


# In[ ]:


score_mean=np.mean(np.array(score_set))
standard_dev=np.std(np.array(score_set))
print("Standard deviation : ",standard_dev)
print("Mean of accuracy :",score_mean)
print("Accuracy for each validation : ",score_set)
best=model[np.argmax(score_set)]
best_score=best.score(test_data,test_label)
prob_dist=best.predict_proba(test_data).tolist()
label_pred=best.predict(test_data).tolist()
print("Test Accuracy by best model in cross validation : ",best_score)


# In[ ]:


# clf = LogisticRegressionCV(random_state=0, solver='lbfgs',multi_class='multinomial').fit(copy.deepcopy(np.array(train_data)), copy.deepcopy(np.array(train_label)))
# label_predict=clf.predict(np.array(test_data)).tolist()
# score=clf.score(np.array(test_data),np.array(test_label))
# print("Accuracy : ",score)


# In[ ]:


# prob_dist=clf.predictproba(np.array(test_data),np.array(test_label)).tolist()
# prob_dist=np.transpose(prob_dist)


# In[ ]:


#Confusion Matrix
confusionmatrix=[]
temp=[0,0,0,0,0,0,0,0,0,0]
for i in range(10):
    confusionmatrix.append(temp)
    temp=copy.deepcopy(temp)
for i in range(len(test_label)):
    confusionmatrix[test_label[i]][label_pred[i]]=confusionmatrix[test_label[i]][label_pred[i]]+1
seaborn.heatmap(confusionmatrix,annot=True,fmt="d")   


# In[ ]:


# ROC Curve 
prob_dist=np.transpose(prob_dist)
for i in range(10):
    tpr,fpr=roc_design(prob_dist[i],test_label,i)
    plt.plot(fpr, tpr ,label="Class"+str(i+1))
# roccurve(prob_dist,test_label)
plt.xlabel("False +ve Rate")
plt.ylabel("True +ve Rate")
plt.legend()
plt.title("ROC Curve for Class 1 to 11")
plt.show()


# # Using Naive Bayes Algorithm

# In[ ]:


model,score_set=cross_validation_gaussian(train_data,train_label,train_size)


# In[ ]:


score_mean=np.mean(np.array(score_set))
standard_dev=np.std(np.array(score_set))
print("Standard deviation : ",standard_dev)
print("Mean of accuracy :",score_mean)
print("Accuracy for each validation : ",score_set)
best=model[np.argmax(score_set)]
best_score=best.score(test_data,test_label)
prob_dist=best.predict_proba(test_data).tolist()
label_pred=best.predict(test_data).tolist()
print("Test Accuracy by best model in cross validation : ",best_score)


# In[ ]:


# clf = GaussianNB().fit(copy.deepcopy(np.array(train_data)), copy.deepcopy(np.array(train_label)))
# label_predict=clf.predict(np.array(test_data)).tolist()
# score=clf.score(np.array(test_data),np.array(test_label))
# print("Accuracy : ",score)


# In[ ]:


# prob_dist=clf.predict_proba(np.array(test_data)).tolist()
# prob_dist=np.transpose(prob_dist)


# In[ ]:


# #Confusion Matrix
# confusionmatrix=[]
# temp=[0,0,0,0,0,0,0,0,0,0]
# for i in range(10):
#     confusionmatrix.append(temp)
#     temp=copy.deepcopy(temp)
# for i in range(len(test_label)):
#     confusionmatrix[test_label[i]][label_predict[i]]=confusionmatrix[test_label[i]][label_predict[i]]+1
# seaborn.heatmap(confusionmatrix,annot=True,fmt="d")   


# In[ ]:


# # ROC Curve 
# # prob_dist=np.transpose(prob_dist)
# for i in range(10):
#     print(i)
#     tpr,fpr=roc_design(prob_dist[i],test_label,i)
#     plt.plot(fpr, tpr ,label="Class"+str(i+1))
# # roccurve(prob_dist,test_label)
# plt.xlabel("False +ve Rate")
# plt.ylabel("True +ve Rate")
# plt.legend()
# plt.title("ROC Curve for Class 1 to 11")
# plt.show()


# # SVM for multiclass

# In[ ]:


model,score_set=cross_validation_svm(train_data,train_label,train_size)


# In[ ]:


score_mean=np.mean(np.array(score_set))
standard_dev=np.std(np.array(score_set))
print("Standard deviation : ",standard_dev)
print("Mean of accuracy :",score_mean)
print("Accuracy for each validation : ",score_set)
best=model[np.argmax(score_set)]
best_score=best.score(test_data,test_label)
prob_dist=best.predict_proba(test_data).tolist()
label_pred=best.predict(test_data).tolist()
print("Test Accuracy by best model in cross validation : ",best_score)


# In[ ]:


# clf = SVC(gamma='auto')
# clf.fit(np.array(train_data),np.array(train_label))
# label_predict=clf.predict(np.array(test_data)).tolist()
# score=clf.score(np.array(test_data),np.array(test_label))
# print("Accuracy : ",score)


# In[ ]:


# prob_dist=clf.predict_proba(np.array(test_data)).tolist()
# prob_dist=np.transpose(prob_dist)


# In[ ]:


# #Confusion Matrix
# confusionmatrix=[]
# temp=[0,0,0,0,0,0,0,0,0,0]
# for i in range(10):
#     confusionmatrix.append(temp)
#     temp=copy.deepcopy(temp)
# for i in range(len(test_label)):
#     confusionmatrix[test_label[i]][label_predict[i]]=confusionmatrix[test_label[i]][label_predict[i]]+1
# seaborn.heatmap(confusionmatrix,annot=True,fmt="d")   


# In[ ]:


# # ROC Curve 
# # prob_dist=np.transpose(prob_dist)
# for i in range(10):
#     print(i)
#     tpr,fpr=roc_design(prob_dist[i],test_label,i)
#     plt.plot(fpr, tpr ,label="Class"+str(i+1))
# # roccurve(prob_dist,test_label)
# plt.xlabel("False +ve Rate")
# plt.ylabel("True +ve Rate")
# plt.legend()
# plt.title("ROC Curve for Class 1 to 11")
# plt.show()


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


# # Using histogram values as features

# In[ ]:


def histogram(train_data):
    hist=[0]*256
    for i in range(len(train_data)):
        hist[train_data[i]]+=1
    return hist
        


# In[ ]:


train_data1=[]
for i in range(len(train_data)):
    train_data1.append(histogram(train_data[i]))
test_data1=[]
for i in range(len(test_data)):
    test_data1.append(histogram(test_data[i]))


# In[ ]:


# train_data1[0]


# In[ ]:


train_data=copy.deepcopy(train_data1)
test_data=copy.deepcopy(test_data1)


# # Naive Bayes Algorithm

# In[ ]:


clf = GaussianNB().fit(copy.deepcopy(np.array(train_data)), copy.deepcopy(np.array(train_label)))
label_predict=clf.predict(np.array(test_data)).tolist()
score=clf.score(np.array(test_data),np.array(test_label))
print("Accuracy : ",score)


# In[ ]:


prob_dist=clf.predict_proba(np.array(test_data)).tolist()
prob_dist=np.transpose(prob_dist)


# In[ ]:


#Confusion Matrix
confusionmatrix=[]
temp=[0,0,0,0,0,0,0,0,0,0]
for i in range(10):
    confusionmatrix.append(temp)
    temp=copy.deepcopy(temp)
for i in range(len(test_label)):
    confusionmatrix[test_label[i]][label_predict[i]]=confusionmatrix[test_label[i]][label_predict[i]]+1
seaborn.heatmap(confusionmatrix,annot=True,fmt="d")   


# In[ ]:


# ROC Curve 
# prob_dist=np.transpose(prob_dist)
for i in range(10):
    print(i)
    tpr,fpr=roc_design(prob_dist[i],test_label,i)
    plt.plot(fpr, tpr ,label="Class"+str(i+1))
# roccurve(prob_dist,test_label)
plt.xlabel("False +ve Rate")
plt.ylabel("True +ve Rate")
plt.legend()
plt.title("ROC Curve for Class 1 to 11")
plt.show()


# # Logistic Regression

# In[ ]:


clf = LogisticRegressionCV(random_state=0, solver='sag',multi_class='multinomial',n_jobs=4).fit(np.array(train_data), np.array(train_label))
label_predict=clf.predict(np.array(test_data)).tolist()
score=clf.score(np.array(test_data),np.array(test_label))
print("Accuracy : ",score)


# In[ ]:


prob_dist=clf.predictproba(np.array(test_data),np.array(test_label)).tolist()
prob_dist=np.transpose(prob_dist)


# In[ ]:


#Confusion Matrix
confusionmatrix=[]
temp=[0,0,0,0,0,0,0,0,0,0]
for i in range(10):
    confusionmatrix.append(temp)
    temp=copy.deepcopy(temp)
for i in range(len(test_label)):
    confusionmatrix[test_label[i]][label_predict[i]]=confusionmatrix[test_label[i]][label_predict[i]]+1
seaborn.heatmap(confusionmatrix,annot=True,fmt="d")   


# In[ ]:


# ROC Curve 
# prob_dist=np.transpose(prob_dist)
for i in range(10):
    print(i)
    tpr,fpr=roc_design(prob_dist[i],test_label,i)
    plt.plot(fpr, tpr ,label="Class"+str(i+1))
# roccurve(prob_dist,test_label)
plt.xlabel("False +ve Rate")
plt.ylabel("True +ve Rate")
plt.legend()
plt.title("ROC Curve for Class 1 to 11")
plt.show()


# # Enseamble Learning

# # Adaboosting

# In[ ]:


k_max=40
d=3000
class_label=list(set(train_label))
weights=[1/float(len(train_data)) for i in train_data]


# In[ ]:


Ck,alpha_k,train_result,test_result,train_acc,test_acc=adaboost(k_max,train_data,train_label,test_data,test_label,weights,d)


# In[ ]:


print("Accuracy in training : ",train_acc)
print("Accuracy in testing : ",test_acc)


# In[ ]:


#Confusion Matrix
confusionmatrix=[]
temp=[0,0,0,0,0,0,0,0,0,0]
for i in range(10):
    confusionmatrix.append(temp)
    temp=copy.deepcopy(temp)
for i in range(len(test_label)):
    confusionmatrix[test_label[i]][test_result[i]]=confusionmatrix[test_label[i]][test_result[i]]+1
seaborn.heatmap(confusionmatrix,annot=True,fmt="d")   


# # Bagging 

# In[ ]:


k_max=300
d=3000
Ck,train_result,test_predict,train_acc,test_acc=bagging(k_max,train_data,train_label,test_data,test_label,d)


# In[ ]:


#Confusion Matrix
confusionmatrix=[]
temp=[0,0,0,0,0,0,0,0,0,0]
for i in range(10):
    confusionmatrix.append(temp)
    temp=copy.deepcopy(temp)
for i in range(len(test_label)):
    confusionmatrix[test_label[i]][test_predict[i]]=confusionmatrix[test_label[i]][test_predict[i]]+1
seaborn.heatmap(confusionmatrix,annot=True,fmt="d")   


# In[ ]:


print("Test Accuracy in bagging : ",test_acc)


# # Using HOG values as features

# In[ ]:


train_data1=[]
for i in range(len(train_data)):
    train_data1.append(histogram(train_data[i]))
test_data1=[]
for i in range(len(test_data)):
    test_data1.append(histogram(test_data[i]))


# In[ ]:


train_data=copy.deepcopy(train_data1)
test_data=copy.deepcopy(test_data1)


# # Naive Bayes Algorithm

# In[ ]:


model,score_set=cross_validation_gaussian(train_data,train_label,train_size)


# In[ ]:


score_mean=np.mean(np.array(score_set))
standard_dev=np.std(np.array(score_set))
print("Standard deviation : ",standard_dev)
print("Mean of accuracy :",score_mean)
print("Accuracy for each validation : ",score_set)
best=model[np.argmax(score_set)]
best_score=best.score(test_data,test_label)
prob_dist=best.predict_proba(test_data).tolist()
label_pred=best.predict(test_data).tolist()
print("Test Accuracy by best model in cross validation : ",best_score)


# # SVM for multi-class

# In[ ]:


model,score_set=cross_validation_svm(train_data,train_label,train_size)


# In[ ]:


score_mean=np.mean(np.array(score_set))
standard_dev=np.std(np.array(score_set))
print("Standard deviation : ",standard_dev)
print("Mean of accuracy :",score_mean)
print("Accuracy for each validation : ",score_set)
best=model[np.argmax(score_set)]
best_score=best.score(test_data,test_label)
prob_dist=best.predict_proba(test_data).tolist()
label_pred=best.predict(test_data).tolist()
print("Test Accuracy by best model in cross validation : ",best_score)


# # Logistic Regression 

# In[ ]:


model,score_set=cross_validation_logistic(train_data,train_label,train_size)


# In[ ]:


score_mean=np.mean(np.array(score_set))
standard_dev=np.std(np.array(score_set))
print("Standard deviation : ",standard_dev)
print("Mean of accuracy :",score_mean)
print("Accuracy for each validation : ",score_set)
best=model[np.argmax(score_set)]
best_score=best.score(test_data,test_label)
prob_dist=best.predict_proba(test_data).tolist()
label_pred=best.predict(test_data).tolist()
print("Test Accuracy by best model in cross validation : ",best_score)

