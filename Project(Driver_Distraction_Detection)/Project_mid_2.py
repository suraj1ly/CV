import pywt
import cv2 as cv
import numpy as np
import copy as cp
import os
import glob
import csv
from sklearn import metrics
from sklearn.decomposition import KernelPCA
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
from skimage.feature import hog
from skimage.feature import local_binary_pattern as LBP
from sklearn import preprocessing as PP
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score as CVS

def read_train(path):
    images=[]; labels=[];
    for i in range(10):
       print('Loading: ',i)
       p=path+str(i)+'\\'
       for file in glob.glob(p+'*jpg'):
          im = cv.imread(file)
          im=cv.resize(im,(64,64))
          images.append(cp.deepcopy(im))
          labels.append(cp.deepcopy(i))
    images=np.array(images); labels=np.array(labels)
    #images=images[idx]; labels=labels[idx]; names=names[idx]
    images,labels=shuffle(images,labels)
    return np.array(images),np.array(labels)


def read_test(path):
    images=[]; names=[]
    print('Reading Test data...'); i=0
    for file in glob.glob(path+'*jpg'):
        i+=1
        if(i%5000==0):
            print('Reading: ',i)
        x=file.split('\\')
        x=x[len(x)-1]
        names.append(cp.deepcopy(x))
        im = cv.imread(file)
        im=cv.resize(im,(64,64))
        images.append(cp.deepcopy(im))
    print(i,' test images')
    return np.array(images),np.array(names)

    
def read_new(d):
    os.chdir(d)
    list_dir=os.listdir()
    train_data=[]; train_label=[]
    test_data=[];  test_label=[]
    count1=0
    for i in list_dir:   
        print(i)
        os.chdir(i)
        list_doc=os.listdir()
        count=0
        for j in list_doc:
            #print('Unable to read: ',j)
            count+=1
            img=cv.imread(j)
            #img = cv.resize(img, (32, 32)) 
            # print(img)
            #img=img.ravel()
            if count<800:
                test_data.append(img)
                test_label.append(count1)
            else:          
                train_data.append(img)
                train_label.append(count1)
                count1+=1
    os.chdir('..')
    train_data=np.array(train_data); train_label=np.array(train_label)
    test_data=np.array(test_data); test_label=np.array(test_label)
    train_data,train_label=shuffle(train_data,train_label)
    test_data,test_label= shuffle(test_data,test_label)
    return train_data, train_label, test_data, test_label

##------------------------------------Features-------------------------------
def feat_wavelet(X):
    feat=[]
    for im in X:
        im=cv.cvtColor(im,cv.COLOR_BGR2GRAY)
        coff=pywt.dwt2(im,'haar')
        LL,(LH,HL,HH) = coff
        imf=np.vstack((np.hstack((LL,LH)),np.hstack((HL,HH))))
        imf=np.reshape(im,(imf.shape[0]*imf.shape[1]))
        feat.append(imf)
    return np.array(feat)

def feat_HOG(Xtr,Xte):
    print(Xtr.shape,Xte.shape)
    feat_train=[]; feat_test=[]
    for i in range(Xtr.shape[0]):
        #im=Xtr[i]
        im=cv.cvtColor(Xtr[i],cv.COLOR_BGR2GRAY)
        fd = hog(im, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1),)#,multichannel=True)
        feat_train.append(cp.deepcopy(fd))
    for i in range(Xte.shape[0]):
        #im=Xte[i]
        im=cv.cvtColor(Xte[i],cv.COLOR_BGR2GRAY)
        fd = hog(im, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1)) #,multichannel=True)
        feat_test.append(cp.deepcopy(fd))
    
    return np.array(feat_train),np.array(feat_test)

def feat_LBP(Xtr,Xte):
    radius = 3
    n_points = 8 * radius
    feat_train=[]; feat_test=[]
    for i in range(Xtr.shape[0]):
        #im=Xtr[i]
        im=cv.cvtColor(Xtr[i],cv.COLOR_BGR2GRAY)
        fd=LBP(im,n_points,radius)
        fd=np.histogram(fd)[0]
        fd=fd/np.max(fd)
        #fd = hog(im, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1))#,multichannel=True)
        feat_train.append(cp.deepcopy(fd))
    for i in range(Xte.shape[0]):
        #im=Xte[i]
        im=cv.cvtColor(Xte[i],cv.COLOR_BGR2GRAY)
        fd=LBP(im,n_points,radius)
        fd=np.histogram(fd)[0]
        fd=fd/np.max(fd)
        #fd = hog(im, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1)) #,multichannel=True)
        feat_test.append(cp.deepcopy(fd))
    
    return np.array(feat_train),np.array(feat_test)

def feat_HOG_LBP(Xtr,Xte):
    radius = 3
    n_points = 8 * radius
    feat_train=[]; feat_test=[]
    for i in range(Xtr.shape[0]):
        im=Xtr[i]
        if(len(im.shape)!=2):
          im=cv.cvtColor(Xtr[i],cv.COLOR_BGR2GRAY)
        fd1=LBP(im,n_points,radius)
        fd1=fd1.reshape(fd1.shape[0]*fd1.shape[1])
        fd1=np.histogram(fd1)[0]
        fd1=fd1/np.max(fd1)
        fd2 = hog(im, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1))#,multichannel=True)
        fd=np.concatenate((np.array(fd1),np.array(fd2)))
        feat_train.append(cp.deepcopy(fd))
    for i in range(Xte.shape[0]):
        if(len(im.shape)!=2):
          im=cv.cvtColor(Xte[i],cv.COLOR_BGR2GRAY)
        fd1=LBP(im,n_points,radius)
        fd1=fd1.reshape(fd1.shape[0]*fd1.shape[1])
        fd1=np.histogram(fd1)[0]
        fd1=fd1/np.max(fd1)
        fd2 = hog(im, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1)) #,multichannel=True)
        fd=np.concatenate((np.array(fd1),np.array(fd2)))
        feat_test.append(cp.deepcopy(fd))
    return np.array(feat_train),np.array(feat_test)


#def cross_validate(data,label):
#    X,y=divide(data,label)


def main_1(data,lbl,feature,n_comp,classifier):
   n=int(data.shape[0]*0.7)
   Xtr=np.array(data[0:n]); Ytr=np.array(lbl[0:n])
   Xte=np.array(data[n:data.shape[0]]); Yte=np.array(lbl[n:lbl.shape[0]])
        
   if('wavelet' in feature):
     print('Calculating wavelet features')
     train_feat=feat_wavelet(Xtr)
     test_feat=feat_wavelet(Xte)
     
   if('hog' in feature):
     print('Calculating HOG features')  
     train_feat,test_feat=feat_HOG(Xtr,Xte)
     print(train_feat.shape,Ytr.shape)
   
   if('lbp' in feature):
       print('Calculating LBP features')
       train_feat,test_feat=feat_LBP(Xtr,Xte)
   if('hog+lbp' in feature):
       train_feat,test_feat=feat_HOG_LBP(Xtr,Xte)
   
   if(classifier=='SVM'):
      clf = svm.SVC(kernel='linear',probability=True)
      clf.fit(train_feat, Ytr)
      
   if(classifier=='LR'):
     clf=LR(random_state=0, solver='lbfgs',multi_class='multinomial')
     clf.fit(train_feat,Ytr)
   
   if(classifier=='GNB'):
      clf=GNB()
      clf.fit(train_feat,Ytr)
      
   prob=clf.predict_proba(test_feat)
   scores=[]
   for i in range(10):
       fpr, tpr, thresholds = metrics.roc_curve(Yte, prob[:,i], pos_label=i)
       plt.plot(tpr,fpr,label='C'+str(i))
       plt.legend()
   
   plt.xlabel('FPR')
   plt.ylabel('TPR')
   plt.title(feature[0]+'_'+classifier)
   
   #clf=cross_validate(data,label)
   
   #scores = cross_val_score(clf, iris.data, iris.target, cv=5)
   #score=CVS(clf,Xtr,Ytr,cv=5)
   p_lbl=clf.predict(test_feat)
   #prob=clf.predict_proba(test_feat)
   confmat=confusion_matrix(p_lbl,Yte)
   plt.figure()
   sns.heatmap(confmat,annot=True,fmt='d')
   acc=100*np.sum(np.diag(confmat))/np.sum(confmat)
   print('---------Configuration------- ')
   print('Feature: ',feature)
   print('Classifier: ',classifier)
   
   print('Classification Accuracy: ',acc,'%')
   #prob=clf.predict_proba(test_feat)
   #return confmat,prob
   
   return clf,test_feat
   

def write_result(name,prob,file_name):
    with open(file_name,'w',newline='') as file:
        writer = csv.writer(file) 
        for i in range(len(name)):
            row=[name[i]]  
            for j in range(10):
               x='{0:.15f}'.format(prob[i,j])
               row.append(x)
            writer.writerow(row)
    file.close()

if __name__=="__main__":
   path_train='C:\\Users\\arjun_000\\Desktop\\IIITD_notsync\\Dataset\\train\\c'
   path_test='C:\\Users\\arjun_000\\Desktop\\IIITD_notsync\\Dataset\\test\\'
   data,label=read_train(path_train)
   #Xte,names=read_test(path_test)
   feature=['lbp']       #hog, lbp ,hog+lbp, wavelet
   pca=False; n_comp=100
   classifier='SVM'     # LR, SVM, GNB
   preprocessing=False
   clf,test_feat=main_1(data,label,feature,n_comp,classifier)
   prob=clf.predict_proba(test_feat)
   #write_result(names,prob,'hog_GNB.csv')
   