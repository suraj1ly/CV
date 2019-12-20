
# coding: utf-8

# In[1]:


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


# In[2]:


from sklearn.svm import SVC


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


def mean_cal(data):
    main_mean=[]
    for i in range(len(data[0])):
        temp=[]
        for j in range(len(data)):
            temp.append(data[j][i])
        mean1=np.mean(np.array(temp))
        main_mean.append(mean1)
    return main_mean
        


# In[5]:


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
        


# In[6]:


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


# In[7]:


#geeks for geeks 
def sort_list(list1, list2): 
  
    zipped_pairs = zip(list2, list1) 
  
    z = [x for _, x in sorted(zipped_pairs)] 
      
    return z 


# In[8]:


def labelling(predict,true):
    h=[]
    for i in range(len(predict)):
        if predict[i]==true[i]:
            h.append(1)
        else:
            h.append(0)
    return h


# In[9]:


def accuracy(predict,true):
    count=0
    for i in range(len(predict)):
        if predict[i]==true[i]:
            count+=1
    return count/float(len(predict))


# In[10]:


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


# In[11]:


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
            


# In[12]:


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
            


# In[13]:


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


# In[14]:


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
    
    


# In[15]:


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
    
    


# In[16]:


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
    
    


# In[17]:


os.chdir('train')


# In[18]:


list_dir=os.listdir()


# In[19]:


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
        img = cv2.resize(img, (32, 32)) 
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


# In[20]:


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


# In[21]:


fold=5
train_size=len(train_data)
valid_size=int(train_size/float(fold))


# # Using Feature as Intensity Value

# # Implementing PCA

# In[24]:


transf_mat=pca(train_data,80)
train_data1=np.dot(train_data,transf_mat).real
test_data1=np.dot(test_data,transf_mat)


# In[25]:


len(train_label)


# In[26]:


len(test_label)


# In[ ]:


clf = LogisticRegressionCV(random_state=0, solver='lbfgs',multi_class='multinomial').fit(copy.deepcopy(np.array(train_data1)), copy.deepcopy(np.array(train_label)))
label_predict=clf.predict(np.array(test_data1)).tolist()
score=clf.score(np.array(test_data1),np.array(test_label))
print("Accuracy : ",score)


# # Using Logistic Regression Algorithm

# In[24]:


model,score_set=cross_validation_logistic(train_data,train_label,train_size)


# In[25]:


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


# In[27]:


#Confusion Matrix
confusionmatrix=[]
temp=[0,0,0,0,0,0,0,0,0,0]
for i in range(10):
    confusionmatrix.append(temp)
    temp=copy.deepcopy(temp)
for i in range(len(test_label)):
    confusionmatrix[test_label[i]][label_pred[i]]=confusionmatrix[test_label[i]][label_pred[i]]+1
seaborn.heatmap(confusionmatrix,annot=True,fmt="d")   


# In[28]:


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

# In[27]:


model,score_set=cross_validation_gaussian(train_data,train_label,train_size)


# In[28]:


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


# In[61]:


# clf = GaussianNB().fit(copy.deepcopy(np.array(train_data)), copy.deepcopy(np.array(train_label)))
# label_predict=clf.predict(np.array(test_data)).tolist()
# score=clf.score(np.array(test_data),np.array(test_label))
# print("Accuracy : ",score)


# In[11]:


# prob_dist=clf.predict_proba(np.array(test_data)).tolist()
# prob_dist=np.transpose(prob_dist)


# In[12]:


# #Confusion Matrix
# confusionmatrix=[]
# temp=[0,0,0,0,0,0,0,0,0,0]
# for i in range(10):
#     confusionmatrix.append(temp)
#     temp=copy.deepcopy(temp)
# for i in range(len(test_label)):
#     confusionmatrix[test_label[i]][label_predict[i]]=confusionmatrix[test_label[i]][label_predict[i]]+1
# seaborn.heatmap(confusionmatrix,annot=True,fmt="d")   


# In[13]:


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

# In[63]:


model,score_set=cross_validation_svm(train_data,train_label,train_size)


# In[64]:


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

# In[29]:


def histogram(train_data):
    hist=[0]*256
    for i in range(len(train_data)):
        hist[train_data[i]]+=1
    return hist
        


# In[30]:


train_data1=[]
for i in range(len(train_data)):
    train_data1.append(histogram(train_data[i]))
test_data1=[]
for i in range(len(test_data)):
    test_data1.append(histogram(test_data[i]))



# In[ ]:


# train_data1[0]


# In[31]:


train_data=copy.deepcopy(train_data1)
test_data=copy.deepcopy(test_data1)


# # Naive Bayes Algorithm

# In[32]:


clf = GaussianNB().fit(copy.deepcopy(np.array(train_data)), copy.deepcopy(np.array(train_label)))
label_predict=clf.predict(np.array(test_data)).tolist()
score=clf.score(np.array(test_data),np.array(test_label))
print("Accuracy : ",score)


# In[33]:


prob_dist=clf.predict_proba(np.array(test_data)).tolist()
prob_dist=np.transpose(prob_dist)


# In[34]:


#Confusion Matrix
confusionmatrix=[]
temp=[0,0,0,0,0,0,0,0,0,0]
for i in range(10):
    confusionmatrix.append(temp)
    temp=copy.deepcopy(temp)
for i in range(len(test_label)):
    confusionmatrix[test_label[i]][label_predict[i]]=confusionmatrix[test_label[i]][label_predict[i]]+1
seaborn.heatmap(confusionmatrix,annot=True,fmt="d")   


# In[35]:


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

# In[24]:


k_max=40
d=3000
class_label=list(set(train_label))
weights=[1/float(len(train_data)) for i in train_data]


# In[25]:


Ck,alpha_k,train_result,test_result,train_acc,test_acc=adaboost(k_max,train_data,train_label,test_data,test_label,weights,d)


# In[26]:


print("Accuracy in training : ",train_acc)
print("Accuracy in testing : ",test_acc)


# In[27]:


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

# In[28]:


k_max=300
d=3000
Ck,train_result,test_predict,train_acc,test_acc=bagging(k_max,train_data,train_label,test_data,test_label,d)


# In[29]:


#Confusion Matrix
confusionmatrix=[]
temp=[0,0,0,0,0,0,0,0,0,0]
for i in range(10):
    confusionmatrix.append(temp)
    temp=copy.deepcopy(temp)
for i in range(len(test_label)):
    confusionmatrix[test_label[i]][test_predict[i]]=confusionmatrix[test_label[i]][test_predict[i]]+1
seaborn.heatmap(confusionmatrix,annot=True,fmt="d")   


# In[30]:


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


# In[29]:


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

