# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 19:54:56 2020

@author: PC Arwa
"""

from keras.preprocessing.image import ImageDataGenerator,  img_to_array, load_img
import glob
import cv2
from PIL import Image
import numpy as np
import random
from sklearn import svm
import xgboost as xg
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,accuracy_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#Question1
dataDog=glob.glob('C:/Users/PC Arwa/Desktop/Arwa_Jouini/Animals/dog*')
dataCat=glob.glob('C:/Users/PC Arwa/Desktop/Arwa_Jouini/Animals/cat*')

#Question2
dataAug = ImageDataGenerator(rotation_range = 40,shear_range = 0.2, 
        zoom_range = 0.2,horizontal_flip = True, 
        width_shift_range=0.2,height_shift_range=0.2,brightness_range = (0.5, 1.5))


def imgFunc (data,c):
    for img in data:
         pic = load_img(img)
         array_img = img_to_array(pic) 
         array_img = array_img.reshape((1, ) + array_img.shape)
         j = 0
         for i in dataAug.flow(array_img, batch_size=1,save_to_dir='Animals', save_prefix=c, save_format='jpg'):
            j += 1
            if j > 60:
                break

imgFunc(dataDog,'dog')
imgFunc(dataCat,'cat')

#Question3

new_dataDog=glob.glob('C:/Users/PC Arwa/Desktop/Arwa_Jouini/Animals/dog*')
new_dataCat=glob.glob('C:/Users/PC Arwa/Desktop/Arwa_Jouini/Animals/cat*')

def colorFunc(data):
    for pic in data:
        img = cv2.imread(pic)
        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(pic, image_gray)
        
colorFunc(new_dataDog)
colorFunc(new_dataCat)

#Question4
def resizeFunc(data):
    for pic in data:
        img = Image.open(pic)
        img = img.resize((400,400))
        img.save(pic)


resizeFunc(new_dataDog)
resizeFunc(new_dataCat)



#Question5
mydata=[]
cat=0
dog=0
for pic in new_dataCat:
    im1=Image.open(pic)
    img = np.array(im1)
    img = img.reshape(img.shape[0]*img.shape[1])
    mydata.append(img)
    cat+=1

for pic in new_dataDog:
    im1=Image.open(pic)
    img = np.array(im1)
    img = img.reshape(img.shape[0]*img.shape[1])
    mydata.append(img)
    dog+=1

l=[]
for i in range(cat):
    l.append(1)
for i in range(dog):
    l.append(0)

x = list(zip(mydata, l))
random.shuffle(x)
mydata,l = zip(*x)
mydata=list(mydata)
l=list(l)


#Question 6 
#PCA
model=PCA(n_components=1000)
X1=model.fit_transform(mydata)
#LDA
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#model = LDA()
#X2 = model.fit_transform(mydata, l)

#Question 7 Random Forest
X1_train, X1_test, y_train, y_test= train_test_split(X1,l, test_size=0.3, random_state=0)
rf = RandomForestRegressor(n_estimators = 1000, random_state = 40)
rf.fit(X1_train, y_train)
y_pred=rf.predict(X1_test)
print("Random Forest :\nAccuracy:",metrics.accuracy_score(y_test, y_pred.round())*100)
classification_report(y_test, y_pred)


#SVM ploy
model = svm.SVC(kernel='poly',degree=5)
#model=svm.SVC(kernel='sigmoid',gamma=0.1,Coef0=2)
#model=svm.SVC(kernel='rbf',C=1,gamma=0.1)
#model=svm.SVC(kernel='linear',C=1)
model.fit(X1_train, y_train)
predictions_poly = model.predict(X1_test)
print("SVM:\n Accuracy  " + str(100 * accuracy_score(y_test, predictions_poly))+'%')
classification_report(y_test, y_pred)




#XGBOOST
model = xg.DMatrix(data=X1,label=l)
xgm = xg.XGBRegressor(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 20, alpha = 150, n_estimators = 1000)
xgm.fit(X1_train,y_train)
preds = xgm.predict(X1_test)
print("XGBOOST \n Accuracy:",metrics.accuracy_score(y_test, preds.round())*100)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: "+ str(rmse))
classification_report(y_test, y_pred)




