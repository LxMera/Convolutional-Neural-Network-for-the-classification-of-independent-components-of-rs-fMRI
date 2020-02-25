# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:42:15 2020

@author: leonel Mera
"""

import os
import cv2
import glob
import numpy as np
import scipy.io as sio
from nilearn import image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Flatten, Dense, Activation

brain_pca = PCA(n_components=3)
scaler = StandardScaler()
kernel=np.ones((4,4))
umb=2
ICA='/filtered_func_data.ica'
MAP='/melodic_IC.nii.gz'
SAG=57
COR=73
AXI=57
IMG_CHANNELS = 3
shap={'Axial': (SAG, COR, IMG_CHANNELS), 'Coronal': (SAG, AXI, IMG_CHANNELS), 'Sagittal': (AXI, COR, IMG_CHANNELS)}
maxR=0.4
nb_classes=2

def CutImages(im1, im2, im3, minx):
  Paxi=minx[0]
  Pcor=minx[1]
  Psag=minx[2]
  
  i1=im1[Paxi[1]:Paxi[1]+Paxi[3],Paxi[0]:Paxi[0]+Paxi[2],:]
  i2=im2[Pcor[1]:Pcor[1]+Pcor[3],Pcor[0]:Pcor[0]+Pcor[2],:]
  i3=im3[Psag[1]:Psag[1]+Psag[3],Psag[0]:Psag[0]+Psag[2],:]

  return i1, i2, i3

def ContourIma(con, ima):
  pro=ima.copy()
  pro=cv2.drawContours(pro, [con], 0, (0,0,255), 1)
  return pro

def ContourImages(im1, im2, im3, cn1, cn2, cn3):
  i1=ContourIma(cn1, im1)
  i2=ContourIma(cn2, im2)
  i3=ContourIma(cn3, im3)
  return i1, i2, i3

def axisImages(im,cpx):
  sag, cr, ax, cp=np.shape(im)
  sx=int(sag/2)
  cx=int(cr/2)
  xa=int(ax/2)

  mx=15
  mi=-2

  im[im>mx]=mx
  im[im<mi]=mi

  im3=np.array((im[sx,:,:,cpx]-mi)*255/(mx-mi), np.dtype('uint8'))
  im2=np.array((im[:,cx,:,cpx]-mi)*255/(mx-mi), np.dtype('uint8'))
  im1=np.array((im[:,:,xa,cpx]-mi)*255/(mx-mi), np.dtype('uint8'))

  im1=cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)
  im2=cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)
  im3=cv2.cvtColor(im3, cv2.COLOR_GRAY2RGB)

  return im1, im2, im3

def filterBlur(im1, im2, im3):
  i1=cv2.medianBlur(im1, 7)
  i2=cv2.medianBlur(im2, 7)
  i3=cv2.medianBlur(im3, 7)
  return i1, i2, i3

#Saggital, coronal, axial
roundV=np.vectorize(round)
vecInt=np.vectorize(int)
def positions(rx):
  vS=vecInt(roundV(np.arange(rx[1]/48,rx[1],rx[1]/24.)))
  vC=vecInt(roundV(np.arange(rx[3]/48,rx[3],rx[3]/24.)))
  vA=vecInt(roundV(np.arange(rx[5]/48,rx[5],rx[5]/24.)))
  return vS, vC, vA

def CompressVolumen(im,cmp,roi3):
  sag, cr, ax, cp=np.shape(im)

  vecs, vecc, veca=positions(roi3)
  veca=veca+roi3[4]
  vecc=vecc+roi3[2]
  vecs=vecs+roi3[0]
  #Axial
  ima3D=np.zeros((sag,cr,3), np.dtype('uint8'))
  for z in range(8):
    val=2**z
    bina0=np.array((im[:,:,veca[z],cmp]>umb)*val, np.dtype('uint8'))
    bina0 = cv2.morphologyEx(bina0, cv2.MORPH_CLOSE, kernel)
    ima3D[:,:,0]=ima3D[:,:,0]+bina0

    bina1=np.array((im[:,:,veca[8+z],cmp]>umb)*val, np.dtype('uint8'))
    bina1 = cv2.morphologyEx(bina1, cv2.MORPH_CLOSE, kernel)
    ima3D[:,:,1]=ima3D[:,:,1]+bina1

    bina2=np.array((im[:,:,veca[16+z],cmp]>umb)*val, np.dtype('uint8'))
    bina2 = cv2.morphologyEx(bina2, cv2.MORPH_CLOSE, kernel)
    ima3D[:,:,2]=ima3D[:,:,2]+bina2

  #Coronal
  ima3C=np.zeros((sag,ax,3), np.dtype('uint8'))
  for z in range(8):
    val=2**z
    bina0=np.array((im[:,vecc[z],:,cmp]>umb)*val, np.dtype('uint8'))
    bina0 = cv2.morphologyEx(bina0, cv2.MORPH_CLOSE, kernel)
    ima3C[:,:,0]=ima3C[:,:,0]+bina0

    bina1=np.array((im[:,vecc[8+z],:,cmp]>umb)*val, np.dtype('uint8'))
    bina1 = cv2.morphologyEx(bina1, cv2.MORPH_CLOSE, kernel)
    ima3C[:,:,1]=ima3C[:,:,1]+bina1

    bina2=np.array((im[:,vecc[16+z],:,cmp]>umb)*val, np.dtype('uint8'))
    bina2 = cv2.morphologyEx(bina2, cv2.MORPH_CLOSE, kernel)
    ima3C[:,:,2]=ima3C[:,:,2]+bina2
  
  #sagittal
  ima3S=np.zeros((cr, ax,3), np.dtype('uint8'))
  for z in range(8):
    val=2**z
    bina0=np.array((im[vecs[z],:,:,cmp]>umb)*val, np.dtype('uint8'))
    bina0 = cv2.morphologyEx(bina0, cv2.MORPH_CLOSE, kernel)
    ima3S[:,:,0]=ima3S[:,:,0]+bina0

    bina1=np.array((im[vecs[8+z],:,:,cmp]>umb)*val, np.dtype('uint8'))
    bina1 = cv2.morphologyEx(bina1, cv2.MORPH_CLOSE, kernel)
    ima3S[:,:,1]=ima3S[:,:,1]+bina1

    bina2=np.array((im[vecs[16+z],:,:,cmp]>umb)*val, np.dtype('uint8'))
    bina2 = cv2.morphologyEx(bina2, cv2.MORPH_CLOSE, kernel)
    ima3S[:,:,2]=ima3S[:,:,2]+bina2
   
  return ima3D, ima3C, ima3S

def get_Axis(minx):
  Paxi=minx[0]
  Pcor=minx[1]
  Psag=minx[2]
  sg=Ds=cr=Dc=ax=Dx=0

  if Paxi[1]==Pcor[1] and Paxi[3]==Pcor[3]:
    sg=Paxi[1]
    Ds=Paxi[3]
  else:
    sg=np.min((Paxi[1],Pcor[1]))
    Ds=np.max((Paxi[3],Pcor[3]))

  if Paxi[0]==Psag[1] and Paxi[2]==Psag[3]:
    cr=Paxi[0]
    Dc=Paxi[2]
  else:
    cr=np.min((Paxi[0],Psag[1]))
    Dc=np.max((Paxi[2],Psag[3]))

  if Psag[0]==Pcor[0] and Psag[2]==Pcor[2]:
    ax=Pcor[0]
    Dx=Pcor[2]
  else:
    ax=np.min((Psag[0],Pcor[0]))
    Dx=np.max((Psag[2],Pcor[2]))
  return sg, Ds, cr, Dc, ax, Dx

def rectangles(con1, con2, con3):
  Ax= cv2.boundingRect(con1)
  Cr= cv2.boundingRect(con2)
  Sg= cv2.boundingRect(con3)
  return Ax, Cr, Sg

def ContourVector(pro):
  ima1=cv2.cvtColor(pro, cv2.COLOR_RGB2GRAY)
  color=ima1[0,0]
  ima1=np.array((ima1!=color), np.dtype('uint8'))
  contours,_ = cv2.findContours(ima1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
  ind=0
  if np.shape(contours)[0]!=1:
    mx=[]
    for xi in range(np.shape(contours)[0]):
      area=cv2.contourArea(contours[xi])
      mx.append(area)
    ind=np.argmax(mx)
  cont=contours[ind]
  return cont

def getContours(im1, im2, im3):
  con1=ContourVector(im1)
  con2=ContourVector(im2)
  con3=ContourVector(im3)
  
  return con1, con2, con3

def verification(sub):
  exi=True
  if not os.path.isdir(sub+ICA):
    print('The subject dont have ICA folder')
    exi=False
  else:
    if not os.path.isfile(sub+ICA+MAP):
      print('The subject dont have independent components')
      exi=False
  return exi

def segmentationPCA(imagex,k):
  sag, cr, ax, cp=np.shape(imagex)
  if k>cp or k<0:
    print('The component number',str(k),' doesnt exist')
  #Axial
  brainF = np.zeros((1, sag*cr))
  ima3D=np.zeros((sag,cr,3))  
  for sl in range(ax):   
    slic=np.transpose(imagex[:,:,sl,k].flatten())
    brainF = np.concatenate((brainF,[slic]), axis=0)
  brainF=brainF[1:,:]  
  brain_pca.fit(brainF) 

  ima = np.transpose(brain_pca.components_)
  ima = np.transpose(scaler.fit_transform(ima))
  ima=(ima+1)*255/2
  ima[ima<0]=0
  ima[ima>255]=255 
  for chanel in range(3):
    ima3D[:,:,chanel]=ima[chanel].reshape(sag,cr)
  ima3D=np.array(ima3D, np.dtype('uint8'))

  #Coronal
  brainC = np.zeros((1, sag*ax))
  ima3C=np.zeros((sag,ax,3))  
  for sl in range(cr):    
    slic=np.transpose(imagex[:,sl,:,k].flatten())
    brainC = np.concatenate((brainC,[slic]), axis=0)
  brainC=brainC[1:,:]  
  brain_pca.fit(brainC) 

  imaC = np.transpose(brain_pca.components_)
  imaC = np.transpose(scaler.fit_transform(imaC))
  imaC=(imaC+1)*255/2
  imaC[imaC<0]=0
  imaC[imaC>255]=255 
  for chanel in range(3):
    ima3C[:,:,chanel]=imaC[chanel].reshape(sag,ax)
  ima3C=np.array(ima3C, np.dtype('uint8'))

  #Saggital
  brainS = np.zeros((1, cr*ax))
  ima3S=np.zeros((cr,ax,3))  
  for sl in range(sag):    
    slic=np.transpose(imagex[sl,:,:,k].flatten())
    brainS = np.concatenate((brainS,[slic]), axis=0)
  brainS=brainS[1:,:]  
  brain_pca.fit(brainS) 

  imaS = np.transpose(brain_pca.components_)
  imaS = np.transpose(scaler.fit_transform(imaS))
  imaS=(imaS+1)*255/2
  imaS[imaS<0]=0
  imaS[imaS>255]=255 
  for chanel in range(3):
    ima3S[:,:,chanel]=imaS[chanel].reshape(cr,ax)
  ima3S=np.array(ima3S, np.dtype('uint8'))
  
  return ima3D, ima3C, ima3S

def image_RPBC(prin, ncmp, method='RCBP', contour=True):
  imaA, imaC, imaS=segmentationPCA(prin,ncmp)
  conA, conC, conS=getContours(imaA, imaC, imaS)
  mini=rectangles(conA, conC, conS)
  roi=get_Axis(mini)
  CmaA, CmaC, CmaS=CompressVolumen(prin,ncmp,roi)

  if method=='PCA':
    if contour:
      res=ContourImages(imaA, imaC, imaS, conA, conC, conS) 
      res=CutImages(res[0], res[1], res[2], mini)
    else: 
      res=CutImages(imaA, imaC, imaS, mini)

  if method=='RCBP':
    if contour:
      res=ContourImages(CmaA, CmaC, CmaS, conA, conC, conS)
      res=CutImages(res[0], res[1], res[2], mini)
    else: 
      res=CutImages(CmaA, CmaC, CmaS, mini)

  if method=='BLUR':
    fmaA, fmaC, fmaS=filterBlur(CmaA, CmaC, CmaS)
    if contour:
      res=ContourImages(fmaA, fmaC, fmaS, conA, conC, conS)
      res=CutImages(res[0], res[1], res[2], mini)
    else:       
      res=CutImages(fmaA, fmaC, fmaS, mini)
  return res

def showReduction(imx):
  plt.figure(figsize=(15,5))
  plt.subplot(1,3,1)
  plt.imshow(cv2.cvtColor(imx[0], cv2.COLOR_RGB2BGR))
  plt.axis('off')
  plt.subplot(1,3,2)
  plt.imshow(cv2.cvtColor(imx[1], cv2.COLOR_RGB2BGR))
  plt.axis('off')
  plt.subplot(1,3,3)
  plt.imshow(cv2.cvtColor(imx[2], cv2.COLOR_RGB2BGR))
  plt.axis('off')

def get_Layer_4(Slic, WEIGHTS, showM=False):
    print(Slic+' model shape ', shap[Slic])
    tf.reset_default_graph()
    tf.keras.backend.clear_session()    
    
    inputs = Input(shap[Slic])
    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = MaxPooling2D((2, 2)) (c1)    
    c1 = Dropout(maxR) (c1) 

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    c2 = MaxPooling2D((2, 2)) (c2)    
    c2 = Dropout(maxR) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    c3 = MaxPooling2D((2, 2)) (c3)    
    c3 = Dropout(maxR) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    c4 = MaxPooling2D((2, 2)) (c4)    
    c4 = Dropout(maxR) (c4)
    
    #Flatten layers
    c6 = Flatten()(c4)

    c6 = Dense(512, activation='relu')(c6)
    c6 = Dropout(maxR)(c6)
        
    c6 = Dense(128, activation='relu')(c6)
    c6 = Dropout(maxR)(c6)       
       
    outputs = Dense(nb_classes, activation='softmax')(c6)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if showM:
      model.summary()      
    model.load_weights(WEIGHTS)
        
    return model

def Reduction_IC(SUB, showI=False):

  if verification(SUB):
    maps=image.image.load_img(SUB+ICA+MAP)
    mapsA=maps.get_data()
    comps=np.shape(mapsA)[3]
    print(str(comps)+' components')
    os.system('mkdir '+SUB+'/RCBP')

    s_train = np.zeros((comps, AXI, COR, 3), dtype=np.uint8)
    c_train = np.zeros((comps, SAG, AXI, 3), dtype=np.uint8)
    a_train = np.zeros((comps, SAG, COR, 3), dtype=np.uint8)

    for k in range(comps):
      print('\rReduction of IC to images: Process', round(k*100/(comps-1),2), '%...' , end ="")
      imageRPBC=image_RPBC(mapsA, k, method='RCBP', contour=True)
      sag=cv2.resize(imageRPBC[2], (AXI, COR), interpolation = cv2.INTER_AREA)
      cor=cv2.resize(imageRPBC[1], (AXI, SAG), interpolation = cv2.INTER_AREA)
      axi=cv2.resize(imageRPBC[0], (COR, SAG), interpolation = cv2.INTER_AREA)
      
      cv2.imwrite(SUB+'/RCBP/Comp-'+str(k)+'-axi.png',axi)
      cv2.imwrite(SUB+'/RCBP/Comp-'+str(k)+'-cor.png',cor)
      cv2.imwrite(SUB+'/RCBP/Comp-'+str(k)+'-sag.png',sag)

      s_train[k]=np.rot90(sag)
      c_train[k]=cor
      a_train[k]=axi

    print('   ...finished process')
    if showI:
      showReduction([axi, cor, sag])
    return s_train, c_train, a_train
  else:
    print('Fatal error: didnt find ICA in '+SUB+ICA+MAP)

def classificationIC_by_CNN(SUB, mod):
  imaIC=Reduction_IC(SUB, showI=False)
  typeOfIC = CNN.predict(imaIC[0])
  noise=typeOfIC.argmax(axis=1)
  noise=np.where(noise==0)[0]+1

  file = open(SUB+"/auto_labels_noise.txt", "w")
  file.write("./filtered_func_data.ica" + os.linesep)
  file.write("[")
  for i in noise[:-1]:
    file.write(str(i)+", ")
  file.write(str(noise[-1])+"]")
  file.close()
  return noise, typeOfIC

def downloadH5(SUB):
  os.system('wget https://github.com/LxMera/Convolutional-Neural-Network-for-the-classification-of-independent-components-of-rs-fMRI/raw/master/LAYER4-C_COM-Saggital-0.h5 -P '+SUB)
  Path=SUB+'/LAYER4-C_COM-Saggital-0.h5'
  return Path

if __name__=="__main__":
  #Ejecución de un código
  
  Subject='HCP_hp2000/1.ica'
  
  #Descargar los pesos del modelo en la dirección indicada
  DIR=downloadH5(Subject)

  #Obtener el modelo con los pesos ya entrendos
  CNN=get_Layer_4('Sagittal', DIR)

  #Generar la clasificación
  Clas,_=classificationIC_by_CNN(Subject, CNN)
  
  print('Componentes de ruido', Clas)
