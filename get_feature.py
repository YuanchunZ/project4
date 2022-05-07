import cv2
import numpy as np

import torch
import torch.nn
import torchvision.models as models
import torch.cuda
import torchvision.transforms as transforms
 
import pandas as pd
from PIL import Image

TARGET_IMG_SIZE = 224
img_to_tensor = transforms.ToTensor()


import random
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

def make_model():
    model=models.resnet34(pretrained=True)	# Basically, it's located on the 28th floor
    model=model.eval()	# Increase the speed of operation
    return model

def resnet_get_feature(net,input_data):
    x = net.conv1(input_data)

    x = net.bn1(x)

    x = net.relu(x)
 
    x = net.maxpool(x)
 
    x = net.layer1(x)
  
    x = net.layer2(x)
 
    x = net.layer3(x)
  
    x = net.layer4(x)

    x = net.avgpool(x)

    x = torch.flatten(x, 1)

    return x


def extract_feature(model,imgpath):
    model.eval()
    
    img=Image.open(imgpath)		# Read the pictures
    img=img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    tensor=img_to_tensor(img)	# Translate the picture into tensor
    
    result=resnet_get_feature(model, tensor[None, ...])
    result_npy=result.data.cpu().numpy()[0]
    # print(result_npy.shape)
    return result_npy	

# Feature extractor
# Feature extraction
def extract_features(image_path, vector_size=32):
    image = cv2.imread(image_path)
    image = cv2.resize(image,(28, 28))
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    try:
        alg = cv2.SIFT_create()
        # Finding image keypoints
        kps, dsc = alg.detectAndCompute(image,None)
        # Getting first 32 of them.

        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)

        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector

        # Flatten all of them in one big vector - our feature vector

        if(dsc is None):
            return None
        dsc = dsc.flatten()
        print(dsc.shape)
        # Making descriptor of same size

        # Descriptor vector size is 64
               needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros
            # at the end of our feature vector

            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print ('Error: ', e)
        return None


    return dsc
    
def batch_extractor(data_directory, kind="train"):
    model = make_model()
    features = []
    labels = []
    df = pd.read_csv("./data/structure.csv")
    classes = set(df['Template Name'])
    cla2num = {}
    for i,k in enumerate(classes):
        cla2num[k] = i
    label = df['Template Name']
    paths = df['Image']

    for f,l in zip(paths, label):

        feature = extract_feature(model, './data/'+f)
        features.append(feature)
        labels.append(int(cla2num[l]))
            
    features = np.array(features)
    labels = np.array(labels)
    if(not os.path.exists("./feature")):
        os.makedirs("./feature")
    np.save("./feature/"+kind+"_feature.npy", features)
    np.save("./feature/"+kind+"_label.npy", labels)

def dim_reduction(dim, kind='train'):
    pca = PCA(n_components=dim)
    ori_feature = np.load("./feature/"+kind+"_feature.npy", allow_pickle=True)
    print(ori_feature.shape)
    reduction_feature = pca.fit_transform(ori_feature)
    np.save("./feature/"+kind+"_feature_reduction"+str(dim)+".npy", reduction_feature)


def classification():
    x_train = np.load("./feature/train_feature_reduction2.npy")
    # x_train = np.load("./feature/train_feature.npy")
    y_train = np.load("./feature/train_label.npy")

    # x_test = np.load("./feature/test_feature_reduction2.npy")
    # y_test = np.load("./feature/test_label.npy")

    xgbc = XGBClassifier()        #XGBoos does not take any arguments, so use the original arguments
    xgbc.fit(x_train, y_train)     #Model fitting, model training
    # prediction_result = xgbc.predict(x_)

    print('The Accuracy is', xgbc.score(x_train, y_train))


if __name__ =="__main__":
    # batch_extractor("./data/BelgiumTSC_Training/", 'train')
    # batch_extractor("./data/BelgiumTSC_Testing/", 'test')
    # dim_reduction(2,'test')
    # dim_reduction(3,'train')
    classification()





        