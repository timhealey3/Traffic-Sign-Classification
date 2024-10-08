import torch
from torch import nn
import pickle
import random
import numpy as np 
import cv2

def read_data():
    np.random.seed(0)
    # read pickeled data files
    with open('german-traffic-signs/train.p', 'rb') as file:
        train_data = pickle.load(file)
    with open('german-traffic-signs/valid.p', 'rb') as file:
        val_data = pickle.load(file)
    with open('german-traffic-signs/test.p', 'rb') as file:
        test_data = pickle.load(file)

    X_train, y_train = train_data['features'], train_data['labels']
    X_val, y_val = val_data['features'], val_data['labels']
    X_test, y_test = test_data['features'], test_data['labels']

# check images and labels are same
    assert(X_train.shape[0] == y_train.shape[0]), "Num train images != num images"
    assert(X_val.shape[0] == y_val.shape[0]), "Num val images != num images"
    assert(X_test.shape[0] == y_test.shape[0]), "Num train images != num images"
    assert(X_train.shape[1:] == (32, 32, 3)), "Dimension of train images are not 32x32x3"
    assert(X_val.shape[1:] == (32, 32, 3)), "Dimension of val images are not 32x32x3"
    assert(X_test.shape[1:] == (32, 32, 3)), "Dimension of test images are not 32x32x3"
    return X_train, y_train, X_val, y_val, X_test, y_test

def setDevice():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

class modifiedModel(nn.Module):
    def __init__(self):
        pass

# returns imgs in tensors
def shapeImgData(X_train, X_val, X_test, y_train, y_val, y_test, num_classes):
     # process images 
    X_train = np.array(list(map(preprocessing, X_train)))
    X_val = np.array(list(map(preprocessing, X_val)))
    X_test = np.array(list(map(preprocessing, X_test)))
    
    X_train = X_train.reshape(34799, 32, 32, 1)
    X_test = X_test.reshape(12630, 32, 32, 1)
    X_val = X_val.reshape(4410, 32, 32, 1)
    # convert to tensor
    X_train = torch.tensor(X_train) 
    X_test = torch.tensor(X_test)
    X_val = torch.tensor(X_val)
    # one hot encode labels
    y_train = torch.nn.functional.one_hot(torch.tensor(y_train, dtype=torch.long), num_classes=num_classes).float()
    y_val = torch.nn.functional.one_hot(torch.tensor(y_val, dtype=torch.long), num_classes=num_classes).float()
    y_test = torch.nn.functional.one_hot(torch.tensor(y_test, dtype=torch.long), num_classes=num_classes).float()

    return X_train, X_test, X_val, y_train, y_val, y_test


X_train, y_train, X_val, y_val, X_test, y_test = read_data()
if __name__ == "__main__":
    num_classes = 43
    setDevice()
    X_train, X_val, X_test, y_train, y_val, y_test = shapeImgData(X_train, X_val, X_test, y_train, y_val, y_test, num_classes)
    #
    # one hot encode for 43 categories
   #modifiedModel().to(device)

