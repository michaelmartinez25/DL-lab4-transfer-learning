import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim import Adam
from torchvision import datasets
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import time
from PIL import Image
import pathlib

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_data(t):
    """ 
    Loads the CIFAR10 dataset and splits it into training and testing data.
    
    parameters:
        t -- the transformation to apply to the dataset
    """
    # Load CIFAR10 dataset
    # Set download to True to download the dataset (first time only)
    cifar_train = datasets.CIFAR10(root='~/data/CIFAR', download=True, train=True, transform=t)
    cifar_test = datasets.CIFAR10(root='~/data/CIFAR', download=True, train=False, transform=t)
    
    # Split training data in half
    half_train = len(cifar_train)//2
    indices_train = list(range(half_train))
    subset_train = Subset(cifar_train, indices_train)
    # Load training data
    train_dl = DataLoader(subset_train, batch_size=32, shuffle=True)

    # Split test data in half
    half_test = len(cifar_test)
    indices_test = list(range(half_test))
    subset_test = Subset(cifar_test, indices_test)
    # Load test data
    test_dl = DataLoader(subset_test, batch_size=32, shuffle=True)
    
    return train_dl, test_dl


class TrainTestModel():
    """ This class has operations that trains and tests a model. 

        Attributes:
            train_dl       -- the training dataset's dataloader
            test_dl        -- the testing dataset's dataloader
            model          -- the neural network that is being trained and tested
            opt            -- the optimization function used to train the model
            loss_func      -- the loss function used for training the model
            time_to_train  -- the time it took to train the model
            train_accuracy -- the accuracy of the model on the training dataset
            test_accuracy  -- the accuracy of the model on the testing dataset
    """
    def __init__(self, train_dl, test_dl, model):
        """ Initialization function
        
        parameters:
            train_dl  -- the training dataset's dataloader
            test_dl   -- the testing dataset's dataloader
            model     -- the neural network that is being trained and tested
            opt       -- the optimization function used to train the model
            loss_func -- the loss function used for training the model
        """
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.model = model
        self.opt = Adam(model.parameters(), lr=1e-3)
        self.loss_func = nn.CrossEntropyLoss()
        self.time_to_train = 0
        self.train_accuracy = 0
        self.test_accuracy = 0
    
    def train_model(self):
        """ Trains model by training several batches in several epochs """
        n_epochs = 10
        before = time.time()
        for epoch in range(n_epochs):
            print(f"Running epoch {epoch + 1} of {n_epochs}")
            for batch in self.train_dl:
                x, y = batch
                self.train_batch(x.to(device), y.to(device))
        self.time_to_train = time.time() - before

        return
    
    def train_batch(self, x, y):
        """ Trains model based on batch. Takes loss after and adjusts weights in gradient descent. 

        parameters:
            x -- the set of batch values
            y -- the set of accompanying labels for the batch
        """
        self.model.train()
        # Flush memory
        self.opt.zero_grad()                   
        # Compute loss
        batch_loss = self.loss_func(self.model(x), y)  
        # Compute gradients
        batch_loss.backward()              
        # Make a GD step
        self.opt.step()                         

        return batch_loss.detach().cpu().numpy()
    
    @torch.no_grad() 
    def batch_accuracy(self, x, y,):
        """ Calculates the accuracy of a batch

        parameters:
            x -- the set of batch values
            y -- the set of accompanying labels for the batch
        
        return:
            s -- batch accuracy
        """
        self.model.eval() 
        # Check model prediction
        prediction = self.model(x)
        # Compute the predicted labels for the batch
        argmaxes = prediction.argmax(dim=1) 
        # Compute accuracy
        s = torch.sum((argmaxes == y).float())/len(y)
        return s.cpu().numpy()

    @torch.no_grad()
    def test_model(self):
        """ Tests the model on training and testing data. Stores the results of both accuracies
            in the Object
        """
        test_accuracies = []
        train_accuracies = []

        for batch in self.train_dl:
            x, y = batch
            batch_accuracy = self.batch_accuracy(x.to(device), y.to(device))
            train_accuracies.append(batch_accuracy)
        self.train_accuracy = np.mean(train_accuracies)
        

        for batch in self.test_dl:
            x, y = batch
            batch_accuracy = self.batch_accuracy(x.to(device), y.to(device))
            test_accuracies.append(batch_accuracy)
        self.test_accuracy = np.mean(test_accuracies)


def test_vgg(version):
    if version == "11":
        vgg_t = models.VGG11_Weights.IMAGENET1K_V1.transforms()
        weights = models.VGG11_Weights.IMAGENET1K_V1
        vgg_model = models.vgg11(weights=weights).to(device)
    else: 
        vgg_t = models.VGG19_Weights.IMAGENET1K_V1.transforms()
        weights = models.VGG19_Weights.IMAGENET1K_V1
        vgg_model = models.vgg19(weights=weights).to(device)

    # Don't train vgg weights
    for param in vgg_model.parameters():
        param.requires_grad = False

    # train classifier instead
    # for param in vgg.classifier.parameters():
    #     param.requires_grad = True
    
    vgg_model.classifier = nn.Sequential(
        nn.Linear(vgg_model.classifier[0].in_features, 512), 
        nn.ReLU(),
        nn.Linear(512, 64), 
        nn.ReLU(),
        nn.Linear(64, 10)).to(device)
    
    train_dl, test_dl = get_data(vgg_t)
    trainTest = TrainTestModel(train_dl=train_dl, test_dl=test_dl, model=vgg_model)
    trainTest.train_model()
    trainTest.test_model()
    
    time_taken, train_accuracy, test_accuracy = trainTest.time_to_train, \
            trainTest.train_accuracy, trainTest.test_accuracy

    return time_taken, train_accuracy, test_accuracy

def test_resnet(version):
    if version == "34":
        resnet_t = models.ResNet34_Weights.IMAGENET1K_V1.transforms()
        weights = models.ResNet34_Weights.IMAGENET1K_V1
        resnet_model = models.resnet34(weights=weights).to(device)
    else: 
        resnet_t = models.ResNet50_Weights.IMAGENET1K_V1.transforms()
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        resnet_model = models.resnet50(weights=weights).to(device)
    # all resnet models use the same transformations
    resnet_model.fc = nn.Sequential(
        nn.Linear(resnet_model.fc.in_features, 512), 
        nn.ReLU(),
        nn.Linear(512, 64), 
        nn.ReLU(),
        nn.Linear(64, 10)).to(device)

    for param in resnet_model.parameters():
        param.requires_grad = False

    # for param in resnet.fc.parameters():
    #     param.requires_grad = True

    train_dl, test_dl = get_data(resnet_t)
    trainTest = TrainTestModel(train_dl=train_dl, test_dl=test_dl, model=resnet_model)
    trainTest.train_model()
    # torch.save(trainTest.model.state_dict(), 'resnet' + version)
    trainTest.test_model()

    time_taken, train_accuracy, test_accuracy = trainTest.time_to_train, \
            trainTest.train_accuracy, trainTest.test_accuracy

    return time_taken, train_accuracy, test_accuracy

def classify_image(model, image_path, model_t, class_names):
    img = Image.open(image_path)

    model.eval()
    with torch.inference_mode():
        transformed_img = model_t(img).unsqueeze(dim=0)
        prediction = model(transformed_img.to(device))
    
    prediction_prob = torch.softmax(prediction, dim=1)
    prediction_label = torch.argmax(prediction_prob, dim=1)
    
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[prediction_label]} | Prob: {prediction_prob.max():.3f}")
    plt.axis(False)
    return prediction_label, prediction_prob
    

def main():
    vgg11_time, vgg11_train_acc, vgg11_test_acc = test_vgg("11")

    print(f"VGG11 Time: {vgg11_time}")
    print(f"VGG11 Train Accuracy: {vgg11_train_acc}")
    print(f"VGG11 Test Accuracy: {vgg11_test_acc}")

    vgg19_time, vgg19_train_acc, vgg19_test_acc = test_vgg("19")

    print(f"VGG19 Time: {vgg19_time}")
    print(f"VGG19 Train Accuracy: {vgg19_train_acc}")
    print(f"VGG19 Test Accuracy: {vgg19_test_acc}")

    resnet34_time, resnet34_train_acc, resnet34_test_acc = test_resnet("34")

    print(f"ResNet34 Time: {resnet34_time}")
    print(f"ResNet34 Train Accuracy: {resnet34_train_acc}")
    print(f"ResNet34 Test Accuracy: {resnet34_test_acc}")

    resnet50_time, resnet50_train_acc, resnet50_test_acc = test_resnet("50")

    print(f"ResNet50 Time: {resnet50_time}")
    print(f"ResNet50 Train Accuracy: {resnet50_train_acc}")
    print(f"ResNet50 Test Accuracy: {resnet50_test_acc}")


    # train_dl, test_dl = get_data(vgg19_t)
    # print("here")
    # print(models.VGG19_Weights)

    # resnet34 = torchvision.models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device)

    # print("RESNET34 RESULTS")
    # print(test_resnet(resnet34, "34"))
    # resnet50 = torchvision.models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)

    # print("RESNET50 RESULTS")
    # print(test_resnet(resnet50, "50"))

    # vgg19 = torchvision.models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
    # print("VGG19 RESULTS")
    # print(test_vgg(vgg19, "19"))
    # vgg11 = torchvision.models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1).to(device)
    # print("VGG11 RESULTS")
    # print(test_vgg(vgg11, "11"))
    # vgg11 = torchvision.models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
    # vgg11.classifier = nn.Linear(vgg11.classifier[0].in_features, 10)
    # vgg11_t = models.VGG11_Weights.IMAGENET1K_V1.transforms()

if __name__ == '__main__':
    main()