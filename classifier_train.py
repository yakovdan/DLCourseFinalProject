from __future__ import print_function
from __future__ import division

import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset as BaseDataset
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import logging
from PIL import Image
import cv2
import os

IMAGE_PATH = os.getcwd() + "/DataSet_all_aligned_and_segmented/"


class load_data(BaseDataset):
    def __init__(self, batchsize, transforms, is_val):

        self.batchsize = batchsize

        self.batch_size = batchsize
        self.is_val = is_val
        self.num_classes = num_classes
        Open_dir = IMAGE_PATH + "/Open"
        close_dir = IMAGE_PATH + "/Close"

        open_images = []
        for filename in os.listdir(Open_dir):
            img = cv2.imread(os.path.join(Open_dir, filename))

            if img is not None:
                img = img[:, :, ::-1].copy()
                open_images.append(transforms(Image.fromarray(img)))
        close_images = []
        for filename in os.listdir(close_dir):
            img = cv2.imread(os.path.join(close_dir, filename))
            if img is not None:
                img = img[:, :, ::-1].copy()
                close_images.append(transforms(Image.fromarray(img)))
        self.images = open_images + close_images
        # self.labels = np.concatenate((np.matlib.repmat([1,0], len(open_images), 1),np.matlib.repmat([0,1], len(close_images), 1)))
        self.labels = np.concatenate(
            (np.ones((len(open_images), 1)), np.zeros((len(close_images), 1))))

        Open_val_dir = IMAGE_PATH + "/Open_val"
        close_val_dir = IMAGE_PATH + "/Close_val"

        open_val_images = []
        for filename in os.listdir(Open_val_dir):
            img = cv2.imread(os.path.join(Open_val_dir, filename))
            if img is not None:
                img = img[:, :, ::-1].copy()
                open_val_images.append(transforms(Image.fromarray(img)))
        close_val_images = []
        for filename in os.listdir(close_val_dir):
            img = cv2.imread(os.path.join(close_val_dir, filename))
            if img is not None:
                img = img[:, :, ::-1].copy()
                close_val_images.append(transforms(Image.fromarray(img)))
        self.images_val = open_val_images + close_val_images
        # self.labels_val = np.concatenate((np.matlib.repmat([1,0], len(open_val_images), 1),np.matlib.repmat([0,1], len(close_val_images), 1)))
        self.labels_val = np.concatenate(
            (np.ones((len(open_val_images), 1)), np.zeros((len(close_val_images), 1))))

    def __getitem__(self, idx):
        if (self.is_val == False):
            sample = {'images': self.images[idx],
                      'labels': self.labels[idx]}  # scores_slices
        else:
            sample = {'images': self.images_val[idx],
                      'labels': self.labels_val[idx]}
        return sample

    def __len__(self):
        if (self.is_val == False):
            return len(self.labels)
        else:
            return len(self.labels_val)


print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 2  # 6

# Batch size for training (change depending on how much memory you have)
# batch_size = 32#3# 1#3#5#1#8

# Number of epochs to train for
num_epochs = 20

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True  # True


def my_logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logging.basicConfig(
        filename=logger_name,
        filemode='w',
        format='%(asctime)s, %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    logger.addHandler(file_handler)
    return logger


def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    softmax = nn.Softmax(dim=1)
    # return torch.mean(torch.sum(- soft_targets * logsoftmax(pred),
    return torch.mean(torch.sum(- soft_targets * torch.log(softmax(pred)), 1))


def train_model(model, dataloaders, criterion, optimizer, batch_size, num_epochs=25, is_inception=False):
    since = time.time()
    # settings = SegSettings(setting_dict, write_logger=True)
    # my_logger(settings.simulation_folder + '\logger')

    val_acc_history = []
    val_acc0_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_corrects_class0 = 0.0
            non_zero_count = 0
            # Iterate over data.
            for i, sample in enumerate(dataloaders[phase]):

                images = sample['images'].float()
                # print(images.shape)
                try:
                    inputs = images.to(device)
                except:
                    continue

                labels = sample['labels'].long()
                # labels = torch.tensor(sample['labels']).long().to(device)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        # _, trues = torch.max(labels,axis=1)
                        # softmax= nn.Softmax(dim=1)
                        # outputs = softmax(outputs)
                        # m = nn.Sigmoid()
                        labels = labels[:, 0]
                        loss = criterion(outputs, labels)
                        # loss = cross_entropy(outputs, labels)

                    # correct = len(np.nonzero((np.argsort(outputs.cpu().detach().numpy(),axis=1)[:,-1]==np.argsort(labels.cpu().numpy(),axis=1)[:,-1]))[0])+len(np.nonzero((np.argsort(outputs.cpu().detach().numpy(),axis=1)[:,-1]==np.argsort(labels.cpu().numpy(),axis=1)[:,-2]))[0])
                    _, preds = torch.max(outputs, 1)
                    # if phase == 'val':
                    #     print(preds)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_corrects_class0 += torch.sum(preds[labels.data == 0] == labels.data[labels.data == 0])
                non_zero_count += torch.count_nonzero(labels.data == 0)
            epoch_loss = running_loss / (len(dataloaders[phase].dataset))
            epoch_acc = running_corrects.double() / (len(dataloaders[phase].dataset))
            if phase == 'val':
                val_acc_history.append(epoch_acc.cpu().detach().item())
                v0 = running_corrects_class0 / non_zero_count
                val_acc0_history.append(v0.cpu().detach().item())
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val':
                print('Best val accuracy: {:.4f}'.format(max(val_acc_history)))
                print('Best val0 accuracy: {:.4f}'.format(max(val_acc0_history)))
            # logging.info(phase)
            # logging.info("Loss %.4f" % epoch_loss)
            # logging.info("Acc %.4f" % epoch_acc)

            # deep copy the model
            if phase == 'val' and ((epoch_acc == best_acc) or (epoch_acc > best_acc)):
                best_acc = epoch_acc
                best_acc_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

        save_path = os.path.join(

            IMAGE_PATH + '/classification_model', "s_checkpoint_%04d.pt" % (epoch)
        )
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_loss": epoch_acc,
                "optimizer": optimizer.state_dict(),
            },
            save_path,
        )

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, best_acc_epoch, best_acc, val_acc0_history


def test_model(model, dataloaders, criterion, optimizer, batch_size, num_epochs=25, is_inception=False):
    since = time.time()
    val_acc_history = []
    val_acc0_history = []
    val_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, sample in enumerate(dataloaders[phase]):

                images = sample['images'].float()
                # print(images.shape)
                try:
                    inputs = images.to(device)
                except:
                    continue

                labels = sample['labels'].long()
                # labels = torch.tensor(sample['labels']).long().to(device)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        # _, trues = torch.max(labels,axis=1)
                        # softmax= nn.Softmax(dim=1)
                        # outputs = softmax(outputs)
                        # m = nn.Sigmoid()
                        labels = labels[:, 0]
                        loss = criterion(outputs, labels)
                        # loss = cross_entropy(outputs, labels)

                    # correct = len(np.nonzero((np.argsort(outputs.cpu().detach().numpy(),axis=1)[:,-1]==np.argsort(labels.cpu().numpy(),axis=1)[:,-1]))[0])+len(np.nonzero((np.argsort(outputs.cpu().detach().numpy(),axis=1)[:,-1]==np.argsort(labels.cpu().numpy(),axis=1)[:,-2]))[0])
                    _, preds = torch.max(outputs, 1)
                    if preds.data == 0:
                        print("FOUND ****")

                    # if phase == 'val':
                    #     print(preds)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / (len(dataloaders[phase].dataset))
            epoch_acc = running_corrects.double() / (len(dataloaders[phase].dataset))
            val_acc_history.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} Best val accuracy: {:.4f}'.format(phase, max(val_acc_history)))
            # logging.info(phase)
            # logging.info("Loss %.4f" % epoch_loss)
            # logging.info("Acc %.4f" % epoch_acc)

            # deep copy the model
            if phase == 'val' and ((epoch_acc == best_acc) or (epoch_acc > best_acc)):
                best_acc = epoch_acc
                best_acc_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()
        save_path = os.path.join(

            IMAGE_PATH + '/classification_model', "s_checkpoint_%04d.pt" % (epoch)
        )
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_loss": epoch_acc,
                "optimizer": optimizer.state_dict(),
            },
            save_path,
        )

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, best_acc_epoch, best_acc


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# Data augmentation and normalization for training
# Just normalization for validation
# Setup the loss fxn
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 224
data_transforms = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Initialize the model for this run
best_validation_acc = 0
for batch_size in [16]:  # [4, 8, 16, 32, 64]:
    for weight_decay_power in [0.0003]:  # [0.01, 0.007, 0.003, 0.001, 0.0007, 0.0003, 0.0001]:
        print(f'Testing : batch size: {batch_size}, weight decay: {weight_decay_power}')
        model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
        # Print the model we just instantiated
        # print(model_ft)

        print("Initializing Datasets and Dataloaders...")
        image_datasets = {'train': load_data(batch_size, data_transforms, 0),
                          'val': load_data(batch_size, data_transforms, 1)}
        # Create training and validation datasets
        # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
        # Create training and validation dataloaders
        print('image_datasets loaded')
        dataloaders_dict = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x
            in ['train', 'val']}

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9, weight_decay=weight_decay_power)
        model_ft = model_ft.to(device)
        model_ft, hist, best_acc_epoch, best_acc, hist0 = train_model(model_ft, dataloaders_dict, criterion,
                                                                      optimizer_ft,
                                                                      batch_size, num_epochs=num_epochs,
                                                                      is_inception=(model_name == "inception"))
        torch.save({"state_dict": model_ft.state_dict()}, IMAGE_PATH + "/best_classifier.pt")
        current_validation_acc = np.max(hist)
        val0_acc = np.max(hist0)
        if best_validation_acc < current_validation_acc:
            best_validation_acc = current_validation_acc
            best_val0 = val0_acc
            best_parameters = (batch_size, weight_decay_power)
            print(f"Found current best: {best_parameters}, val: {current_validation_acc}, val0: {val0_acc}")

print(best_parameters, best_validation_acc, best_val0)
