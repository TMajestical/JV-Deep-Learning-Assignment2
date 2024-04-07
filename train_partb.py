#JV

import os
import time
from tqdm.notebook import tqdm

import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import wandb

import torch.optim as optims
from torch.utils.data import Dataset, DataLoader,ChainDataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

import matplotlib.pyplot as plt

import argparse

seed = 76 #setting this as seed wherever randomness comes

torch.manual_seed(seed)
np.random.seed(seed)

class DataPreparation:

    def __init__(self,data_dir,device,default_transforms=None):

        self.base_dir  = data_dir
        self.device = device
        self.default_transforms = default_transforms


    def create_dataloader(self,sub_dir,batch_size=16,shuffle=True,num_workers=2,data_augmentation_transforms = None,pin_memory = False):

        """
        Method to create dataset and return dataloader after applying all necessary transforms.

        params:

            sub_dir : "train/" or "validation/" or "test/"
            batch_size : The batch size in which training has to be performed.
            shuffle : whether shuffling must be done before sampling.
            num_works : Number of workers to be used on the dataset.
            data_augmentation_transforms : Either None or List of List of transforms, with each sub-list leading to a dataset.

        Returns:

            Dataloader corresponding to the dataset.

        """

        print(f"Preparing data from {sub_dir}")


        ## The most basic list of transforms applied to the orignal train dataset and validation and test dataset.
        vanilla_transforms = [self.default_transforms]

        if ("train" in sub_dir) and (data_augmentation_transforms): ## if data augmentation is to be done

            original_dataset = torchvision.datasets.ImageFolder(root=self.base_dir+sub_dir,transform=transforms.Compose(vanilla_transforms))

            dataset_list = [original_dataset]

            for aug_transform in data_augmentation_transforms:

                cur_data_transforms_list = [self.default_transforms] + aug_transform
                cur_dataset = torchvision.datasets.ImageFolder(root=self.base_dir+sub_dir,transform=transforms.Compose(cur_data_transforms_list))
                dataset_list.append(cur_dataset)

            self.dataset = ConcatDataset(dataset_list)
        else:

            self.dataset = torchvision.datasets.ImageFolder(root=self.base_dir+sub_dir,transform=transforms.Compose(vanilla_transforms))



        ## Now create the data loader

        sampler = None ## unless the dataloading is distributed across devices or processes.

        if "train" in sub_dir:

            #torch.distributed.init_process_group(rank=0,world_size = 4)

            #sampler = DistributedSampler(self.dataset)

            self.loader = torch.utils.data.DataLoader(dataset=self.dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory = pin_memory,sampler=sampler)

        else:

            num_workers = 3
            pin_memory = False

            self.loader = torch.utils.data.DataLoader(self.dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory = pin_memory,sampler=sampler)

        return self.loader

class Experiment:

    """
    Class to create and conduct experiments
    """

    def __init__(self,device,base_dir,wandb_logging=False,on_kaggle = False):

        self.device = device
        self.base_data_dir = base_dir
        self.wandb_logging = wandb_logging
        self.on_kaggle = on_kaggle

    def create_dataloaders(self,batch_size,shuffle,list_of_train_data_augmentation_transforms,num_workers=0,pin_memory=False):

        """
        Method to create dataloaders for train,test and validation datasets, with the help from the DataPreparation class.

        params:

            batch_size : The training batch size (also applied to test and validation loaders, but anyway its still the same).
            shuffle : True/False, whether to shuffle data before sampling.
            list_of_train_data_augmentation_transforms : None, if no data augmentation or List of List of transforms, with each sub-list leading to a dataset.
            num_workers : Number of workers to support the dataloader, default is 0.
            pin_memory : Default is False. Pinning memory makes data loading efficent when a accelerator is used and num of workers>0.

        Returns:

            Torch dataloader objects for training,testing and validation data.

        """
        ## Create an object of the data preparation class
        dataprep = DataPreparation(data_dir=self.base_data_dir,device = self.device,default_transforms = self.default_transforms)

        if (not self.device == "cpu") and num_workers>0:
            pin_memory = True

        ## create a train dataset loader
        self.train_loader = dataprep.create_dataloader(sub_dir = "train/",batch_size = batch_size,shuffle = shuffle, num_workers = num_workers,data_augmentation_transforms = list_of_train_data_augmentation_transforms,pin_memory=pin_memory)

        ## it is not efficient to pin memory for validation and test datasets,as they are relatively small.
        self.val_loader = dataprep.create_dataloader(sub_dir = "validation/",batch_size = batch_size,shuffle = shuffle, num_workers = num_workers,pin_memory=False)
        self.test_loader = dataprep.create_dataloader(sub_dir = "test/",batch_size = batch_size,shuffle = shuffle, num_workers = num_workers,pin_memory=False)

        return self.train_loader,self.val_loader,self.test_loader


    def createResNet(self,num_output_neurons):

        """
        Method to Create the ResNet architecture with the pre-trained weights. Making suitable for finetuning.

        Params:

            num_output_neurons: Number of neurons in the output layer.


        Returns:

            None.

        """

        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) ##using weights of IMAGENET1K_V2, which gave an accuracy of 80%

        weights = ResNet50_Weights.IMAGENET1K_V2
        self.default_transforms = weights.transforms() ## these are the transforms to be applied on the input images before feeding to the model

        # Freeze all layers except the last layer
        for param in self.model.parameters():
            param.requires_grad = False

        ## Now the output layer of ResNet50 model simply passes through the output of the penultimate hidden layer.
        ## last fully connected layer of resnet can be accessed using model.fc

        resnet_last_fc_size = self.model.fc.in_features ## first store the output size.
        self.model.fc = nn.Identity()

        ## now add an output layer with 10 neurons to ResNet50 model
        self.model.fc = nn.Linear(resnet_last_fc_size, out_features=num_output_neurons)

        ## Initializae weights and biases of this layer. Using xavier initialization for weights
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        self.model.fc.bias.data.fill_(0.01)

        # Make the weights of the last layer trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True

        ## Use LogSoftmax activation for the output layer
        #model.fc = nn.LogSoftmax(model.fc)

        self.model.to(self.device) ## move the model to the device


    def compute_accuracy(self,model,data_iterator):

        """
        Method to compute the accuracy of the given model over the dataset in the data_iterator.

        params:

            model : The torch neural net model whose performance has to be measured.

            data_iterator : The data iterator over which the computation of the metrics has to be done.

        Returns:

            loss,accuracy of the "model" over the "data_iterator".
        """

        correct_preds = 0
        total_preds = 0

        loss = 0
        train_mode = model.training

        # since we're testing, switch of train mode if it is on.
        if train_mode:
            model.eval()

        with torch.no_grad(): ##don't compute gradients
            for data in data_iterator:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device) ## move the inputs and labels to the device

                # calculate outputs by running images through the network
                outputs = model(images)
                loss += self.criterion(outputs, labels).item() * images.size(0) ## loss.item() is average loss of the batch, so multiply by batch size.

                preds = torch.max(outputs.data, 1)[1]

                total_preds += images.size(0)
                correct_preds += (preds == labels).sum().item()

        if train_mode: # if model was originally in train mode, switch it back to train mode.
            model.train() ## switch back to train mode

        #print(f'Accuracy of the model on the {len(data_iterator.dataset.samples)} test images: {round(100*correct/total,2)} %')

        accuracy = round(100*correct_preds/total_preds,2)
        loss = round(loss/total_preds,2)

        return loss,accuracy

    def train(self,lr,weight_decay,loss,optimiser,epochs):

        """
        The method to perform the training, assuming model is already created using createResNet method.

        Params:

            lr : Learning rate
            weight_decay : l2 regularization parameter.
            loss : string, loss type. currently only "crossentropy" is supported
            optimiser : "adam","nadam","rmsprop".
            epochs : number of epochs to train.

        Returns:

            None.
        """

        ## specify the optimiser
        if optimiser.lower() == "adam":
            self.optimiser = optim.Adam(self.model.parameters(), lr=lr,weight_decay=weight_decay)

        elif optimiser.lower() == "nadam":
            self.optimiser = optim.NAdam(self.model.parameters(), lr=lr,weight_decay=weight_decay)

        elif optimiser.lower() == "rmsprop":
            self.optimiser = optim.RMSprop(self.model.parameters(), lr=lr,weight_decay=weight_decay)

        ## Specify the loss criteria
        if loss.lower() == "crossentropy":
            self.criterion = nn.CrossEntropyLoss().to(self.device)



        start_time = time.time()

        ## loop over the dataset multiple times
        for epoch in tqdm(range(epochs)):

            correct_preds = 0
            total = 0
            count = 0
            epoch_loss = 0.0


            for i, data in enumerate(self.train_loader):

                ## i is batch index

                images, labels = data[0].to(self.device),data[1].to(self.device)  ## move the images and labels to the device.


                # zero the parameter gradients
                self.optimiser.zero_grad()

                # forward + backward + optimize
                # logsoftmax is the output activation
                outputs = F.log_softmax(self.model(images).to(self.device),dim=1)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimiser.step()

                epoch_loss +=  loss.item() * images.size(0) ## loss.item() is average loss of the batch, so multiply by batch size.

                preds = torch.max(outputs.data, 1)[1]

                total += images.size(0)
                correct_preds += (preds == labels).sum().item()


            train_accuracy = round(100*correct_preds/total,2)
            train_loss = epoch_loss/total

            val_loss,val_accuracy = self.compute_accuracy(self.model,self.val_loader)

            if epoch%5 == 0:


                if self.on_kaggle:
                    torch.save(self.model, "/kaggle/working/model")
                else:
                    torch.save(self.model, "Model")


            if epoch == 0:
                print(f"Samples in Train Data : {total}")

            if self.wandb_logging:

                wandb.log({'train loss': train_loss, 'train accuracy': train_accuracy, 'Validation loss': val_loss, 'Validation accuracy': val_accuracy,'epoch': epoch+1})

            print(f'Epoch : {epoch+1}\t Train Accuracy : {train_accuracy:.2f}%\t Train loss: {train_loss:.2f}\t Validation Accuracy : {val_accuracy:.2f}%\t Validation Loss : {val_loss:.2f}')
            epoch_loss = 0.0

        print('Finished Training!!')

        end_time = time.time() - start_time
        print(f"Time Taken for Training: {round(end_time/60,2)}")

    def test_model(self):
        
        ## Compute and Report the test accuracy

        test_loss,test_accuracy = self.compute_accuracy(self.model,self.test_loader)
        print(f'Test Accuracy : {test_accuracy:.2f}%\t Test loss: {test_loss:.2f}')

def setup_and_start_expt(config,wandb_log=False,data_dir = "inaturalist_12K/",device_to_use=None):
    ##using apple silicon GPU

    wandb_logging = wandb_log

    if not device_to_use == None:
        device = device_to_use
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_data_dir = "inaturalist_12K/"

    print(f"Using {device}")

    ## dataloader creation hyperparams:

    batch_size = config['batch_size']
    shuffle = True
    num_workers = 1
    pin_memory = False

    #RandomSolarize(threshold=192.0)

    if config['data_aug']:

        train_data_augmentation_transforms1 = [transforms.RandomPerspective(p=1)] ## Random perspective transform
        train_data_augmentation_transforms2 = [transforms.ColorJitter(brightness=.5, hue=.5)] ## colour jitter
        #train_data_augmentation_transforms = [transforms.ElasticTransform()] ## this is good, but computationally extremely expensive

        list_of_train_data_augmentation_transforms = [train_data_augmentation_transforms1,train_data_augmentation_transforms2]
        list_of_train_data_augmentation_transforms = list_of_train_data_augmentation_transforms[:config['data_aug']]

    else: ## if no data augmentation, train_loader returned would just be on the original dataset.

        list_of_train_data_augmentation_transforms = []


    ## create an experiment
    experiment = Experiment(device=device,base_dir = base_data_dir, wandb_logging=wandb_logging)

     ## CNN Hyperparams
    num_output_neurons =  10

    ## create CNN model
    experiment.createResNet(num_output_neurons)

    ##create data loaders for train, validation and test datasets.
    train_loader,val_loader,test_loader  = experiment.create_dataloaders(batch_size=batch_size,shuffle=shuffle,list_of_train_data_augmentation_transforms=list_of_train_data_augmentation_transforms,num_workers=num_workers,pin_memory=pin_memory)


    ##training Hyper Params:

    lr = config['lr']
    weight_decay = config['weight_decay']
    optimiser = config['optimiser']
    epochs = config['epochs']
    loss = "crossentropy"

    experiment.train(lr = lr,weight_decay = weight_decay,loss = loss,optimiser = optimiser,epochs = epochs)

    experiment.test_model()


custom_config = {

        'optimiser': "adam",

        'lr' : 1e-3,

        'weight_decay': 0,

        'epochs' : 3,

        'batch_size': 32,

        'data_aug' : 1
    }

if __name__ == '__main__': ## run the following lines only if this code is explicitly invoked, rather than just imported, may be by WandBSweep.py module
    """
    Provide support for some important arguments
    """

    parser = argparse.ArgumentParser()


    parser.add_argument("-wp", "--wandb_project", type=str, default=None, help="Project name used to track experiments in Weights & Biases dashboard.")

    parser.add_argument("-we", "--wandb_entity", type=str, default=None, help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")

    parser.add_argument("-e", "--epochs", type=int, default=3,help="Number of epochs to train neural network.")

    parser.add_argument("-b", "--batch_size", type=int, default=32,help="Batch size used to train neural network.")

    parser.add_argument("-o", "--optimizer", type=str, default="adam",choices=["rmsprop", "adam", "nadam"],help="Optimizer used to minimize the loss.")

    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3,help="Learning rate used to optimize model parameters.")

    parser.add_argument("-d", "--device", type=str, default=None,help="The device on which the training happens.")

    ## these are fixed to make things easier for the evaluator.
    data_aug = 1

    args = parser.parse_args()

    print("This is the configuration:\n")

    print(f"WandB Project : {args.wandb_project}\n")
    print(f"WandB Entity : {args.wandb_entity}\n")
    print(f"Optimizer : {args.optimizer}\n")
    print(f"Learning Rate : {args.learning_rate}\n")
    print(f"Batch Size : {args.batch_size}\n")
    print(f"Epochs : {args.epochs}\n")
    
    """
    ************ Only when WandB project or entity is specified, wandb logging would happen ************
    """

    if not args.wandb_project and not args.wandb_entity:
        wandb_log = False
    else:
        wandb_log = True


    if wandb_log:

        run = wandb.init(project=args.wandb_project,entity=args.wandb_entity)

    
    seed = 76 #setting this as seed wherever randomness comes
    torch.manual_seed(seed)
    np.random.seed(seed)

    hyperparm_config = {

        'optimiser': args.optimizer,

        'lr' : args.learning_rate,

        'weight_decay': 0,

        'epochs' : args.epochs,

        'batch_size': args.batch_size,

        'data_aug' : 1
    }

    setup_and_start_expt(hyperparm_config,device_to_use=args.device)