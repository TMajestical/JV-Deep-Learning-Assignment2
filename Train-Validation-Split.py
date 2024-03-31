import os
import shutil
import numpy as np

"""
This code has to be run exactly once during the project setup.

Place this code in the same directory as "inaturalist_12K/".

This code randomly splits 20% of the data as validation data and stores it in a directory called validation.

This should run smoothly on any Linux based machine or MACs.

"""

def make_dir(dir,returnIfDirAlreadyExists=False):
    """
    Function to create a directory, if it doesn't exist
    """
    try:
        os.mkdir(dir)
    except Exception as e:
        if "File exists" in str(e):
            if returnIfDirAlreadyExists:
                return True
            pass
        else:
            print(e)

## Train data downloaded from the given source (https://storage.googleapis.com/wandb_datasets/nature_12K.zip)

"""
Now, the goal is to split 20% of train data, in "train" folder to get validation data.
"""



data_base_dir = 'inaturalist_12K/'

def train_validation_split(base_dir,seed = 76):
    """
    Function to split 20% of the train data into validation data, Uniformly At Random (UAR). Import os and shutil before using this method.

    Note  : Instead of taking 20% of samples randomly out of the entire train data; 20% of train data of each class is taken (UAR), 
    so that for training there is a balance between the number of samples per class.

    Params:

        base_dir : The path to the directory in which the "train/" and "test/" directories are present after unzipping. It is assumed that the given dir path string has a "/ at the end.

        seed : The seed use in the random number generator, default : 76.

    Returns :

        None.
    """

    base_data_dir = base_dir
    train_base_dir = base_data_dir+'train/'
    train_data_class_dirs = os.listdir(train_base_dir)
    
    ## remove dirs starting with "." from the list
    train_data_class_dirs = [i for i in train_data_class_dirs if i[0] != "." ]

    ## Test data is called as val, which is confusing, hence renaming it to test
    os.rename(data_base_dir+"val/",base_data_dir+"test/")
    
    
    ## validation dir
    val_base_dir = base_data_dir+'validation/'
    make_dir(val_base_dir)
    
    ## Iterate over each class and
    ## take 20% data of each class at random as validation data
    
    random_num_generator = np.random.RandomState(seed)
    
    for class_label in train_data_class_dirs:
    
        current_class_train_filenames = os.listdir(train_base_dir+class_label+"/")
    
        num_of_files = len(current_class_train_filenames)
        
        validation_indices = random_num_generator.choice(num_of_files,int(0.2*num_of_files),replace=False)
        train_indices = np.array(list(set(np.arange(num_of_files)).difference(set(validation_indices))))
    
        ##create class dir validation dir
        cur_validation_dir = val_base_dir + class_label +"/"
        make_dir(cur_validation_dir)
        
        for i in validation_indices:
            shutil.move(train_base_dir+class_label+"/"+current_class_train_filenames[i],cur_validation_dir+current_class_train_filenames[i])
        
        print(f"Validation Split for {class_label} is Done!")


base_data_dir = "inaturalist_12K/"

train_validation_split(base_data_dir)

