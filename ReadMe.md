JV
CS23M036, MALLADI TEJASVI

This is the Repo for the Second Assignment of CS6910, Fundamentals of Deep Learning at IIT Madras.

The assignment is about building and training-finetuning CNN models for the iNaturalist dataset. Also to finetune an existing model ResNet 50.

After downloading and unzipping the data to get the inaturalist_12K directory.

Step1 :  Make sure that all the codes in this repo are in the same directory level as "inaturalist_12K/".

Step2 : Run Train-Validation-Split.py which randomly samples 20% of the traindata to get validation data.
        This has to be done exactly once, during the project setup.

train_parta.py : This module implements CNN (from scratch) and all the supporting functionalities including data handling, training etc.

The wandb sweeps for hyperparameter tuning were conducted this code. But currently hyperparamter tuning part is disabled.

The default parameters of this code are set according to the best model found during hyper parameter tuning. Most of these hyperparameters could be changed using the following arguments:

       Name	              Default Value	    Description
       
-wp,    --wandb_project    	None	          Project name used to track experiments in Weights & Biases dashboard
-we,    --wandb_entity	    None	          Wandb Entity used to track experiments in the Weights & Biases dashboard.
-e,     --epochs	          8	              Number of epochs to train neural network.
-b,     --batch_size	      16	            Batch size used to train neural network.
-o,     --optimizer	        rmsprop	        choices: ["rmsprop", "adam", "nadam"]
-lr,    --learning_rate	    3e-4            Learning rate used to optimize model parameters
-w_d,   --weight_decay	    5e-4	          Weight decay used by optimizers.
-nhl,   --num_layers	      1	              Number of dense layers in CNN.
-sz,    --hidden_size	      64	            Number of hidden neurons in a dense layers of CNN.
-a,     --activation	      relu	          choices: ["silu", "tanh", "relu"]
-d      --device            None            The device on which the training happens. [When None, the code automatically detects and uses cuda:0 gpu]
        --conv_layers       5               Number of convolutional layers in CNN.
        --conv_factor       1               choices=[0.5,1,2], The factor deciding the number of filters in next layer.
        --num_filters       32              Number of Filters in the first layer of the CNN model.
        --filter_size       5               The size of the convolutional filters.
        --batch_norm        True            choices=[True,False],Whether batch norm should be done.
        --model_path        None            path to the trained model for testing the performance.


The model with best validation accuracy available in the repo as BestModelPartA. Passing this file name along with the argument --model_path would avoid training by loading the model and computing the test and validation accuracy.

To perform training, either the default configuration of best hyperparameters could be used or they could be modified as required. Results for this run could be logged to wandb by adding the wandb.login with key to the code and passing the wandb project name using argument parsing.


train_partb.py : This module implements the finetuning of the pre-trained ResNet50 model on the iNaturalist dataset.

A few experiments have been manually run and hyperparameters that gave the best results were set as the default values, they can be modified using the following hyperparameters:

       Name	              Default Value	    Description
       
-wp,    --wandb_project    	None	          Project name used to track experiments in Weights & Biases dashboard
-we,    --wandb_entity	    None	          Wandb Entity used to track experiments in the Weights & Biases dashboard.
-e,     --epochs	          3               Number of epochs to train neural network.
-b,     --batch_size	      32	            Batch size used to train neural network.
-o,     --optimizer	        adam	          choices: ["rmsprop", "adam", "nadam"]
-lr,    --learning_rate	    1e-3            Learning rate used to optimize model parameters
-d      --device            None            The device on which the training happens. [When None, the code automatically detects and uses cuda:0 gpu]



For the convinience of the evaluator, both train_parta.py and train_partb.py also compute the test accuracy at the end of a run.

Kindly note that in both partA and B, if you find the training/processing to be slow, consider increasing the num_workers in setup_and_start_expt, this must be done consciously based on the number of vCPUs on the system, else the code would crash. Hence this is not provided as an argument.

Thanks.
        
        
