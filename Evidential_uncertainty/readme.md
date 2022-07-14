# Evidential Uncertainty Estimation

The purpose of this repository is to train Evidential Uncertainty Estimation model. 

This code is modified version of the base (original) code provided by the [paper: Evidential Deep Learning to Quantify Classification Uncertainty](https://arxiv.org/abs/1806.01768) and [base code](https://github.com/dougbrion/pytorch-classification-uncertainty)

### Folder and file descriptions
* `Data folder - add training and validation dataset to this folder`
* `Output folder - if you are using shell scripts to run the code then output files are generated in this folder`
* `results folder - trained model in the format given by base paper is saved in this folder`
* `saved_model folder - trained model in normal format is saved in this folder`
* `data file - dataset is imported and pre-processed in this file`
* `helpers file - helper methods are written in this file`
* `losses file - different loss functions are defined in this file`
* `train file - training method is defined in this file`
* `main file - main class is defined in this file`
* `exp_train.sh - shell function is written in this file to run the main function (can be ignored if shell programming format is not followed)`

### How to run
* Shell file `exp_train.sh` can be used to run the code on the server

* To run the code on the terminal: `python main.py --train --dropout --uncertainty --mse --epochs 50`

* Some of the arguements which can be used are : ``` python main.py [--h] [--train] [--epochs] [--dropout] [--uncertainty] [--mse] [--examples]   

  --h, --help       show this help message and exit   
  
  --train           to train the network   
  
  --epochs EPOCHS   desired number of epochs   
  
  --dropout         whether to use dropout or not   
  
  --uncertainty     to use uncertainty    
  
  --mse             to use mse uncertainty. Sets loss function to Expected Mean Square Error    
  
  --examples        to print example data           ``` 
