# Evidential Uncertainty Estimation

The purpose of this repository is to train Evidential Uncertainty Estimation model. 

This code is modified version of the base (original) code provided by the [paper: Evidential Deep Learning to Quantify Classification Uncertainty](https://arxiv.org/abs/1806.01768) [base code](https://github.com/dougbrion/pytorch-classification-uncertainty)

## folder and file descriptions
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
