# Generative Dynamic Patch Attack on Evidential Uncertainty trained model

This folder provides code to perform GDPA on evidential uncertainty trained model. GDPA performed is based on [paper: Generative Dynamic Patch Attack](https://arxiv.org/pdf/2111.04266.pdf) and [base code](https://github.com/lxuniverse/gdpa)

### Folder and file descriptions
* `Data folder - add training and validation dataset to this folder`
* `Output folder - if you are using shell scripts to run the code then output files are generated in this folder`
* `data file - dataset is imported and pre-processed in this file`
* `gdpa file - main class and GDPA attack functions are defined in this file`
* `models file - custom models are defined in this file`
* `utils file - help methods are defined in this file`
* `gdpa.sh - shell file to run the main function

### How to run
* Shell file `gdpa.sh` can be used to run the code on the server

* To run the code on the terminal: 
 
 `python gdpa.py --dataset vggface --data_path Data --vgg_model_path saving_normal_way_model_Face_data224.pt --epochs 50 --patch_size 23`

* Some of the arguements which can be used are : 

``` 
python main.py [--dataset] [--data_path] [--vgg_model_path] [--epochs] [--patch_size] [--batch_size] [--device]   

  --dataset           name of the dataset used   
  
  --data_path         dataset path   
  
  --epochs            desired number of epochs   
  
  --patch_size        patch size with respect to dataset is entered here   
  
  --batch_size        batch size is defined here    
  
  --vgg_model_path    path to the trained model is given here
  
  --device            cpu or gpu or other server where code is ran is mentioned here           
``` 
* output plots are generated in root folder
