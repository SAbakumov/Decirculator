
# Repository accompanying the Circulator paper. 

## How to run
There are two ways to explore this repository: either through Google Colab, or by cloning the repository locally. The Google
Colab environment is highly interactive and we encourage the users to first explore the methods proposed in our work through 
the interactive notebooks provided.

### Colab
To use Colab, you need a gmail account, and the only thing you have to do is to go to:

- https://colab.research.google.com/github/SAbakumov/Decirculator

There you can select out of 2 available notebooks:

- decirculator_smlm.ipynb -> This is the notebook focussing on the neural network application to PSF reconstruction. There, you can
run the CNN and compare the obtained reconstructed images
- origami_tracking.ipynb -> This is the notebook about the origami tracking analysis. In there, all the analysis regarding tracking,
MSD and diffusion coefficient calculation can be found and reproduced. 

### Locally

Installation:

Pull the repository into an empty directory and download all the necessary data, which includes the model weights from: 

https://doi.org/10.5281/zenodo.11196841

Install Python 3.9.7+ from https://www.python.org/. Then, follow the instructions:

Open Terminal or Powershell(Windows):
1) Create a new python environment as where PATH_TO_ENVIRONMENT is the desired installation folder.
   - For MAC/Linux: Open terminal. Then -> sudo pip install virtualenv -> mkdir PATH_TO_ENVIRONMENT && cd PATH_TO_ENVIRONMENT -> virtualenv env -> source env/bin/activate
   - For Windows: Open powershell. Then -> python -m venv PATH_TO_ENVIRONMENT -> PATH_TO_ENVIRONMENT\Scripts\Activate.ps1
2) Install the required packages by first navigating to the cloned directory folder:
   - cd REPOSITORY_FOLDER
   - pip install -r requirements.txt
3) Now everything is set!


To perform reconstruction, you'd need to run the reconstruct.py from within the repository folder. There are 4 (3+1 optional) arguments.

- The folder_list: this is a .txt file containing all absolute paths to your .tif files that you wish to reconstruct. If there is chronological order in the data,
make sure that the files are ordered accordingly. An example of what the folder_list.txt should look like is provided in the repository as folder_list.txt.

- The weights: this is a .h5 file containing the weights of the network that you have downloaded from the link above. An absolute path to this file is required as input

- Contrast: this is a number, that is multiplied with your input data if the contrast differs from the one that the network has been trained on. 

- [Optional] Background:  Can be either mean or none. Default is mean. This argument is to tell which type of subtraction to use for the background.

An example call of the function: -> 

```
python reconstruct.py --folder_list PATH/TO/FOLDER/LIST/folder_list.txt --weights PATH/TO/WEIGHTS/model_final_weights.h5 --contrast 1 --backgr mean
```


The help command can also guide you if needed: python reconstruct.py -h
```
usage: reconstruct.py [-h] --folder_list FOLDER_LIST --weights WEIGHTS --contrast CONTRAST [--backgr [BACKGR]]

Process PSF duplicated microscopy images and obtain their reconstruction

optional arguments:
  -h, --help            show this help message and exit
  --folder_list FOLDER_LIST
                        The path to .txt file containing all paths to .tif files to be analyzed.
  --weights WEIGHTS     Path to .h5 file of the weights of the network
  --contrast CONTRAST   Contrast adjustment value for the .tif files.
  --backgr [BACKGR]     [Optional] Background subtraction. Can be mean or none
```

### Output

The output of the reconstruction pipeline are three files: T0_decirculated.tif, T1_decirculated.tif, T2_decirculated.tif, which are 3 different colors corresponding to PSF's in the microscopy setup. These are stored in the same directory as where you've cloned your repository to.

### Training

To run the training, you can call main_train.py with the correct arguments. 

The help command can also guide you if needed: python main_train.py -h
```

usage: main_train.py [-h] --train_x TRAIN_X --train_y TRAIN_Y --valid_x VALID_X --valid_y VALID_Y

Training for reconstruction of PSF duplicated images

optional arguments:
  -h, --help         show this help message and exit
  --train_x TRAIN_X  The path to circulator input training data. Must contain a single .tif file with dimensions of NxN
  --train_y TRAIN_Y  The path to model output training data. Must contain 3 .tif files, corresponding to 3 channels, with suffices _T0, _T1, _T2
  --valid_x VALID_X  The path to circulator input validation data. Must contain a single .tif file with dimensions of NxN
  --valid_y VALID_Y  The path to model output training data. Must contain 3 .tif files, corresponding to 3 channels, with suffices _T0, _T1, _T2

```


The files in the train and validation paths must be .tif files, in separate folders, formatted as in description. An example of the data input folder structure can be found in training\example_data\x_data and training\example_data\y_data. 
