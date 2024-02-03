from models.unet import *
from training.trainer import * 
from data.loader import *  
import sys, numpy as np
import argparse


import tifffile, os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

parser = argparse.ArgumentParser(description='Process PSF duplicated microscopy images and obtain their reconstruction')
parser.add_argument('--folder_list', type=str, required=True,help='The path to .txt file containing all paths to .tif files to be analyzed.')
parser.add_argument('--weights', type=str, required=True, help='Path to .h5 file of the weights of the network')
parser.add_argument('--contrast', type=float, required=True,help = 'Contrast adjustment value for the .tif files.')
parser.add_argument('--backgr', nargs='?', const='mean', type=str, help ='[Optional] Background subtraction. Can be mean or none')

args = parser.parse_args()

folder_list = []
with open(args.folder_list) as f:
    for line in f:
        folder_list.append(line)




model = GetFullModel((None, None, 1),(None, None, 3))
trainer = TrainModel(model)
contrast = np.float64(args.contrast)
weights  = args.weights




if not os.path.isfile(weights):
    raise Exception('Not a valid weights file. Provide an absolute path to the .h5 weights file.')

if args.backgr not in ['mean', 'none']:
    raise Exception('Invalid argument for background subtraction')


y_data = []  
for file_path in  folder_list:
    if os.path.isfile(file_path):
        test_data = Loader(file_path, '')
        test_data.load_test_data(file_path,background_subtraction=args.backgr )
        y_data.append(np.uint16(trainer.predict_transposed(contrast*test_data.test_data, weights)))

y_data = np.concatenate(y_data,axis=0)




colors = ['T0','T1','T2']
for id, color in enumerate(colors):
    tifffile.imwrite(color+'_decirculated.tif',np.uint16(y_data[:,:,:,id]))