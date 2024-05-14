print('Starting training..', flush=True)

from models.unet import *
from training.trainer import * 
from data.loader import *  

import sys, argparse

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
physical_devices = tf.config.experimental.list_physical_devices('GPU')






parser = argparse.ArgumentParser(description='Training for reconstruction of PSF duplicated images')
parser.add_argument('--train_x', type=str, required=True,help='The path to circulator input training data. Must contain a single .tif file with dimensions of NxN')
parser.add_argument('--train_y', type=str, required=True, help='The path to model output training data. Must contain 3 .tif files, corresponding to 3 channels, with suffices _T0, _T1, _T2')
parser.add_argument('--valid_x', type=str, required=True,help='The path to circulator input validation data. Must contain a single .tif file with dimensions of NxN')
parser.add_argument('--valid_y', type=str, required=True, help='The path to model output training data. Must contain 3 .tif files, corresponding to 3 channels, with suffices _T0, _T1, _T2')

args = parser.parse_args()

train_data = Loader(args.train_x, args.train_y)
valid_data = Loader(args.valid_x, args.valid_y)

train_data.load_xy_data()
valid_data.load_xy_data()

model  = GetFullModel((None, None, 1),(None, None,3))

print('Generated model',flush=True)
print(physical_devices,flush=True)

trainer = TrainModel(model)
trainer.set_train_data(train_data.x_data, train_data.y_data, valid_data.x_data, valid_data.y_data)
trainer.set_save_path(os.getcwd())
trainer.train()

