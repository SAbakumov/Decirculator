import os, numpy as np  
import tifffile


class Loader():
    def __init__(self,x_data_path, y_data_path):

        self.x_data_path = x_data_path    
        self.y_data_path = y_data_path   

    def load_xy_data(self):
        for file in os.listdir(self.x_data_path):
            if '.tif' in file:
                self.x_data = np.float16(np.expand_dims(tifffile.imread(os.path.join(self.x_data_path,file)),-1))
                self.x_data = self.x_data - np.mean(self.x_data)

        self.y_data  = []
        for channel in ['T0','T1','T2']:
            if channel in file:
                self.y_data = np.expand_dims(np.float16(tifffile.imread(os.path.join(self.y_data_path,file))),-1)


        self.y_data = [self.y_data[key] for key in self.y_data.keys()]
        self.y_data = np.float16(np.stack(self.y_data,axis = -1))
    
    def load_test_data(self, path,background_subtraction):

        self.test_data = np.float16(np.expand_dims(tifffile.imread(path),-1))
        if background_subtraction=='mean':
            self.test_data = self.test_data -  np.mean(self.test_data)
        if background_subtraction=='none':
            return
            # self.test_data = self.test_data -  np.mean(self.test_data)
