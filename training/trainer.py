import tensorflow as tf
from tensorflow.keras import layers

import os , time, numpy as np 

print("Tensorflow version " + tf.__version__)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.TPUStrategy(tpu)
    set_tpu = True
except:
    print('No TPU detected, running locally on GPU or CPU')
    set_tpu = False


class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self,save_path, save_best_metric='val_loss',train_loss = 'loss'):
        self.save_best_metric = save_best_metric
        self.train_loss       = train_loss
        self.best = float('inf')
        self.save_path = save_path
    def on_epoch_begin(self, epoch, logs=None):
        self.mark = time.time()

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        t = time.time()- self.mark

        print('Val loss: '+  str(logs[self.save_best_metric])+ ', Train loss: ' +str(logs[self.train_loss])+ ' , Time elapsed:' + str(np.round(t,2)),flush=True)

        if metric_value < self.best:
            self.best = metric_value
            self.model.save_weights('model_denoiserUnet600kRand_Batch16.weights.h5' )

class TrainModel():
    def __init__(self, model):
        self.model = model  

    def set_train_data(self,x_data, y_data,val_data_x,val_data_y):
        self.x_data = x_data  
        self.y_data = y_data 
        self.xval_data  = val_data_x
        self.yval_data  = val_data_y

    def set_save_path(self,save_path):
        self.save_path = save_path

    def penalized_loss(self,y_true, y_pred):
        alpha  =  0.00001
        loss = tf.reduce_mean(tf.keras.metrics.mse(y_true, y_pred))  +alpha* tf.reduce_sum(tf.abs(y_pred))
        return loss
    


    
    def train(self):
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss=self.penalized_loss)
        save_best_model = SaveBestModel(self.save_path)
        self.model.fit( self.x_data, self.y_data,validation_data=( self.xval_data, self.yval_data),batch_size=16,epochs=1000,verbose=0,callbacks = save_best_model,shuffle=True)


    def predict_denoiser(self,test_data, weight_data):
        self.model.load_weights(weight_data)
        y_data = []
        for d in range(0,test_data.shape[0],10):
            data = test_data[d,:,:,:]
            predicted_data = self.model.predict(np.expand_dims(data,0))

           

            y_data.append(predicted_data)

        y_data = np.concatenate(y_data)
        y_data = np.squeeze(np.array(y_data))
        return y_data


    def predict(self, test_data, weight_data):
        self.model.load_weights(weight_data)
        if set_tpu:
            with tpu_strategy.scope():
                self.model.compile()
                self.model.load_weights(weight_data)
        y_data = []
        for d in range(0,test_data.shape[0],10):

            data = test_data[d:d+10,:,:,:]
            if set_tpu:
                with tpu_strategy.scope():
                    print('Using TPU')
                    predicted_data = self.model.predict(data)
            else:
                predicted_data = self.model.predict(data)


            predicted_data[:,:,:,0] = np.roll(predicted_data[:,:,:,0],12, axis=1)
            predicted_data[:,0:12,:,0] = 0
            predicted_data[:,:,:,1] = np.roll(predicted_data[:,:,:,1],-12, axis=1)
            predicted_data[:,-12:-1,:,1] = 0

            y_data.append(predicted_data)

        y_data = np.concatenate(y_data)
        y_data = np.squeeze(np.array(y_data))
        y_data[y_data<=0] = 0
        return y_data

    def predict_transposed(self, test_data, weight_data):
        self.model.load_weights(weight_data)
        y_data = []
        if set_tpu:
            with tpu_strategy.scope():
                self.model.load_weights(weight_data)
        for d in range(0,test_data.shape[0],10):
            data = np.swapaxes(test_data[d:d+10,:,:,:],1,2)
            if set_tpu:
                with tpu_strategy.scope():
                    print('Using TPU')
                    predicted_data = self.model.predict(data)
            else:
                predicted_data = self.model.predict(data)

            predicted_data = np.swapaxes(self.model.predict(data),1,2)

            predicted_data[:,:,:,0] = np.roll(predicted_data[:,:,:,0],12, axis=1)
            predicted_data[:,0:12,:,0] = 0
            predicted_data[:,:,:,1] = np.roll(predicted_data[:,:,:,1],-12, axis=1)
            predicted_data[:,-12:-1,:,1] = 0

            y_data.append(predicted_data)

        y_data = np.concatenate(y_data)
        y_data = np.squeeze(np.array(y_data))
        y_data[y_data<=0] = 0
        return y_data






