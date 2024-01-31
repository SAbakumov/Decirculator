import tensorflow as tf
import  tensorflow.keras.layers as layers

def ConvBlock(x , num_kernel,kernel_size):
    x = layers.Conv2D(num_kernel,kernel_size,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.Conv2D(num_kernel,kernel_size,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    
    return x 


def UNet(input,num_kernels,kernel_size,output_size):
   
    x = tf.math.divide(input,1)
    output_layers = []
    for num_kernel in num_kernels:
        x = ConvBlock(x , num_kernel,kernel_size)
        output_layers = [x] + output_layers  
        x = layers.MaxPooling2D(pool_size=2)(x)


    x = ConvBlock(x , num_kernels[-1]*2,kernel_size)

    num_kernels.reverse()
    for id, num_kernel in enumerate(num_kernels):
        x = layers.Conv2DTranspose(num_kernel,kernel_size,strides = 2 ,padding='same',activation='elu')(x)
        x = layers.Concatenate()([output_layers[id],x])
        x = ConvBlock(x , num_kernel,kernel_size)

    
    output = layers.Conv2D(output_size[-1], 1, activation="relu", padding="same")(x)
    return output



def GetFullModel(input_shape,output_shape):

    input = layers.Input(shape = input_shape)
    model_head_1 = UNet(input,[64,128,256,512],(3,3),output_shape)
    output_head = tf.keras.Model(input,model_head_1)
    output_head.summary()
    return  output_head