"""
 " License:
 " -----------------------------------------------------------------------------
 " Copyright (c) 2017, Gabriel Eilertsen.
 " All rights reserved.
 " 
 " Redistribution and use in source and binary forms, with or without 
 " modification, are permitted provided that the following conditions are met:
 " 
 " 1. Redistributions of source code must retain the above copyright notice, 
 "    this list of conditions and the following disclaimer.
 " 
 " 2. Redistributions in binary form must reproduce the above copyright notice,
 "    this list of conditions and the following disclaimer in the documentation
 "    and/or other materials provided with the distribution.
 " 
 " 3. Neither the name of the copyright holder nor the names of its contributors
 "    may be used to endorse or promote products derived from this software 
 "    without specific prior written permission.
 " 
 " THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 " AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 " IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 " ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 " LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 " CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 " SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 " INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 " CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 " ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 " POSSIBILITY OF SUCH DAMAGE.
 " -----------------------------------------------------------------------------
 "
 " Description: TensorFlow autoencoder CNN for HDR image reconstruction.
 " Author: Gabriel Eilertsen, gabriel.eilertsen@liu.se
 " Date: Aug 2017
"""


import tensorflow as tf
import tensorlayer as tl
import numpy as np

# The HDR reconstruction autoencoder fully convolutional neural network

def model(x, batch_size=1, is_training=False):

    # Encoder network (VGG16, until pool5)
    x_in = tf.scalar_mul(255.0, x)
    with tf.compat.v1.Session() as val:
      x_in_shape = val.run(tf.shape(x_in))
    print('x_in: \n',x_in_shape)
    #net_in = tl.layers.Input(x_in, name='input_layer')
    net_in = tl.layers.Input( x_in_shape, name='input_layer')
    conv_layers, skip_layers = encoder(net_in)

    # Fully convolutional layers on top of VGG16 conv layers
    #network = tl.layers.Conv2dLayer(conv_layers,
    #                act = tf.identity,
    #                shape = [3, 3, 512, 512],
    #                strides = [1, 1, 1, 1],
    #                padding='SAME',
    #                name ='encoder/h6/conv')
    network = tl.layers.Conv2d(n_filter = 512,
                    filter_size  = (3, 3),
                    strides = (1, 1),
                    act = tf.identity,
                    padding = 'SAME',
                    in_channels = 512,
                    name = 'encoder/h6/conv')(conv_layers)
    #network = tl.layers.BatchNormLayer(network, is_train=is_training, name='encoder/h6/batch_norm')
    network = tl.layers.BatchNorm(is_train=is_training, name='encoder/h6/batch_norm')(network)
    
    #network.outputs = tf.nn.relu(network.outputs, name='encoder/h6/relu')
    network = tf.nn.relu(network, name='encoder/h6/relu')
    # Decoder network
    network = decoder(network, skip_layers, batch_size, is_training)

    if is_training:
        return network, conv_layers

    return network


# Final prediction of the model, including blending with input
def get_final(network, x_in):
    sb, sy, sx, sf = x_in.get_shape().as_list()
    y_predict = network.outputs

    # Highlight mask
    thr = 0.05
    alpha = tf.reduce_max(x_in, reduction_indices=[3])
    alpha = tf.minimum(1.0, tf.maximum(0.0, alpha-1.0+thr)/thr)
    alpha = tf.reshape(alpha, [-1, sy, sx, 1])
    alpha = tf.tile(alpha, [1,1,1,3])

    # Linearied input and prediction
    x_lin = tf.pow(x_in, 2.0)
    y_predict = tf.exp(y_predict)-1.0/255.0

    # Alpha blending
    y_final = (1-alpha)*x_lin + alpha*y_predict
    
    return y_final


# Convolutional layers of the VGG16 model used as encoder network
#@tf.function
def encoder(input_layer):

    VGG_MEAN = [103.939, 116.779, 123.68]

    # Convert RGB to BGR
    print('encoder input_layer: \n',input_layer)
    red, green, blue = tf.split(input_layer, axis=3, num_or_size_splits= 3)
    bgr = tf.concat([ blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2] ], axis=3)
    with tf.compat.v1.Session() as val:
      bgr_shape = val.run(tf.shape(bgr))
    #network = tl.layers.Input(bgr, name='encoder/input_layer_bgr')
    network = tl.layers.Input(bgr_shape, name='encoder/input_layer_bgr')
    # Convolutional layers size 1
    network     = conv_layer(network, [3, 64], 'encoder/h1/conv_1')
    beforepool1 = conv_layer(network, [64, 64], 'encoder/h1/conv_2')
    network     = pool_layer(beforepool1, 'encoder/h1/pool')

    # Convolutional layers size 2
    network     = conv_layer(network, [64, 128], 'encoder/h2/conv_1')
    beforepool2 = conv_layer(network, [128, 128], 'encoder/h2/conv_2')
    network     = pool_layer(beforepool2, 'encoder/h2/pool')

    # Convolutional layers size 3
    network     = conv_layer(network, [128, 256], 'encoder/h3/conv_1')
    network     = conv_layer(network, [256, 256], 'encoder/h3/conv_2')
    beforepool3 = conv_layer(network, [256, 256], 'encoder/h3/conv_3')
    network     = pool_layer(beforepool3, 'encoder/h3/pool')

    # Convolutional layers size 4
    network     = conv_layer(network, [256, 512], 'encoder/h4/conv_1')
    network     = conv_layer(network, [512, 512], 'encoder/h4/conv_2')
    beforepool4 = conv_layer(network, [512, 512], 'encoder/h4/conv_3')
    network     = pool_layer(beforepool4, 'encoder/h4/pool')

    # Convolutional layers size 5
    network     = conv_layer(network, [512, 512], 'encoder/h5/conv_1')
    network     = conv_layer(network, [512, 512], 'encoder/h5/conv_2')
    beforepool5 = conv_layer(network, [512, 512], 'encoder/h5/conv_3')
    network     = pool_layer(beforepool5, 'encoder/h5/pool')

    return network, (input_layer, beforepool1, beforepool2, beforepool3, beforepool4, beforepool5)


# Decoder network
#@tf.function
def decoder(input_layer, skip_layers, batch_size=1, is_training=False):
    sb, sx, sy, sf = input_layer.get_shape().as_list()
    alpha = 0.0

    # Upsampling 1
    network = deconv_layer(input_layer, (batch_size,sx,sy,sf,sf), 'decoder/h1/decon2d', alpha, is_training)

    # Upsampling 2
    network = skip_connection_layer(network, skip_layers[5], 'decoder/h2/fuse_skip_connection', is_training = False)
    network = deconv_layer(network, (batch_size,2*sx,2*sy,sf,sf), 'decoder/h2/decon2d', alpha, is_training)

    # Upsampling 3
    network = skip_connection_layer(network, skip_layers[4], 'decoder/h3/fuse_skip_connection', is_training)
    network = deconv_layer(network, (batch_size,4*sx,4*sy,sf,sf/2), 'decoder/h3/decon2d', alpha, is_training)

    # Upsampling 4
    network = skip_connection_layer(network, skip_layers[3], 'decoder/h4/fuse_skip_connection', is_training)
    network = deconv_layer(network, (batch_size,8*sx,8*sy,sf/2,sf/4), 'decoder/h4/decon2d', alpha, is_training)

    # Upsampling 5
    network = skip_connection_layer(network, skip_layers[2], 'decoder/h5/fuse_skip_connection', is_training)
    network = deconv_layer(network, (batch_size,16*sx,16*sy,sf/4,sf/8), 'decoder/h5/decon2d', alpha, is_training)

    # Skip-connection at full size
    network = skip_connection_layer(network, skip_layers[1], 'decoder/h6/fuse_skip_connection', is_training)

    # Final convolution
    network = tl.layers.Conv2dLayer(network,
                        act = tf.identity,
                        shape = [1, 1, int(sf/8), 3],
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        W_init = tf.contrib.layers.xavier_initializer(uniform=False),
                        b_init = tf.constant_initializer(value=0.0),
                        name ='decoder/h7/conv2d')

    # Final skip-connection
    network = tl.layers.BatchNormLayer(network, is_train=is_training, name='decoder/h7/batch_norm')
    network.outputs = tf.maximum(alpha*network.outputs, network.outputs, name='decoder/h7/leaky_relu')
    network = skip_connection_layer(network, skip_layers[0], 'decoder/h7/fuse_skip_connection')

    return network


# Load weights for VGG16 encoder convolutional layers
# Weights are from a .npy file generated with the caffe-tensorflow tool
def load_vgg_weights(network, weight_file, session):
    params = []

    if weight_file.lower().endswith('.npy'):
        npy = np.load(weight_file, encoding='latin1')
        for key, val in sorted(npy.item().items()):
            if(key[:4] == "conv"):
                print("  Loading %s" % (key))
                print("  weights with size %s " % str(val['weights'].shape))
                print("  and biases with size %s " % str(val['biases'].shape))
                params.append(val['weights'])
                params.append(val['biases'])
    else:
        print('No weights in suitable .npy format found for path ', weight_file)

    print('Assigning loaded weights..')
    tl.files.assign_params(session, params, network)

    return network


# === Layers ==================================================================

# Convolutional layer
def conv_layer(input_layer, sz, str):
    #network = tl.layers.Conv2d(input_layer,
    #                act = tf.nn.relu,
    #                shape = [3, 3, sz[0], sz[1]],
    #                strides = [1, 1, 1, 1],
    #                padding = 'SAME',
    #                name = str)
    #print(sz[1]); print(sz[0])
    #print(str)
    #print(input_layer)
    network = tl.layers.Conv2d(n_filter = sz[1],
                    filter_size  = (3, 3),
                    strides = (1, 1),
                    act = tf.nn.relu,
                    padding = 'SAME',
                    in_channels = sz[0],
                    name = str)(input_layer)
    return network


# Max-pooling layer
def pool_layer(input_layer, str):
    #network = tl.layers.PoolLayer(input_layer,
    #                ksize=[1, 2, 2, 1],
    #                strides=[1, 2, 2, 1],
    #                padding='SAME',
    #                pool = tf.nn.max_pool,
    #                name = str)
    network = tl.layers.PoolLayer(filter_size=(1, 2, 2, 1),
                    strides=(1, 2, 2, 1),
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name = str)(input_layer)
    return network


# Concatenating fusion of skip-connections
def skip_connection_layer(input_layer, skip_layer, str, is_training):
    _, sx, sy, sf = input_layer.get_shape().as_list()
    _, sx_, sy_, sf_ = skip_layer.get_shape().as_list()
    
    assert (sx_,sy_,sf_) == (sx,sy,sf)

    # skip-connection domain transformation, from LDR encoder to log HDR decoder
    skip_layer = tf.math.log(tf.pow(tf.scalar_mul(1.0/255, skip_layer), 2.0)+1.0/255.0)

    # specify weights for fusion of concatenation, so that it performs an element-wise addition
    
    #weights = tf.zeros([sf+sf_, sf],tf.dtypes.float32)
    #output_list = []
    #tensor_shape = weights.get_shape(); print('tensor_shape',tensor_shape)

    #for i in range(sf):
    #    weights = weights[i, i].assign(1)
    #    weights = weights[i+sf_, i].assign(1)
    #print('weights', weights)
    weights = np.float32(np.zeros((sf+sf_, sf)))
    for i in range(sf):
        weights[i, i] = 1
        weights[i+sf_, i] = 1

    #add_init = tf.constant_initializer(value =weights)
    #b = tf.constant_initializer(value=0.0); print('b_init: \n', b)
    print('input_layer: \n',input_layer)
    print('skip layer: \n',skip_layer)
    print('weights: \n',weights)
    print('is training: \n',is_training)
    print('str: \n',str)
    
    #print('add_init:  \n',add_init)
    # concatenate layers
    network_concat = tf.concat([input_layer, skip_layer], axis=3, name='%s/skip_connection'%str)
    print('network_concat',network_concat)
    # fuse concatenated layers using the specified weights for initialization
    #network = tl.layers.Conv2dLayer(network,
    #                act = tf.identity,
    #                shape = [1, 1, sf+sf_, sf],
    #                strides = [1, 1, 1, 1],
    #                padding = 'SAME',
    #                W_init = add_init,
    #                b_init = tf.constant_initializer(value=0.0),
    #                name = str)
    
    network = tl.layers.Conv2d(n_filter = sf,
                    filter_size = (1, 1),
                    strides = (1, 1),
                    act = tf.identity,
                    padding = 'SAME',
                    W_init = tf.constant_initializer(value =weights),
                    b_init = tf.constant_initializer(value=0.0),
                    in_channels = sf+sf_,
                    name=str)(network_concat)
    return network

# Deconvolution layer
def deconv_layer(input_layer, sz, str, alpha, is_training=False):
    scale = 2
    print('sz: \n',sz)
    kernel_size = (2 * scale - scale % 2)
    num_in_channels = int(sz[3])
    num_out_channels = int(sz[4])

    # create bilinear weights in numpy array
    bilinear_kernel = np.zeros([kernel_size, kernel_size], dtype=np.float32)
    scale_factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(kernel_size):
        for y in range(kernel_size):
            bilinear_kernel[x,y] = (1 - abs(x - center) / scale_factor) * \
                                   (1 - abs(y - center) / scale_factor)
    weights = np.zeros((kernel_size, kernel_size, num_out_channels, num_in_channels))
    for i in range(num_out_channels):
        weights[:, :, i, i] = bilinear_kernel

    #init_matrix = tf.constant_initializer(value=weights, dtype=tf.float32)
    init_matrix = tf.constant_initializer(value=weights)
    #network = tl.layers.DeConv2dLayer(input_layer,
    #                            shape = [filter_size, filter_size, num_out_channels, num_in_channels],
    #                            output_shape = [sz[0], sz[1]*scale, sz[2]*scale, num_out_channels],
    #                            strides=[1, scale, scale, 1],
    #                            W_init=init_matrix,
    #                            padding='SAME',
    #                            act=tf.identity,
    #                            name=str)
    network = tl.layers.DeConv2d(n_filter = num_out_channels,
                                filter_size  = (kernel_size,kernel_size),
                                in_channels = num_in_channels,
                                strides=(scale, scale),
                                W_init=init_matrix,
                                padding='SAME',
                                act=tf.identity,
                                name=str)(input_layer)
    #network = tf.keras.layers.Conv2DTranspose(filters = num_out_channels,
    #                  kernel_size = (filter_size,filter_size),
    #                  strides = (scale,scale),
    #                  padding='same',
    #                  activation= tf.identity,
    #                  kernel_initializer=init_matrix)(input_layer)
    network = tl.layers.BatchNorm2d(is_train=is_training, name='%s/batch_norm_dc'%str)(network)
    network = tf.maximum(alpha*network, network, name='%s/leaky_relu_dc'%str)

    return network
