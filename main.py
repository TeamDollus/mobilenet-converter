import subprocess
import os
import wget

import tensorflow as tf

from collections import namedtuple
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, BatchNormalization, Input
from lightspeeur.layers import DepthwiseConv2D, Conv2D, ReLU

DepthwiseConvBlock = namedtuple("DepthwiseConvBlock", ["name", "stride", "in_channels", "out_channels"])
MOBILENET_DEPTHWISE_CONV_LAYERS = [
    DepthwiseConvBlock(name="conv2_1", stride=1, in_channels=32, out_channels=64),
    DepthwiseConvBlock(name="conv3_1", stride=2, in_channels=64, out_channels=128),
    DepthwiseConvBlock(name="conv3_2", stride=1, in_channels=128, out_channels=128),
    DepthwiseConvBlock(name="conv4_1", stride=2, in_channels=128, out_channels=256),
    DepthwiseConvBlock(name="conv4_2", stride=1, in_channels=256, out_channels=256),
    DepthwiseConvBlock(name="conv5_1", stride=2, in_channels=256, out_channels=512),
    DepthwiseConvBlock(name="conv5_2", stride=1, in_channels=512, out_channels=512),
    DepthwiseConvBlock(name="conv5_3", stride=1, in_channels=512, out_channels=512),
    DepthwiseConvBlock(name="conv5_4", stride=1, in_channels=512, out_channels=512),
    DepthwiseConvBlock(name="conv5_5", stride=1, in_channels=512, out_channels=512),
    DepthwiseConvBlock(name="conv5_6", stride=1, in_channels=512, out_channels=512),
    DepthwiseConvBlock(name="conv6_1", stride=2, in_channels=512, out_channels=1024),
    DepthwiseConvBlock(name="conv6_2", stride=1, in_channels=1024, out_channels=1024),
]

CHIP_ID = '2803'
CHECKPOINT_DIR = 'checkpoints'
PRETRAINED_CHECKPOINT = 'mobilenet_v1_1.0_224'
PRETRAINED_CHECKPOINT_NAME = 'MobilenetV1'
PRETRAINED_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, '{}.ckpt'.format(PRETRAINED_CHECKPOINT))
RESULT_MODEL_NAME = '{}_lightspeeur.h5'.format(PRETRAINED_CHECKPOINT)
RESULT_FOLDED_MODEL_NAME = '{}_lightspeeur_folded.h5'.format(PRETRAINED_CHECKPOINT)
CONVOLUTION_MAP = {
    'Conv2d_0': 'conv1_1',
    'Conv2d_1_depthwise': 'conv2_1_dw',
    'Conv2d_1_pointwise': 'conv2_1_pw',
    'Conv2d_2_depthwise': 'conv3_1_dw',
    'Conv2d_2_pointwise': 'conv3_1_pw',
    'Conv2d_3_depthwise': 'conv3_2_dw',
    'Conv2d_3_pointwise': 'conv3_2_pw',
    'Conv2d_4_depthwise': 'conv4_1_dw',
    'Conv2d_4_pointwise': 'conv4_1_pw',
    'Conv2d_5_depthwise': 'conv4_2_dw',
    'Conv2d_5_pointwise': 'conv4_2_pw',
    'Conv2d_6_depthwise': 'conv5_1_dw',
    'Conv2d_6_pointwise': 'conv5_1_pw',
    'Conv2d_7_depthwise': 'conv5_2_dw',
    'Conv2d_7_pointwise': 'conv5_2_pw',
    'Conv2d_8_depthwise': 'conv5_3_dw',
    'Conv2d_8_pointwise': 'conv5_3_pw',
    'Conv2d_9_depthwise': 'conv5_4_dw',
    'Conv2d_9_pointwise': 'conv5_4_pw',
    'Conv2d_10_depthwise': 'conv5_5_dw',
    'Conv2d_10_pointwise': 'conv5_5_pw',
    'Conv2d_11_depthwise': 'conv5_6_dw',
    'Conv2d_11_pointwise': 'conv5_6_pw',
    'Conv2d_12_depthwise': 'conv6_1_dw',
    'Conv2d_12_pointwise': 'conv6_1_pw',
    'Conv2d_13_depthwise': 'conv6_2_dw',
    'Conv2d_13_pointwise': 'conv6_2_pw'
}

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print('Downloading MobileNetV1 pre-trained checkpoint from TensorFlow storage...')
filename = wget.download('http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/{}.tgz'
                         .format(PRETRAINED_CHECKPOINT))
subprocess.run(['tar', '-xvf', filename, '-C', CHECKPOINT_DIR])


def conv_block(inputs, units, kernel_size=(1, 1), strides=(1, 1), folded_shape=False, name=None):
    outputs = Conv2D(units,
                     kernel_size,
                     CHIP_ID,
                     strides=strides,
                     bit_mask=2,
                     use_bias=folded_shape,
                     quantize=folded_shape,
                     name='{}/{}'.format(name, 'conv'))(inputs)
    if not folded_shape:
        outputs = BatchNormalization(name='{}/{}'.format(name, 'batch_norm'))(outputs)
    outputs = ReLU(CHIP_ID, name='{}/{}'.format(name, 'relu'))(outputs)
    return outputs


def depthwise_conv_block(inputs, strides=(1, 1), folded_shape=False, name=None):
    outputs = DepthwiseConv2D(CHIP_ID,
                              strides=strides,
                              bit_mask=2,
                              use_bias=folded_shape,
                              quantize=folded_shape,
                              name='{}/{}'.format(name, 'conv'))(inputs)
    if not folded_shape:
        outputs = BatchNormalization(name='{}/{}'.format(name, 'batch_norm'))(outputs)
    outputs = ReLU(CHIP_ID, name='{}/{}'.format(name, 'relu'))(outputs)
    return outputs


def build_mobilenet(folded_shape=False):
    inputs = Input(shape=(224, 224, 3))
    outputs = conv_block(inputs, 32, kernel_size=(3, 3), strides=(2, 2), folded_shape=folded_shape, name='conv1_1')
    for layer in MOBILENET_DEPTHWISE_CONV_LAYERS:
        outputs = depthwise_conv_block(outputs, (layer.stride, layer.stride), folded_shape=folded_shape, name='{}_dw'.format(layer.name))
        outputs = conv_block(outputs, layer.out_channels, folded_shape=folded_shape, name='{}_pw'.format(layer.name))
    outputs = GlobalAveragePooling2D(name='global_average_pooling')(outputs)
    outputs = Flatten(name='flatten')(outputs)
    outputs = Dense(1024, activation='relu', name='fc')(outputs)
    outputs = Dense(5, activation='softmax', name='classes')(outputs)

    return Model(name='MobileNetV1', inputs=inputs, outputs=outputs)


print('Organizing lightspeeur MobileNetV1 model...')
initial_model = build_mobilenet()
folded_model = build_mobilenet(folded_shape=True)

print('Converting pre-trained TensorFlow checkpoint to Lightspeeur format...')


def load_weights(model):
    reader = tf.compat.v1.train.NewCheckpointReader(PRETRAINED_CHECKPOINT_PATH)
    for src_layer_name, dst_layer_name in CONVOLUTION_MAP.items():
        batch_norm_scope = '{}/{}/{}'.format(PRETRAINED_CHECKPOINT_NAME, src_layer_name, 'BatchNorm')
        if 'depthwise' in src_layer_name:
            conv_weights = reader.get_tensor('{}/{}/{}'.format(PRETRAINED_CHECKPOINT_NAME, src_layer_name, 'depthwise_weights'))
        else:
            conv_weights = reader.get_tensor('{}/{}/{}'.format(PRETRAINED_CHECKPOINT_NAME, src_layer_name, 'weights'))

        batch_moving_mean = reader.get_tensor('{}/{}'.format(batch_norm_scope, 'moving_mean'))
        batch_moving_variance = reader.get_tensor('{}/{}'.format(batch_norm_scope, 'moving_variance'))
        batch_gamma = reader.get_tensor('{}/{}'.format(batch_norm_scope, 'gamma'))
        batch_beta = reader.get_tensor('{}/{}'.format(batch_norm_scope, 'beta'))

        conv_layer = model.get_layer('{}/{}'.format(dst_layer_name, 'conv'))

        if conv_layer.use_bias:
            conv_bias = conv_layer.get_weights()[1]
            conv_layer.set_weights([conv_weights, conv_bias])
        else:
            conv_layer.set_weights([conv_weights])

        batch_weights = [batch_gamma, batch_beta, batch_moving_mean, batch_moving_variance]
        batch_layer = model.get_layer('{}/{}'.format(dst_layer_name, 'batch_norm'))
        batch_layer.set_weights(batch_weights)


load_weights(initial_model)
load_weights(folded_model)

initial_model.save(RESULT_MODEL_NAME)
folded_model.save(RESULT_FOLDED_MODEL_NAME)
print('All jobs have finished')
