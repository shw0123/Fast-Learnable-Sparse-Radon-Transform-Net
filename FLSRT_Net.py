# #
# # Name                    Version               Build           Channel
# tensorflow                2.0.0           gpu_py37h768510d_0    defaults
# tensorflow-base           2.0.0           gpu_py37h0ec5d1f_0    defaults
# tensorflow-estimator      2.0.0               pyh2649769        defaults
# tensorflow-gpu            2.0.0               h0d30ee6_0        defaults


import tensorflow as tf
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

NumofDataset = 40
NumofTrain = 40
dataset = list()
ma_dataset = list()
m_dataset = list()
model_save_num = 1

for name in ['ma', 'm']:
    for index in range(1, NumofDataset + 1):
        filename = '/new_task/datasyn/' + name + '/' + name + '_' + str(index) + '.mat'
        data = sio.loadmat(filename)[name]
        data = data[0:1496, 0:160]
        dataset.append(data)

dataset = np.array(dataset)
dataset_ = tf.convert_to_tensor(dataset)

ma_dataset = dataset_[0:NumofDataset, 0:1496, 0:160]
m_dataset = dataset_[NumofDataset:2 * NumofDataset, 0:1496, 0:160]

print(ma_dataset.shape)
print(m_dataset.shape)

train_data = tf.data.Dataset.from_tensor_slices(tf.cast(ma_dataset[0:NumofTrain], dtype=tf.float32))
test_data = tf.data.Dataset.from_tensor_slices(tf.cast(ma_dataset[NumofTrain:NumofDataset], dtype=tf.float32))
train_label = tf.data.Dataset.from_tensor_slices(tf.cast(m_dataset[0:NumofTrain], dtype=tf.float32))
test_label = tf.data.Dataset.from_tensor_slices(tf.cast(m_dataset[NumofTrain:NumofDataset], dtype=tf.float32))

train_dataset = tf.data.Dataset.zip((train_data, train_label))


class Sparse(Layer):
    """Sparse Activation Function.

  It follows:
  `f(x) = x - alpha for x > alpha > 0`,
  `f(x) = alpha - x for 0 > alpha > x`,
  `f(x) = 0 for abs(x) <= abs(alpha)`,
  where `alpha` is a learned parameter.

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as the input.

  Arguments:
    alpha_initializer: Initializer function for the weights.
    alpha_regularizer: Regularizer for the weights.
    alpha_constraint: Constraint for the weights.
    shared_axes: The axes along which to share learnable
      parameters for the activation function.
      For example, if the incoming feature maps
      are from a 2D convolution
      with output shape `(batch, height, width, channels)`,
      and you wish to share parameters across space
      so that each filter only has one set of parameters,
      set `shared_axes=[1, 2]`.
  """

    def __init__(self,
                 alpha_initializer='zeros',
                 alpha_regularizer=None,
                 alpha_constraint=None,
                 shared_axes=None,
                 **kwargs):
        super(Sparse, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
        self.alpha = self.add_weight(
            shape=param_shape,
            name='alpha',
            initializer=self.alpha_initializer,
            regularizer=self.alpha_regularizer,
            constraint=self.alpha_constraint)
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):

        pos = K.relu(inputs - self.alpha)
        neg = K.relu(-inputs - self.alpha)
        return pos - neg

    def get_config(self):
        config = {
            'alpha_initializer': initializers.serialize(self.alpha_initializer),
            'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint),
            'shared_axes': self.shared_axes
        }
        base_config = super(Sparse, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


class ConvNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # glorot_uniform
        # glorot_normal
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation='tanh',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            data_format='channels_last'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation='tanh',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            data_format='channels_last'
        )
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=[2, 2],
            data_format='channels_last'
            # strides=2
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            padding='same',
            activation='tanh',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            data_format='channels_last'
        )
        self.conv4 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            padding='same',
            activation='tanh',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            data_format='channels_last'
        )
        self.pool2 = tf.keras.layers.MaxPool2D(
            pool_size=[2, 2],
            data_format='channels_last'
            # strides=2
        )
        self.conv5 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=[3, 3],
            padding='same',
            activation='tanh',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            data_format='channels_last'
        )
        self.conv6 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=[3, 3],
            padding='same',
            activation='tanh',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            data_format='channels_last'
        )
        self.conv_re1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            padding='same',
            activation='tanh',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            data_format='channels_last'
        )
        self.sampling1 = tf.keras.layers.UpSampling2D(
            size=[2, 2],
            data_format='channels_last'
        )
        self.jump1 = tf.keras.layers.Concatenate(axis=-1)
        self.conv7 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            padding='same',
            activation='tanh',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            data_format='channels_last'
        )
        self.conv8 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            padding='same',
            activation='tanh',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            data_format='channels_last'
        )
        self.conv_re2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation='tanh',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            data_format='channels_last'
        )
        self.sampling2 = tf.keras.layers.UpSampling2D(
            size=[2, 2],
            data_format='channels_last'
        )
        self.jump2 = tf.keras.layers.Concatenate(axis=-1)
        self.conv9 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation='tanh',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            data_format='channels_last'
        )
        self.conv10 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation='tanh',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            data_format='channels_last'
        )
        self.conv11 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding='same',
            activation='tanh',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            data_format='channels_last'
        )
        self.conv12 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=[3, 3],
            padding='same',
            activation='linear',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            data_format='channels_last'
        )

        self.S = Sparse()

    def call(self, inputs):
        result1 = self.conv1(inputs)
        result_conv2 = self.conv2(result1)

        result2 = self.pool1(result_conv2)

        result3 = self.conv3(result2)
        result_conv4 = self.conv4(result3)

        result4 = self.pool2(result_conv4)

        result5 = self.conv5(result4)
        result6 = self.conv6(result5)
        result7 = self.conv_re1(result6)

        result8 = self.sampling1(result7)

        result9 = self.conv7(result8 + result_conv4)
        result10 = self.conv8(result9)
        result11 = self.conv_re2(result10)

        result12 = self.sampling2(result11)

        result13 = self.conv9(result12 + result_conv2)
        result14 = self.conv10(result13)
        result15 = self.conv11(result14)
        result16 = self.conv12(result15)

        result = self.S(result16)

        return [result, result_conv2, result_conv4, result7, result11, result13, result16]


boundaries = list()
boundaries_start = 10 * 200  # 16000
values = list()
lr_start = 0.0001
for i in range(0, 4):
    values.append(lr_start)
    lr_start = lr_start * 0.7
for i in range(0, 3):
    boundaries.append(boundaries_start)
    boundaries_start = boundaries_start + 10 * 200

piece_wise_constant_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=boundaries, values=values, name=None)

model = ConvNet()
optimizer = tf.keras.optimizers.Adam(piece_wise_constant_decay)
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
log_dir = './tensorboard_chaoshan/' + 'temp' + TIMESTAMP
summary_writer = tf.summary.create_file_writer(log_dir)

epoch = 1010
batch_size = 1
shape = tf.TensorSpec(shape=(batch_size, 1496, 160, 1), dtype=tf.dtypes.float32, name=None)
model._set_inputs(shape)

loss_list = np.array([])

for index in range(epoch):

    print("epoch %d" % (index + 1))
    shuffle_train_dataset = train_dataset.shuffle(NumofTrain)
    shuffle_train_dataset_batch = shuffle_train_dataset.batch(batch_size)
    res = list(shuffle_train_dataset_batch)

    for i in range(int(np.ceil(NumofTrain / batch_size))):
        X = tf.reshape(res[i][0], [batch_size, 1496, 160, 1])
        y = tf.reshape(res[i][1], [batch_size, 1496, 160, 1])

        with tf.GradientTape() as tape:
            output = model(X)
            y_pred = output[0]

            loss = tf.reduce_mean(tf.square(y_pred - y))
            loss_list = np.append(loss_list, loss.numpy())
            print("batch %d: loss %f" % (i + 1, 1 * loss.numpy()))

            with summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=(i + 1) * batch_size + index * NumofTrain)

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    model_file = './model_chaoshan/model/my_model_' + str(model_save_num)
    if not os.path.exists(model_file):
        os.makedirs(model_file)
    model.save(model_file, save_format="tf")
    model_save_num += 1
