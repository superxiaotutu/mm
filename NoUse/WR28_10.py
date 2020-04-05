import tensorflow as tf

N = (28-1)//3
k = 10

def WD28(x, is_training=False):
    with tf.variable_scope('before_split'):
        conv1 = _conv_bn_activation(
            bottom=x,
            filters=16,
            kernel_size=3,
            strides=1,
            is_training=is_training
        )
    with tf.variable_scope('split'):
        residual_block = conv1
        for i in range(N):
            residual_block = _residual_block(residual_block, 16 * k, 1, 'group_conv2/conv' + str(i + 1), is_training)
        for i in range(N):
            residual_block = _residual_block(residual_block, 32 * k, 2, 'group_conv3/conv' + str(i + 1), is_training)
        for i in range(N):
            residual_block = _residual_block(residual_block, 64 * k, 2, 'group_conv4/conv' + str(i + 1), is_training)
    with tf.variable_scope('after_spliting'):
        bn = _bn(residual_block, is_training)
        relu = tf.nn.relu(bn)
    with tf.variable_scope('group_avg_pool'):
        axes = [1, 2]
        global_pool = tf.reduce_mean(relu, axis=axes, keepdims=False, name='global_pool')
    #     final_dense = tf.layers.dense(global_pool, 10, name='final_dense')
    # with tf.variable_scope('optimizer'):
    #     logit = tf.nn.softmax(final_dense, name='logit')
    return global_pool

def _bn(bottom, is_training):
    bn = tf.layers.batch_normalization(
        inputs=bottom,
        training=is_training
    )
    return bn

def _conv_bn_activation(bottom, filters, kernel_size, strides, is_training, activation=tf.nn.relu):
    conv = tf.layers.conv2d(
        inputs=bottom,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
    )
    bn = _bn(conv, is_training)
    if activation is not None:
        return activation(bn)
    else:
        return bn

def _bn_activation_conv(bottom, filters, kernel_size, strides, is_training, activation=tf.nn.relu):
    bn = _bn(bottom, is_training)
    if activation is not None:
        bn = activation(bn)
    conv = tf.layers.conv2d(
        inputs=bn,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
    )
    return conv

def _residual_block(bottom, filters, strides, scope, is_training):
    with tf.variable_scope(scope):
        with tf.variable_scope('conv_branch'):
            conv = _bn_activation_conv(bottom, filters, 3, strides, is_training)
            # dropout = _dropout(conv, 0.3, 'dropout', is_training)
            dropout = conv
            conv = _bn_activation_conv(dropout, filters, 3, 1, is_training)
        with tf.variable_scope('identity_branch'):
            if strides != 1:
                shutcut = _bn_activation_conv(bottom, filters, 1, strides, is_training)
            else:
                index = 3
                if tf.shape(bottom)[index] != filters:
                    shutcut = _bn_activation_conv(bottom, filters, 1, strides, is_training)
                else:
                    shutcut = bottom

        return conv + shutcut

def _max_pooling(bottom, pool_size, strides, name):
    return tf.layers.max_pooling2d(
        inputs=bottom,
        pool_size=pool_size,
        strides=strides,
        padding='same',
        name=name
    )

def _avg_pooling(bottom, pool_size, strides, name):
    return tf.layers.average_pooling2d(
        inputs=bottom,
        pool_size=pool_size,
        strides=strides,
        padding='same',
        name=name
    )

def _dropout(bottom, prob, name, is_training):
    return tf.layers.dropout(
        inputs=bottom,
        rate=prob,
        training=is_training,
        name=name
    )
