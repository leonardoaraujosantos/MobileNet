import tensorflow as tf
import model_util as util


class CAE_AutoEncoderFE_MaxPool_MobileNet(object):
    def __init__(self, input=None, use_placeholder=True, training_mode=True, img_size = 100, multiplier=1):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='IMAGE_IN')
        self.__label = tf.placeholder(tf.int32, shape=[None, img_size, img_size, 1], name='LABEL_IN')
        self.__use_placeholder = use_placeholder

        ##### ENCODER
        # Calculating the convolution output:
        # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
        # H_out = 1 + (H_in+(2*pad)-K)/S
        # W_out = 1 + (W_in+(2*pad)-K)/S
        # CONV1: Input 100x100x3 after CONV 3x3 P:0 S:1 H_out: 1 + (100-3)/1 = 98, W_out= 1 + (100-3)/1 = 98
        if use_placeholder:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(self.__x, 3, 3, 3, 16, 1, "conv1", viewWeights=True, do_summary=False)
        else:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(input, 3, 3, 3, 16, 1, "conv1", viewWeights=True, do_summary=False)

        self.__conv1_bn = util.batch_norm(self.__conv1, training_mode, name='bn_c1')
        self.__conv1_act = util.relu(self.__conv1_bn, do_summary=False)

        # CONV2: Input 98x98x16 after CONV 3x3 P:0 S:2 H_out: 1 + (98-3)/2 = 48, W_out= 1 + (98-3)/2 = 48
        #self.__conv2 = util.conv2d(self.__conv1_act, 3, 3, 16, 16, 1, "conv2", do_summary=False)
        #self.__conv2_bn = util.batch_norm(self.__conv2, training_mode, name='bn_c2')
        #self.__conv2_act = util.relu(self.__conv2_bn, do_summary=False)
        self.__conv2_act = util.conv2d_separable(self.__conv1_act, 3, 3, 16, 16, 2, training_mode, "conv2",
                                                 do_summary=False, multiplier=multiplier)

        # Add Maxpool
        #self.__conv2_mp_act = util.max_pool(self.__conv2_act, 2,2,2,name="maxpool1")

        # CONV3: Input 48x48x16 after CONV 3x3 P:0 S:1 H_out: 1 + (48-3)/1 = 46, W_out= 1 + (48-3)/1 = 46
        #self.__conv3 = util.conv2d(self.__conv2_mp_act, 3, 3, 16, 32, 1, "conv3", do_summary=False)
        #self.__conv3_bn = util.batch_norm(self.__conv3, training_mode, name='bn_c3')
        #self.__conv3_act = util.relu(self.__conv3_bn, do_summary=False)
        self.__conv3_act = util.conv2d_separable(self.__conv2_act, 3, 3, 16, 32, 1, training_mode, "conv3",
                                                 do_summary=False, multiplier=multiplier)

        # CONV4: Input 46x46x32 after CONV 3x3 P:0 S:2 H_out: 1 + (46-3)/2 = 22, W_out= 1 + (46-3)/2 = 22
        #self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 32, 64, 1, "conv4", do_summary=False)
        #self.__conv4_bn = util.batch_norm(self.__conv4, training_mode, name='bn_c4')
        #self.__conv4_act = util.relu(self.__conv4_bn, do_summary=False)
        self.__conv4_act = util.conv2d_separable(self.__conv3_act, 3, 3, 32, 64, 2, training_mode, "conv4",
                                                 do_summary=False, multiplier=multiplier)

        # Add Maxpool
        #self.__conv4_mp_act = util.max_pool(self.__conv4_act, 2, 2, 2, name="maxpool2")

        # CONV5: Input 22x22x64 after CONV 3x3 P:0 S:1 H_out: 1 + (22-3)/1 = 20, W_out=  1 + (22-3)/1 = 20
        #self.__conv5 = util.conv2d(self.__conv4_mp_act, 3, 3, 64, 64, 1, "conv5", do_summary=False)
        #self.__conv5_bn = util.batch_norm(self.__conv5, training_mode, name='bn_c5')
        #self.__conv5_act = util.relu(self.__conv5_bn, do_summary=False)
        self.__conv5_act = util.conv2d_separable(self.__conv4_act, 3, 3, 64, 64, 1, training_mode, "conv5",
                                                 do_summary=False, multiplier=multiplier)

        ##### DECODER (At this point we have 1x18x64
        # Kernel, output size, in_volume, out_volume, stride
        self.__conv_t5_out = util.conv2d_transpose(self.__conv5_act, (3, 3), (22, 22), 64, 64, 1, name="dconv1",
                                                   do_summary=False)
        self.__conv_t5_out_bn = util.batch_norm(self.__conv_t5_out, training_mode, name='bn_t_c5')
        self.__conv_t5_out_act = util.relu(self.__conv_t5_out_bn, do_summary=False)

        self.__conv_t4_out = util.conv2d_transpose(
            self.__conv_t5_out_act,
            (3, 3), (46, 46), 64, 32, 2, name="dconv2", do_summary=False)
        self.__conv_t4_out_bn = util.batch_norm(self.__conv_t4_out, training_mode, name='bn_t_c4')
        self.__conv_t4_out_act = util.relu(self.__conv_t4_out_bn, do_summary=False)

        self.__conv_t3_out = util.conv2d_transpose(
            self.__conv_t4_out_act,
            (3, 3), (48, 48), 32, 16, 1, name="dconv3", do_summary=False)
        self.__conv_t3_out_bn = util.batch_norm(self.__conv_t3_out, training_mode, name='bn_t_c3')
        self.__conv_t3_out_act = util.relu(self.__conv_t3_out_bn, do_summary=False)

        self.__conv_t2_out = util.conv2d_transpose(
            self.__conv_t3_out_act,
            (3, 3), (98, 98), 16, 16, 2, name="dconv4", do_summary=False)
        self.__conv_t2_out_bn = util.batch_norm(self.__conv_t2_out, training_mode, name='bn_t_c2')
        self.__conv_t2_out_act = util.relu(self.__conv_t2_out_bn, do_summary=False)

        # Observe that the last deconv depth is the same as the number of classes
        self.__conv_t1_out = util.conv2d_transpose(
            self.__conv_t2_out_act,
            (3, 3), (img_size, img_size), 16, 3, 1, name="dconv5", do_summary=False)
        self.__conv_t1_out_bn = util.batch_norm(self.__conv_t1_out, training_mode, name='bn_t_c1')

        # Model output (It's not the segmentation yet...)
        self.__y = util.relu(self.__conv_t1_out_bn, do_summary=False)

        # Calculate flat tensor for Binary Cross entropy loss
        self.__y_flat = tf.reshape(self.__y, [tf.shape(self.__x)[0], img_size * img_size * 3])
        self.__x_flat = tf.reshape(self.__x, [tf.shape(self.__x)[0], img_size * img_size * 3])



    @property
    def output(self):
        return self.__y


    @property
    def input(self):
        if self.__use_placeholder:
            return self.__x
        else:
            return None

    @property
    def label_in(self):
        if self.__use_placeholder:
            return self.__label
        else:
            return None


    @property
    def output_flat(self):
        return self.__y_flat

    @property
    def input_flat(self):
        return self.__x_flat


    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act
