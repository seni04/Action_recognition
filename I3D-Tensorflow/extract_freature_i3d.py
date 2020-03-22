import sys
sys.path.append('/media/senilab/DATA/Master/I3D-Tensorflow/')
import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_test
import math
import numpy as np
# from i3dCopy1 import InceptionI3d
from i3d import InceptionI3d
from utils import *
import pandas as pd
import cv2
from PIL import Image
import hickle

gpu_num = 1
'/home/senilab/Documents/I3D/models/'
batch_size = 1
num_frame_per_clib = 16
crop_size = 224
rgb_channels = 3
classics = 51
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
outfile = '/media/senilab/DATA2/feature_hmdb_i3d_5c/'

#pre_model_save_dir = "/home/senilab/Documents/I3D/rgb_imagenet_10000_101_4_64_0.0001_decay"
rgb_pre_model_save_dir = "/home/senilab/Documents/I3D/models/rgb_imagenet_30000_51_4_64_0.0001_decay/"
flow_pre_model_save_dir = "/home/senilab/Documents/I3D/models/flow_imagenet_30000_51_4_64_0.0001_decay/"
test_list_file = '/media/senilab/DATA/Master/I3D-Tensorflow/list/hmdb_list/test.list'
file = list(open(test_list_file, 'r'))
num_test_videos = len(file)
print("Number of test videos={}".format(num_test_videos))
with tf.Graph().as_default():
    rgb_images_placeholder, flow_images_placeholder, labels_placeholder, is_training = placeholder_inputs(
                    batch_size * gpu_num,
                    num_frame_per_clib,
                    crop_size,
                    rgb_channels
                    )

    with tf.variable_scope('RGB'):
        rgb_logit, _ = InceptionI3d(
                            num_classes=classics,
                            spatial_squeeze=True,
                            final_endpoint='Logits',
                            name='inception_i3d'
                            )(rgb_images_placeholder, is_training)
    with tf.variable_scope('Flow'):
        flow_logit, _ = InceptionI3d(
                            num_classes=classics,
                            spatial_squeeze=True,
                            final_endpoint='Logits',
                            name='inception_i3d'
                            )(flow_images_placeholder, is_training)
    norm_score = tf.nn.softmax(tf.add(rgb_logit, flow_logit))

    # Create a saver for writing training checkpoints.
    rgb_variable_map = {}
    flow_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB' and 'Adam' not in variable.name.split('/')[-1] :
            rgb_variable_map[variable.name.replace(':0', '')] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'Flow'and 'Adam' not in variable.name.split('/')[-1] :
            flow_variable_map[variable.name.replace(':0', '')] = variable
    flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(
                    config=config
                    )
    sess.run(init)

# load pre_train models
ckpt = tf.train.get_checkpoint_state(rgb_pre_model_save_dir)
if ckpt and ckpt.model_checkpoint_path:
    print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
    rgb_saver.restore(sess, ckpt.model_checkpoint_path)
    print("load complete!")
ckpt = tf.train.get_checkpoint_state(flow_pre_model_save_dir)
if ckpt and ckpt.model_checkpoint_path:
    print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
    flow_saver.restore(sess, ckpt.model_checkpoint_path)
    print("load complete!")

#Flow

pre_model_save_dir = "/home/senilab/Documents/I3D/models/flow_imagenet_30000_51_4_64_0.0001_decay/"
test_list_file = '/media/senilab/DATA/Master/I3D-Tensorflow/list/hmdb_list/test.list'
file = list(open(test_list_file, 'r'))
num_test_videos = len(file)
print("Number of test videos={}".format(num_test_videos))
with tf.Graph().as_default():
    rgb_images_placeholder, flow_images_placeholder, labels_placeholder, is_training = placeholder_inputs(
                    batch_size * gpu_num,
                    num_frame_per_clib,
                    crop_size,
                    rgb_channels
                    )

    with tf.variable_scope('Flow'):
        logit, _ = InceptionI3d(
                            num_classes=classics,
                            spatial_squeeze=True,
                            final_endpoint='Mixed_5c',
                            name='inception_i3d'
                            )(flow_images_placeholder, is_training)
    norm_score = tf.nn.softmax(logit)

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(
                    config=config
                    )
    sess.run(init)

ckpt = tf.train.get_checkpoint_state(pre_model_save_dir)
if ckpt and ckpt.model_checkpoint_path:
    print ("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    print ("load complete!")

all_steps = num_test_videos
top1_list = []
for step in xrange(all_steps):
    start_time = time.time()
    s_index = 0
    predicts = []
    data = {}
    top1 = False
    data_new = np.array([])
    print ('step',step)
    while True:
        _, val_images, val_labels, s_index, is_end = input_test.read_clip_and_label(
                        filename=file[step],
                        batch_size=batch_size * gpu_num,
                        s_index=s_index,
                        num_frames_per_clip=num_frame_per_clib,
                        crop_size=crop_size,
                        )

        data_features = sess.run(logit,
                           feed_dict={
                                        flow_images_placeholder: val_images,
                                        labels_placeholder: val_labels,
                                        is_training: False
                                        })

        temp = np.mean(data_features, axis=2)
        tmp = np.mean(temp, axis=2)
        tmp = np.squeeze(tmp)

        if data_new.shape[0] == 0:
            data_new = tmp
        else:
            data_new = np.concatenate([data_new,tmp])

        labell = np.zeros((1,51))
        labell[0,int(val_labels)] = 1

        #print(labell)

        if is_end:
            ndata = int(data_new.shape[0])
            nn = 16
            if ndata <=nn:
                data_new = np.concatenate([data_new,data_new],  axis=0)
                data_new = np.concatenate([data_new,data_new],  axis=0)
                ndata = int(data_new.shape[0])

            sampling = ndata/nn
            ndata2 = data_new[0:int(nn)*int(sampling),:]
            ndata3 = ndata2[0::int(sampling),:]

            data['feat'] = ndata3
            data['label'] = labell


            temp = file[step].split("/")
            namefile = temp[6].split(" ")
            print(namefile[0])


            datasave = outfile + 'test_flow/' + namefile[0]
            print(datasave)
#             np.save(datasave, data)
            hickle.dump(data, datasave)
            break
