import os,sys
import numpy as np
import cv2
from PIL import Image
from multiprocessing import Pool
import argparse
from IPython import embed #to debug
import scipy.misc
import tensorflow as tf
import numpy as np
import hickle
from os import listdir 
from os.path import isfile, join, isdir
import random
import time

def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.
    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    with tf.variable_scope(scope, reuse=True):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)

def multihead_attention(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=16, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
    

    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        Q = tf.contrib.layers.fully_connected(queries, num_units) # (N, T_q, C)
        K = tf.contrib.layers.fully_connected(queries, num_units ) # (N, T_k, C)
        V = tf.contrib.layers.fully_connected(keys, num_units) # (N, T_k, C)
        
        Q1 = tf.reshape(Q,(Q.get_shape().as_list()[0],Q.get_shape().as_list()[1],num_units))
        K1 = tf.reshape(K,(Q.get_shape().as_list()[0],Q.get_shape().as_list()[1],num_units))        
        V1 = tf.reshape(V,(Q.get_shape().as_list()[0],Q.get_shape().as_list()[1],num_units))
        
        
        # Split abboxfnd concat
        Q_ = tf.concat(tf.split(Q1, num_heads, axis = 2),axis =0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K1, num_heads, axis = 2),axis =0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V1, num_heads, axis = 2),axis =0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

                
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
       # 
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
          
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
  # 
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
        matt    = outputs
        
        
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
          
        # Dropouts
        outputs = tf.contrib.layers.dropout(outputs, keep_prob=dropout_rate, is_training=tf.convert_to_tensor(is_training))
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads,axis = 0),axis =2 ) # (N, T_q, C)
              
        # Residual connection
        outputs += queries              
              
        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)
 
    return outputs, matt

def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def get_data(data_batch):
    simpan_matt = np.array([])
    save_label = np.array([])

    data_file = test_dir_flow + data_batch
    save_file = save_dir_test_flow + data_batch
    data = hickle.load(data_file)
    n_frames = 16
    n_hidden = 1024

    group1 = np.array([data['feat']])
    data_label = data['label']

    if save_label.shape[0] == 0:
        save_label = data_label
    else:
        save_label = np.concatenate([save_label, data_label])

    # position_idx  = tf.ones((n_frames,1*n_hidden))
    # positional_encode_tmp = positional_encoding(tf.expand_dims(position_idx, axis=0), n_frames) 
    # group1 = group1 + positional_encode_tmp

    # forward
    multi_g1, matt1  = multihead_attention(queries=group1, keys=group1, num_units=n_hidden, num_heads=16,dropout_rate=0.9,is_training=1,causality=True, scope = 'forward', reuse = tf.AUTO_REUSE)
    # backward
    multi_g2, matt2  = multihead_attention(queries=group1[:,::-1,:] , keys=group1[:,::-1,:] , num_units=n_hidden, num_heads=16,dropout_rate=0.9,is_training=1,causality=True, scope = 'backward', reuse = tf.AUTO_REUSE)
    # combine (bebas bisa diconcatenate ato dijumlah aja)
    multi_g = multi_g1 + multi_g2[:,::-1,:]

    init_op = tf.initialize_all_variables()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init_op)
    feature_attention = sess.run(multi_g)

    feature = {}
    feature['feat'] = feature_attention
    feature['label'] = data_label
    hickle.dump(feature, save_file)
    print(save_file)
    


def get_video_list():
    video_list=[]
    for cls_names in os.listdir(videos_root):
        cls_path=os.path.join(videos_root,cls_names)
        for video_ in sorted([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]):
            video_full = os.path.join(cls_path,video_)
            video_list.append(video_full)
    video_list.sort()
    return video_list,len(video_list)



def parse_args():
    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument('--dataset',default='hmdb51',type=str,help='set the dataset name, to find the data path')
    parser.add_argument('--data_root',default='/media/senilab/DATA/Master/pytorch-coviar/data',type=str)
    parser.add_argument('--new_dir',default='flows',type=str)
    parser.add_argument('--num_workers',default=10,type=int,help='num of workers to act multi-process')
    parser.add_argument('--step',default=1,type=int,help='gap frames')
    parser.add_argument('--bound',default=15,type=int,help='set the maximum of optical flow')
    parser.add_argument('--s_',default=0,type=int,help='start id')
    parser.add_argument('--e_',default=6849,type=int,help='end id')
    parser.add_argument('--mode',default='run',type=str,help='set \'run\' if debug done, otherwise, set debug')
    args = parser.parse_args()
    return args

if __name__ =='__main__':

    # example: if the data path not setted from args,just manually set them as belows.
    #dataset='ucf101'
    #data_root='/S2/MI/zqj/video_classification/data'
    #data_root=os.path.join(data_root,dataset)

    args=parse_args()
    data_root=os.path.join(args.data_root,args.dataset)
    videos_root=os.path.join(data_root,'mpeg4_videos')
    train_dir_rgb = '/media/senilab/DATA2/feature_hmdb_i3d_5c_dp_16/train_rgb/'
    train_dir_flow = '/media/senilab/DATA2/feature_hmdb_i3d_5c_dp_16/train_flow/'
    # train_dir_dp = '/media/senilab/DATA2/feature_hmdb_i3d_5c_dp_16/train_dp/'
    test_dir_rgb = '/media/senilab/DATA2/feature_hmdb_i3d_5c_dp_16/test_rgb/'
    test_dir_flow = '/media/senilab/DATA2/feature_hmdb_i3d_5c_dp_16/test_flow/'
    # test_dir_dp = '/media/senilab/DATA2/feature_hmdb_i3d_5c_dp_16/test_dp/'

    save_dir_train_rgb = '/media/senilab/DATA2/att_feature_5c_16_dp/rgb_train/'
    save_dir_test_rgb = '/media/senilab/DATA2/att_feature_5c_16_dp/rgb_test/'
    save_dir_train_flow = '/media/senilab/DATA2/att_feature_5c_16_dp/flow_train/'
    save_dir_test_flow = '/media/senilab/DATA2/att_feature_5c_16_dp/flow_test/'
    # save_dir_train_dp = '/media/senilab/DATA2/att_feature_5c_16_dp/dp_train/'
    # save_dir_test_dp = '/media/senilab/DATA2/att_feature_5c_16_dp/dp_test/'

    list_train_rgb = sorted([f for f in listdir(train_dir_rgb) if isfile(join(train_dir_rgb, f))])
    list_test_rgb = sorted([f for f in listdir(test_dir_rgb) if isfile(join(test_dir_rgb, f))])
    list_train_flow = sorted([f for f in listdir(train_dir_flow) if isfile(join(train_dir_flow, f))])
    list_test_flow = sorted([f for f in listdir(test_dir_flow) if isfile(join(test_dir_flow, f))])
    # list_train_dp = sorted([f for f in listdir(train_dir_dp) if isfile(join(train_dir_dp, f))])
    # list_test_dp = sorted([f for f in listdir(test_dir_dp) if isfile(join(test_dir_dp, f))])
    list_sisa = sorted([f for f in listdir(save_dir_test_flow) if isfile(join(save_dir_test_flow, f))])

    list_error = ['jonhs_netfreemovies_holygrail_talk_u_nm_np1_le_med_18']
    
    list_sisa_train = np.setdiff1d(list_test_flow, list_sisa)
    #specify the augments
    num_workers=args.num_workers
    mode=args.mode
    #get video list
    video_list=list_sisa_train
    # print ('video_list', video_list)
    video_split = video_list

    len_videos=len(list_sisa_train) # if we choose the ucf101
    print ('find {} videos.'.format(len_videos))
    flows_dirs=[video.split('.')[0] for video in video_split]
    print ('get videos list done! ')
    # print ('video_list', video_list)

    pool=Pool(num_workers)
    if mode=='run':
        pool.map(get_data,(list_sisa_train))
    else: #mode=='debug
        get_data((list_sisa_train))
