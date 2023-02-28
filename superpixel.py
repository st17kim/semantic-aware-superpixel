
import tensorflow as tf
from tf.keras import Input, Model
from tf.keras.layers import Conv2D, LayerNormalization, Dense
from tf.keras.activations import gelu
import tf.constant_initializer as init
import numpy as np
import cv2
import os
# from matplotlib import pyplot as plt
from PIL import Image
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels


#%%

''' Choose one. 
    8x models generates better superpixels but you may suffer from memory issue.
    We recommend to use 16s model for the first trial.
'''

# load_name = "model/dino2248s"
# load_name = "model/dino2248b"
load_name = "model/dino22416s"
# load_name = "model/dino22416b"



img_file = "eg.jpg"
out_path = "superpixels/"

use_crf = False # install pydensecrf to use this option.


#%%

# do not change this

C = 384 if load_name[-1]=="s" else 768
P = 16 if load_name[-2]=="6" else 8

H = C//64



#%%


def crf_inference_label(img, labels, t=20, n_labels=21, gt_prob=0.7):

    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=10, compat=10)
    d.addPairwiseBilateral(sxy=20, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)
    


def img_load(img_file, s=1):
    img = cv2.imread(img_file)[::-1]
    h0,w0 = img.shape[:2]
    h,w = int((img.shape[0]*s-1)//P+1),int((img.shape[1]*s-1)//P+1)
    resimg = cv2.resize(img, (w*P,h*P), interpolation=cv2.INTER_CUBIC)/255
    return np.array([resimg]), h,w, img, h0,w0

def Conv2d(In):
    Weight = init_model["patch_embed.proj.weight"]
    Bias = init_model["patch_embed.proj.bias"]
    w = init(Weight)
    b = init(Bias)
    convlayer = Conv2D(Weight.shape[3], Weight.shape[0], 
                       strides=[Weight.shape[0],Weight.shape[1]], 
                       padding='same', kernel_initializer=w, bias_initializer=b)(In)
    return convlayer

def layernorm(In, w,b):
    W = tf.compat.v1.constant_initializer(w)
    B = tf.compat.v1.constant_initializer(b)
    out = LayerNormalization(beta_initializer=B, gamma_initializer=W, epsilon=1e-6)(In)
    return out

def block(In, n):
    
    out = dict()
    
    qkv_w, qkv_b = init_model["blocks."+n+".attn.qkv.weight"], init_model["blocks."+n+".attn.qkv.bias"]
    proj_w, proj_b = init_model["blocks."+n+".attn.proj.weight"],init_model["blocks."+n+".attn.proj.bias"]
    fc1_w, fc1_b = init_model["blocks."+n+".mlp.fc1.weight"], init_model["blocks."+n+".mlp.fc1.bias"]
    fc2_w, fc2_b = init_model["blocks."+n+".mlp.fc2.weight"], init_model["blocks."+n+".mlp.fc2.bias"]
    
    qkv_W, qkv_B = init(qkv_w), init(qkv_b)
    proj_W, proj_B = init(proj_w), init(proj_b)
    fc1_W, fc1_B = init(fc1_w), init(fc1_b)
    fc2_W, fc2_B = init(fc2_w), init(fc2_b)
    
    out["norm1"] = layernorm(In, init_model["blocks."+n+".norm1.weight"], init_model["blocks."+n+".norm1.bias"])
    out["qkvlayer"] = Dense(qkv_w.shape[1], kernel_initializer=qkv_W, bias_initializer=qkv_B)(out["norm1"])
    qkv = tf.reshape(out["qkvlayer"], (-1, tf.shape(In)[1], 3, H, 64))
    qkvt = tf.transpose(qkv, (2,0,3,1,4))
    
    out["q"] = qkvt[0]
    out["k"] = qkvt[1]
    out["v"] = qkvt[2]
    
    out["mqk"] = tf.matmul(out["q"],out["k"],transpose_b=True)/tf.sqrt(64.)
    out["qksoft"] = tf.nn.softmax(out["mqk"])
    out["mqkv"] = tf.matmul(out["qksoft"],out["v"])
    mqkv_trans = tf.transpose(a=out["mqkv"], perm=[0,2,1,3])
    mqkv_res = tf.reshape(mqkv_trans, [-1,tf.shape(In)[1],C])
    out["projlayer"] = Dense(proj_w.shape[1], kernel_initializer=proj_W, bias_initializer=proj_B)(mqkv_res)
    out["out1"] = In + out["projlayer"]
    
    out["norm2"] = layernorm(out["out1"], init_model["blocks."+n+".norm2.weight"], init_model["blocks."+n+".norm2.bias"])
    out["fc1"] = Dense(fc1_w.shape[1], kernel_initializer=fc1_W, bias_initializer=fc1_B)(out["norm2"])
    out["gelufc1"] = gelu(out["fc1"])
    out["fc2"] = Dense(fc2_w.shape[1], kernel_initializer=fc2_W, bias_initializer=fc2_B)(out["gelufc1"])
    
    out["out2"] = out["out1"] + out["fc2"]
    return out

def _get_voc_pallete(num_cls):
    n = num_cls
    pallete = [0]*(n*3)
    for j in range(0,n):
        lab = j
        pallete[j*3+0] = 0
        pallete[j*3+1] = 0
        pallete[j*3+2] = 0
        i = 0
        while (lab > 0):
            pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
            pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
            pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3
    return pallete

def get_mask_pallete(npimg):
    vocpallete = _get_voc_pallete(256)
    out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
    out_img.putpalette(vocpallete)
    return out_img

#%%

X = Input((None,None,3))

color_mean = tf.constant([[[[0.485,0.456,0.406]]]])
color_std = tf.constant([[[[0.229,0.224,0.225]]]])

init_model = np.load(load_name+".npy",allow_pickle=True).item()

b = dict()
Aug_X = (X-color_mean)/color_std
conv11 = Conv2d(Aug_X)

conv11_vec = tf.reshape(conv11, [-1,tf.shape(conv11)[1]*tf.shape(conv11)[2],C])
cls_token = tf.constant(init_model["cls_token"])*tf.ones([tf.shape(X)[0],1,C])
conv11_197 = tf.concat([cls_token, conv11_vec], axis=1)
posem = tf.constant(init_model["pos_embed"])

repos = tf.reshape(posem[0,1:], [-1,224//P,224//P,C])
respos = tf.image.resize(repos, (tf.shape(X)[1]/P, tf.shape(X)[2]/P), tf.image.ResizeMethod.BICUBIC)
flapos = tf.reshape(respos, [-1,tf.shape(respos)[1]*tf.shape(respos)[2], C])
conpos = tf.concat([posem[:,0:1], flapos], axis=1)

b["0"] = conv11_197+conpos

b["1"] = block(b["0"], "0")
b["2"] = block(b["1"]["out2"], "1")
b["3"] = block(b["2"]["out2"], "2")
b["4"] = block(b["3"]["out2"], "3")
b["5"] = block(b["4"]["out2"], "4")
b["6"] = block(b["5"]["out2"], "5")
b["7"] = block(b["6"]["out2"], "6")
b["8"] = block(b["7"]["out2"], "7")
b["9"] = block(b["8"]["out2"], "8")
b["10"] = block(b["9"]["out2"], "9")
b["11"] = block(b["10"]["out2"], "10")
b["12"] = block(b["11"]["out2"], "11")

model = Model(X, b["12"]["k"])


#%%


def generate_superpixel(t,path,crf=False):
    path1 = out_path + path
    os.makedirs(path1, exist_ok=True)
    if crf == True: 
        path2 = out_path + path+"crf"
        os.makedirs(path2, exist_ok=True)
    img, h,w, img0, h0,w0= img_load(img_file)
    
    out = model(img).numpy()[0,:,1:]
    f = np.transpose(out, (1,0,2)).reshape(w*h,C)
    
    f = f/ np.sqrt(np.sum(f**2, axis=1, keepdims=True))
    apq = f@f.T
    bpq = apq>0
    
    pset = np.ones([h*w,1], dtype=np.bool_)
    selset = np.int32(np.zeros([0,h*w]))
    count = 0
    
    while np.sum(pset) != 0 :
    
        bpq_m = bpq*pset*pset.T
        
        dp = np.sum(bpq_m, 0)
        dp[dp==0] = h*w+1
        
        pstar = np.argmin(dp)       # seed pixel index
        
        S = apq[pstar]
        set1 = (S>t) & (pset[:,0] == 1)
        
        selset = np.concatenate([selset, set1.reshape(1,h*w)], axis=0)
        pset[set1==True] = 0
        count += 1 
    
    print("model: DINO", load_name[13:])
    print("tau =", t, "the number of superpixels: ", count)
    
    npimg = np.argmax(selset, axis=0).reshape(h,w)
    npimg = cv2.resize(npimg, (w0,h0), interpolation=cv2.INTER_NEAREST)
    get_mask_pallete(npimg+1).save(path1+"/"+img_file[:-4]+".png")
    if crf == True: 
        npimg = crf_inference_label(img0, npimg, n_labels=np.max(npimg)+1)
        get_mask_pallete(npimg+1).save(path2+"/"+img_file[:-4]+".png")


''' 
    You can change tau freely ranging from -1 to 1.
    Recommended value is 0 ~ 0.4
'''

generate_superpixel(0,load_name[13:]+"-00",crf=use_crf)
generate_superpixel(0.1,load_name[13:]+"-10",crf=use_crf)
generate_superpixel(0.2,load_name[13:]+"-20",crf=use_crf)
generate_superpixel(0.3,load_name[13:]+"-30",crf=use_crf)
