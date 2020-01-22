import os

import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import seaborn as sns

from skimage.io import imread
from skimage.segmentation import mark_boundaries
from skimage.morphology import label, binary_opening, disk

from sklearn.model_selection import train_test_split

import gc
import datetime


gc.enable()

import keras.backend as K

from keras.layers import *

from keras.models import load_model, Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import binary_crossentropy


image_path = 'image_data/train_v2/'
test_image = 'image_data/test_v2/'

csv_dataset = 'csv_data/train_ship.csv'

exclude_list = ['6384c3e78.jpg', ]
original_img_size = (768, 768)

LR = 0.0001

BATCH_SIZE = 8
EDGE_CROP = 16
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'
# downsampling inside the network
NET_SCALING = (1, 1)
# downsampling in preprocessing
IMG_SCALING = (3, 3)
# number of validation images to use
VALID_IMG_COUNT = 900
# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 1500
MAX_TRAIN_EPOCHS = 99
AUGMENT_BRIGHTNESS = True

weight_path="{}_weights.hdf5".format('seg_model')


df = pd.read_csv(csv_dataset)

df = df[~df['ImageId'].isin(exclude_list)]
df['ships'] = df.groupby(['ImageId'])['ImageId'].transform('count')
df.loc[df['EncodedPixels'].isnull().values,'ships'] = 0

df['file_size_kb'] = df['ImageId'].map(lambda image: 
                                       os.stat(os.path.join(image_path, image)).st_size/1024)
df = df[df['file_size_kb'] > 60]


imgs_w_ships = df[df['ships'] > 0]
imgs_wo_ships = df[df['ships'] == 0].sample(20000, random_state=6978)

selected_imgs = pd.concat((imgs_w_ships, imgs_wo_ships))

selected_imgs['has_ship'] = selected_imgs['ships'] > 0

del df

train_imgs, val_imgs = train_test_split(selected_imgs,
                                        test_size=0.10,
                                        stratify=selected_imgs['has_ship'],
                                        random_state=69278)
                                        
                                        
train_fnames = np.unique(train_imgs['ImageId'].values)
val_fnames = np.unique(val_imgs['ImageId'].values)


_, train_fnames = train_test_split(train_fnames, test_size=0.45, random_state=6978)
_, val_fnames = train_test_split(val_fnames, test_size=0.45, random_state=6978)


reduced_df = selected_imgs[['ImageId', 'EncodedPixels']]

train_df = reduced_df[reduced_df['ImageId'].isin(train_fnames)]
val_df = reduced_df[reduced_df['ImageId'].isin(val_fnames)]

del selected_imgs

def rle_encoder(img, min_max_threshold=1e-3, max_mean_threshold=None):
    """
    function for encoding image to submission dataset
    """
    if np.max(img) < min_max_threshold:
        return ''
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return ''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decoder(mask_rle, shape=(768, 768)):
    """
    function for decoding target image
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def masks2image(in_mask_list):
    """
    function for creating multi-layers mask
    that consist some ships on ine image
    """
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decoder(mask)
    return all_masks

def batch_creator(batch_fname, batch_df):
    height, width = original_img_size # take height and width for shape of input
    input_images = np.zeros((BATCH_SIZE, height, width, 3))
    target_images = np.zeros((BATCH_SIZE, height, width, 1))
    for i, img_name in enumerate(batch_fname):
        input_images[i, ...] = imread(os.path.join(image_path, img_name))
        rle = batch_df[batch_df['ImageId'] == img_name]['EncodedPixels'].values
        target_images[i, ...] = np.expand_dims(masks2image(rle), axis = 2)
    return (input_images, target_images)
    
def generate(df_x, df_y):
    np.random.shuffle(df_x)
    num_of_batches = int(np.ceil(len(df_x) / float(BATCH_SIZE)))
    for batch in range(num_of_batches):
        batch_fname = np.random.choice(df_x, BATCH_SIZE)
        batch_df = df_y[df_y['ImageId'].isin(batch_fname)]
        yield batch_creator(batch_fname, batch_df)
            
            

t_gen = generate(train_fnames, train_df)
v_gen = generate(val_fnames, val_df)


dg_args = dict(featurewise_center = False, 
               samplewise_center = False,
               rotation_range = 45, 
               width_shift_range = 0.1, 
               height_shift_range = 0.1, 
               shear_range = 0.01,
               zoom_range = [0.5, 0.8],  
               horizontal_flip = True, 
               vertical_flip = True,
               fill_mode = 'nearest',
               data_format = 'channels_last')
# brightness can be problematic since it seems to change the labels differently from the images 
if AUGMENT_BRIGHTNESS:
    dg_args['brightness_range'] = [0.5, 1]
image_gen = ImageDataGenerator(**dg_args)

if AUGMENT_BRIGHTNESS:
    dg_args.pop('brightness_range')
    
label_gen = ImageDataGenerator(**dg_args)

def create_aug_gen(in_gen, seed = None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(in_x, 
                             batch_size = in_x.shape[0], 
                             seed = seed, 
                             shuffle=True)
        g_y = label_gen.flow(in_y, 
                             batch_size = in_x.shape[0], 
                             seed = seed, 
                             shuffle=True)
        gc.collect()

        yield next(g_x)/255.0, next(g_y)
        
        
checkpoint = ModelCheckpoint(weight_path,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                   patience=1, verbose=1, mode='min',
                                   min_delta=0.0001, cooldown=0, min_lr=1e-8)

early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      verbose=2,
                      patience=20)

callbacks_list = [checkpoint, reduceLROnPlat, early]


model = load_model('U-Net.h5')

gc.collect()

def dice_score(y_pred, y_target):
    comm = K.sum(y_pred * y_target)
    return (comm * 2.0)/(K.sum(y_pred) + K.sum(y_target))

def dice_loss(y_pred, y_target):
    return 1-dice_score(y_pred, y_target)
    
    
def fit():
    model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=dice_loss, metrics=['binary_accuracy', dice_score])
    
    step_count = min(MAX_TRAIN_STEPS, train_df.shape[0]//BATCH_SIZE)
    #aug_gen = create_aug_gen(t_gen)
    model.fit_generator(t_gen,
                        steps_per_epoch=step_count,
                        epochs=MAX_TRAIN_EPOCHS,
                        validation_data=v_gen,
                        validation_steps = 10,
                        #callbacks=callbacks_list,
                        )
    return model
    
if __name__ == '__main__':
    loss_history = fit()
    print(np.min([mh.history['val_loss'] for mh in loss_history]))
