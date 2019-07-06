import matplotlib.image as img
import numpy as np
import math

from types import SimpleNamespace
from . import glob

def pad_train_image(image):
    n_rows = image.shape[0]
    n_cols = image.shape[1]
    n_chan = image.shape[2]
    
    
    if n_rows % glob.block_size != 0:
        diff = glob.block_size - (n_rows % glob.block_size)
        pad = np.zeros((diff, n_cols, n_chan))
        pad= image[n_rows-diff:n_rows, :, :]
        image = np.concatenate((image, pad), axis=0)
        
    n_rows = image.shape[0]
    
    if n_cols % glob.block_size != 0:
        diff = glob.block_size - (n_cols % glob.block_size)
        pad= image[:, n_cols-diff:n_cols, :]
        image = np.concatenate((image, pad), axis=1)
    
    return image


def pad_test_image(image):
    padding_info = SimpleNamespace()
    
    padding_info.size_pad_left=glob.frame_size
    padding_info.size_pad_right=glob.frame_size
    padding_info.size_pad_top=glob.frame_size
    padding_info.size_pad_bottom=glob.frame_size
    
    #Add frame to entire image
    n_rows = image.shape[0]
    n_cols = image.shape[1]
    n_chan = image.shape[2]
    
    pad = np.zeros((glob.frame_size, n_cols, n_chan), dtype=np.float32)
    image = np.concatenate((pad, image), axis=0)
    image = np.concatenate((image, pad), axis=0)
    
    n_rows = image.shape[0]
    n_cols = image.shape[1]
    n_chan = image.shape[2]
    
    pad = np.zeros((n_rows, glob.frame_size, n_chan), dtype=np.float32)
    image = np.concatenate((pad, image), axis=1)
    image = np.concatenate((image, pad), axis=1)
    
    n_rows = image.shape[0]
    n_cols = image.shape[1]
    n_chan = image.shape[2]
    
    step= glob.block_size - (glob.frame_size * 2)
    print ("step " + str(step))
    print("shape dopo pad cornice " + str(image.shape[0]) + " " + str(image.shape[1]))
    
    v_blks = math.ceil(n_rows/step)
    image = compute_last_v_block_padding(image, n_rows, n_cols, n_chan, step, padding_info, v_blks)
     
    n_rows = image.shape[0]  
    
    h_blks = math.ceil(n_cols/step)
    image = compute_last_h_block_padding(image, n_rows, n_cols, n_chan, step, padding_info, h_blks)
    
    padding_info.number_of_horizontal_blocks = h_blks
    padding_info.number_of_vertical_blocks = v_blks
    print(str(image.shape[0]) + " " + str(image.shape[1]))
    print(str(padding_info.number_of_horizontal_blocks))
    print(str(padding_info.number_of_vertical_blocks))
    return image, padding_info


def compute_last_v_block_padding(image, n_rows, n_cols, n_chan, step, padding_info, v_blks):  
    last_blk_start_idx = (v_blks-1) * step 
    height_last_v_block = n_rows - last_blk_start_idx
    if height_last_v_block < glob.block_size:
        diff = glob.block_size - height_last_v_block
        padding_info.size_pad_bottom= padding_info.size_pad_bottom + diff
        pad = np.zeros((diff, n_cols, n_chan), dtype=np.float32)
        image = np.concatenate((image, pad), axis=0)
    return image


def compute_last_h_block_padding(image, n_rows, n_cols, n_chan, step, padding_info, h_blks):
    last_blk_start_idx = (h_blks-1) * step 
    width_last_h_block = n_cols - last_blk_start_idx
    if width_last_h_block < glob.block_size:    
        diff = glob.block_size - width_last_h_block
        padding_info.size_pad_right = padding_info.size_pad_right + diff
        pad = np.zeros((n_rows, diff, n_chan), dtype=np.float32)
        image = np.concatenate((image, pad), axis=1)
    return image


def unpad_image(decompressed_image, padding_info):
    h_decompressed_img = len(decompressed_image)
    w_decompressed_img = len(decompressed_image[0])
    first_top_idx = padding_info.size_pad_top - glob.frame_size
    first_bottom_idx = h_decompressed_img - padding_info.size_pad_bottom + glob.frame_size
    first_left_idx = padding_info.size_pad_left - glob.frame_size
    first_right_idx = w_decompressed_img - padding_info.size_pad_right + glob.frame_size
                                  
    return decompressed_image[first_top_idx:first_bottom_idx, first_left_idx:first_right_idx, :]                             


def get_train_blocks(image):
    
    image = pad_image(image)
    
    blocks = []
    n_rows = image.shape[0]
    n_cols = image.shape[1]
    n_chan = image.shape[2]
    
    for r in range(0, n_rows, glob.block_size):
        for c in range(0, n_cols, glob.block_size):
            block = image[r:r+glob.block_size, c:c+glob.block_size, :]
            blocks.append(block)
            
    return blocks
       
    
def get_test_blocks(image,  padding_info):
    passo=  glob.block_size - (glob.frame_size * 2)
    
    blocks = []
    n_rows = image.shape[0]
    n_cols = image.shape[1]
    n_chan = image.shape[2]
    
    
    for r in range(0, padding_info.number_of_vertical_blocks):
        for c in range(0, padding_info.number_of_horizontal_blocks):
            start_row_index = r * passo
            start_col_index = c* passo
            block = image[start_row_index:start_row_index+glob.block_size, start_col_index:start_col_index+glob.block_size, :]
            blocks.append(block)
    return blocks


def load_image(image):

    a=img.imread(image)

    alpha_channel=None
    if (a.shape[2]==4):
        alpha_channel= np.expand_dims(a[:,:,3], axis=2)
        
    return a[:,:,0:3], alpha_channel


def get_train_set(dataset_path, n_images):
    train_blocks = []


    for idx, f in enumerate(os.listdir("dataset")):
        
        if idx > n_images-1:
            break
        
        if 'png' in f:
            print("Sto caricando " + "dataset/" + f)
            blocks = get_blocks(load_nef_image("dataset/"+f))
            if blocks == -1:
                os.remove("dataset/"+f)
                continue
            train_blocks.extend(blocks)
        print("Indice è " + str(idx) + " n_images è " + str(n_images))
      
            
    train_blocks = np.asarray(train_blocks).astype(int)
    return train_blocks


def get_psnr(original, compressed):
    h = original.shape[0]
    w = original.shape[1]

    mse = 0
    for r in range(h):
        for c in range(w):
            for v in range(3):
                mse += (original[r, c, v]-compressed[r, c, v])**2

    mse = mse/float(h*w)
    arg = 1/mse**(1/2)
    return 20*math.log10(arg)