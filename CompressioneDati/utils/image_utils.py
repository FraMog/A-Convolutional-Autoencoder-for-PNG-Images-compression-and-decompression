import matplotlib.image as img
import numpy as np
import math

from types import SimpleNamespace
from . import glob

def pad_image(image):
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
    
    padding_info.size_pad_cols_left=glob.frame_size
    padding_info.size_pad_cols_right=glob.frame_size
    padding_info.size_pad_cols_top=glob.frame_size
    padding_info.size_pad_cols_bottom=glob.frame_size
    
    #Aggiungo size_cornice colonne a sinistra e destra e size_cornice righe in alto e in basso
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
    
    passo= glob.block_size - (glob.frame_size * 2)

    if n_rows % passo != 0:
     
        diff = glob.block_size - (n_rows % passo)
        padding_info.size_pad_cols_bottom= padding_info.size_pad_cols_bottom + diff
        pad = np.zeros((diff, n_cols, n_chan), dtype=np.float32)
        image = np.concatenate((image, pad), axis=0)
        
    n_rows = image.shape[0]
    
    if n_cols % passo != 0:
        diff = glob.block_size - (n_cols % passo)
        padding_info.size_pad_cols_right = padding_info.size_pad_cols_right + diff
        pad = np.zeros((n_rows, diff, n_chan), dtype=np.float32)
        image = np.concatenate((image, pad), axis=1)
     
    #Numero dei blocchi block_size x blocks_size x blocks_size che devo considerare sia orizzontalmente che verticalmente, 
    #tenendo conto che sui due assi mi muovo con step di  block_size/2 pixel alla volta.
    
    padding_info.number_of_horizontal_blocks= int(image.shape[1]/passo)
    padding_info.number_of_vertical_blocks= int(image.shape[0]/passo) 
    
    return image, padding_info


def get_blocks(image):
    
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
    #Se i canali sono almeno 4, ovvero, se oltre ai valori RGB ne è presente un altro (solitamente è l'alpha channel).
    if (a.shape[2]==4):
        alpha_channel= np.expand_dims(a[:,:,3], axis=2)
    #Separo RGB ed alpha channel
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