import time
import numpy as np
import matplotlib.image as img
from . import custom_lzma 
from . import image_utils
from . import glob



def evaluate_compression_ratio(original_image_path, code, p=True):
    (compressed_blocks_lz, compressed_blocks_shape, 
    important_errors_lz, imp_errors_shape,
    padding_info, width_immagine_originale,
    height_immagine_originale, alpha_channel_lz) = code
    
    immagine_rgb, alpha_channel=image_utils.load_image(original_image_path)
    

    size_immagine_rgb= number_of_bytes(immagine_rgb)
    if p:
        print("size_immagine_rgb " + str(size_immagine_rgb) + 
          " shape immagine_rgb " + str(np.asarray(immagine_rgb).shape))
        print("type immagine_rgb " + str(type(immagine_rgb[0][0][0])) + "\n")
    
    if(alpha_channel is not None):
        size_alpha_channel= number_of_bytes(alpha_channel)
        if p:
            print("size_alpha_channel " + str(size_alpha_channel))
            print("type alpha_channel " + str(type(alpha_channel[0][0][0])) + "\n")
    else: 
        size_alpha_channel=0
        if p:
             print("size_alpha_channel = 0 \n")
      
    size_compressed_blocks_lz= len(compressed_blocks_lz)
    if p:
        print("size_compressed_blocks_lz " + str(size_compressed_blocks_lz))
        print("type compressed_blocks_lz " + str(type(compressed_blocks_lz)) + "\n")
    
    size_important_errors_lz= len(important_errors_lz)
    if p:
        print("size_important_errors " + str(size_important_errors_lz))
        print("type important_errors_lz " + str(type(important_errors_lz)) + "\n")
       
    if (alpha_channel_lz is not None):
        size_alpha_channel_lz= len(alpha_channel_lz)
        if p:
            print("size_alpha_channel_lz " + str(size_alpha_channel_lz))
            print("type alpha_channel_lz " + str(type(alpha_channel_lz)) + "\n")
    else :
        size_alpha_channel_lz=0
        if p:
             print("size_alpha_channel_lz = 0 \n")
    
 
    compressed_size=size_compressed_blocks_lz  + size_important_errors_lz + size_alpha_channel_lz
    original_size=size_alpha_channel + size_immagine_rgb
    
    print ("COMPRESSION RATIO IS ORIGINAL_SIZE/COMPRESSED_SIZE," +  
           "WHERE ORIGINAL SIZE = size_alpha_channel + size_immagine_rgb \n" + 
           "AND compressed_size=size_compressed_blocks_lz " +
           "+ size_important_errors_lz + size_alpha_channel_lz ")
                  
    return original_size/compressed_size



def  number_of_bytes(data_structure):
    if (data_structure is None):
        return 0
    #Sono tutti float32
    size=1
    for i in range (0, len((np.asarray(data_structure)).shape)):
        size *= np.asarray(data_structure).shape[i]
    return 4 * size