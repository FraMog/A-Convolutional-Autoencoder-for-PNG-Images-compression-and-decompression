import time
import numpy as np
import matplotlib.image as img
from . import custom_lzma 
from . import image_utils
from . import glob
from . import compression_utils

def decompress(code, decoder, p=True):
    (compressed_blocks_lz, compressed_blocks_shape, 
    err_received, imp_errors_shape,
    padding_info, width_immagine_originale,
    height_immagine_originale, alpha_channel_lz) = code
    
    t0= time.clock()
    t1= time.clock()
    if p:
        print("Decompressor STARTS")
    
    #Alpha channel decompression with LZMA
    if alpha_channel_lz is not None:
        alpha_channel_shape = (height_immagine_originale, width_immagine_originale, 1)
        alpha_channel = custom_lzma.decompression(alpha_channel_lz, alpha_channel_shape)
        
        if p:
            print("Alpha channel lz decompressed. Elapsed: {}".format(time.clock()-t0))
            t0= time.clock()
    else:
        alpha_channel=None
    
    alpha_channel=None
    print(str(len(compressed_blocks_lz) ))                            
    # Blocks decompression through LZMA
    compressed_blocks= custom_lzma.decompression(compressed_blocks_lz, compressed_blocks_shape)
    if p:
        print("Image blocks LZMA decompressed. Elapsed: {}".format(time.clock()-t0))
        t0= time.clock()
                                  
    # Lossy image blocks decompression through decoder network.                               
    net_decompressed_image= compression_utils.decompress_image(decoder, compressed_blocks, padding_info, 
                                         width_immagine_originale, height_immagine_originale)
    net_decompressed_image_copy= net_decompressed_image.copy()
    if p:
        print("End network blocks decompression. Elapsed: {}".format(time.clock()-t0))
        t0= time.clock()

    if(alpha_channel is not None):
        net_decompressed_image_copy = np.concatenate((net_decompressed_image_copy, alpha_channel), axis=2)
    img.imsave("predicted/network_decompression.png", net_decompressed_image_copy)
    if p:
        print("Network decompressed image saved. Elapsed: {}".format(time.clock()-t0))
        t0= time.clock()
    
                                  
    #Error correction phase    
    important_errors_lz= err_received
    important_errors= custom_lzma.decompression(important_errors_lz, imp_errors_shape)
    if p:
        print("important errors decompressed through LZMA. Elapsed: {}".format(time.clock()-t0))
    t0= time.clock()
    
    err_corrected_image = compression_utils.migliora_immagine(net_decompressed_image, important_errors)
    if p:
        print("Image error corrected. Elapsed: {}".format(time.clock()-t0))
        t0= time.clock()  
    if(alpha_channel is not None):
        alpha_err_corrected_image = np.concatenate((decompressed_image_migliorata, alpha_channel), axis=2)
    img.imsave("predicted/decompression_error_corrected.png", err_corrected_image)   
    if p:
        print("Error corrected image save. Elapsed: {}".format(time.clock()-t0))
                                  
    print("Decompressor ENDS. Total decompressor computation time is {}".format(time.clock()-t1))