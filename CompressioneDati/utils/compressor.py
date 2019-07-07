import time
import numpy as np
import matplotlib.image as img
from . import custom_lzma 
from . import image_utils
from . import glob
from . import compression_utils

def compress(image_path, compression_net, decompression_net, quality, p=True):
    quality=100-quality
    t0= time.clock()
    t1= time.clock()
    
    if p:
        print("Compressor STARTS")
    
    
    immagine_rgb, alpha_channel=image_utils.load_image(image_path)
    if p:
        print("Image loaded. Elapsed: {}".format(time.clock()-t0))
        t0= time.clock()
    width_immagine_originale=len(immagine_rgb[0])
    height_immagine_originale=len(immagine_rgb)

    
    img_padded, padding_info = image_utils.pad_test_image(immagine_rgb)
    if p:
        print("Image padded. Elapsed: {}".format(time.clock()-t0))
        t0 = time.clock()
    
   
    image_blocks =image_utils.get_test_blocks(img_padded, padding_info)
    if p:
        print("Image separated into blocks. Elapsed: {}".format(time.clock()-t0))
        t0 = time.clock()
    
    
    if alpha_channel is not None:
        alpha_channel_lz= custom_lzma.compression(alpha_channel)
        if p:
            print("Alpha channel LZMA-compressed. Elapsed: {}".format(time.clock()-t0))
            t0 = time.clock()
    else: 
        alpha_channel_lz=None

        
    # Lossy image blocks compression through encoder network
    compressed_blocks=compression_utils.compress_image(compression_net, image_blocks)
    compressed_blocks_shape = np.asarray(compressed_blocks).shape
    if p:
        print("Image blocks compressed through network. Elapsed: {}".format(time.clock()-t0))
        t0 = time.clock() 
   

    # Compressed blocks are recompressed losslessly with LZMA
    compressed_blocks_lz = custom_lzma.compression(compressed_blocks)

    
    #compressed_blocks_lz = custom_lzma.compression(compressed_blocks)
    if p:
        print("Blocks recompressed through LZMA. Elapsed: {}".format(time.clock()-t0))
        t0= time.clock() 
  

    # Error computation phase
    #Compressed blocks are decompressed and reshaped. The outuput is the decompressed_image
    net_decompressed_image= compression_utils.image_decompression_with_autoencoder(decompression_net, compressed_blocks,
                                                                                   padding_info, width_immagine_originale,
                                                                                 height_immagine_originale, p)
    
    net_decompressed_image = image_utils.unpad_image(net_decompressed_image, padding_info)
    if p:
        print("Image decompressed. Elapsed: {}".format(time.clock()-t0))
        t0= time.clock() 
      
    
    # Treshold evaluation
    treshold = compression_utils.get_treshold(immagine_rgb, net_decompressed_image, quality)
    if p:
        print("Proper treshold computed. Elapsed: {}".format(time.clock()-t0))
        print("Treshold is " + str(treshold))
        t0= time.clock()   
    important_errors =  compression_utils.compute_relevant_errors(immagine_rgb, net_decompressed_image, treshold, p)  
    imp_errors_shape=important_errors.shape
    if p:
        print("Relevant errors computed. Elapsed: {}".format(time.clock()-t0))
        t0= time.clock()   
        
        
    # important_errors compressed through LZMA
    important_errors_bytes= important_errors.tobytes()
    important_errors_lz = custom_lzma.compression(important_errors) 
    err_to_send= important_errors_lz 
    if p:
        print("Relevant errors LZMA compressed. Elapsed: {}".format(time.clock()-t0))
        print("Compressor ENDS. Total compressor computation time is {}".format(time.clock()-t1) + "\n")
    
    
    compression_output = (compressed_blocks_lz, compressed_blocks_shape,
                         err_to_send, imp_errors_shape,
                         padding_info, width_immagine_originale,
                         height_immagine_originale, alpha_channel_lz)
    
    return compression_output