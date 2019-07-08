import time
import numpy as np
import math
import matplotlib.image as img
from . import custom_lzma 
from . import image_utils
from . import glob
from . import compression_utils
from . import compressor
from . import decompressor




def evaluate_compression_ratio(rgb_image, alpha_channel, code, p=True):
    (compressed_blocks_lz, compressed_blocks_shape, 
    important_errors_lz, imp_errors_shape,
    padding_info, width_immagine_originale,
    height_immagine_originale, alpha_channel_lz) = code
    
    
    size_rgb_image= number_of_bytes(rgb_image)
    if p:
        print("size_rgb_image " + str(size_rgb_image) + 
          " shape rgb image " + str(np.asarray(rgb_image).shape))
        print("type rgb_image " + str(type(rgb_image[0][0][0])) + "\n")
    
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
    original_size=size_alpha_channel + size_rgb_image
    
    if p:
        print ("compression_ratio = original_size/compressed_size \n" +  
               "original_size = size_alpha_channel + size_rgb_image \n" + 
               "compressed_size = size_compressed_blocks_lz " +
               "+ size_important_errors_lz + size_alpha_channel_lz ")
                  
    return original_size/compressed_size



def number_of_bytes(data_structure):
    if (data_structure is None):
        return 0
    #Sono tutti float32
    size=1
    for i in range (0, len((np.asarray(data_structure)).shape)):
        size *= np.asarray(data_structure).shape[i]
    return 4 * size




# Decompress blocks through a decoder, and evaluate decoder decompression error.
def autoencoder_error_evaluation(rgb_image, compressed_blocks,
                                 decompression_net, width_immagine_originale, 
                                 height_immagine_originale, padding_info,
                                 p=True):
           
    net_decompressed_image = compression_utils.image_decompression_with_autoencoder(decompression_net, compressed_blocks,
                                                                                    padding_info, width_immagine_originale,
                                                                                    height_immagine_originale, p)
    
    net_decompressed_image = image_utils.unpad_image(net_decompressed_image, padding_info)
    
    error_image = net_decompressed_image - rgb_image
    error_image = error_image.reshape(-1)
    mean = np.mean(error_image)
    std = np.std(error_image)

    error_image_absolute = np.absolute(error_image)
    mean_absolute = np.mean(error_image_absolute)
    std_absolute = np.std(error_image_absolute)
    

    return error_image, error_image_absolute, mean, std, mean_absolute, std_absolute
   
    
    
def validation_set_autoencoder_error_evaluation(compression_net, decompression_net, p=True):
    validation_set_errors = []
    for i in range (100):
        image_path = "dataset/valid/" + str(i) + ".png"
        rgb_image, alpha_channel=image_utils.load_image(image_path)

        width_immagine_originale=len(rgb_image[0])
        height_immagine_originale=len(rgb_image)

        img_padded, padding_info = image_utils.pad_test_image(rgb_image)
        image_blocks = image_utils.get_test_blocks(img_padded, padding_info)


        # Lossy image blocks compression through encoder network
        compressed_blocks=compression_utils.compress_image(compression_net, image_blocks)
        compressed_blocks_shape = np.asarray(compressed_blocks).shape
        
        error_informations = autoencoder_error_evaluation(rgb_image, compressed_blocks,
                                                         decompression_net, width_immagine_originale, 
                                                         height_immagine_originale, padding_info,
                                                         p=False)
        error_informations = error_informations [2:6]
        validation_set_errors.append(error_informations)
        
    return validation_set_errors
        
    
    
def get_psnr(original, decompressed):
    h = original.shape[0]
    w = original.shape[1]

    mse = 0
    for r in range(h):
        for c in range(w):
            for v in range(3):
                mse += (original[r, c, v]-decompressed[r, c, v])**2

    mse = mse/float(h*w)
    arg = 1/(mse**(1/2))
    return 20*math.log10(arg)



def estimate_psnr_and_compr_ratio(image_paths, compression_net, decompression_net):
    psnrs = [[], [], []]
    compression_ratios = [[], [], []]
    rgb_images = []
    
    for idx in range (len(image_paths)):
        rgb_image, alpha_channel = image_utils.load_image(image_paths[idx])
        rgb_images.append(rgb_image)
    
    
    
    for quality in range(1, 101):
        for idx in range (len(rgb_images)):
            rgb_image = rgb_images[idx]
            compression_code = compressor.compress(image_paths[idx], compression_net, decompression_net, quality, False)
            
        
            compression_ratios[idx].append(evaluate_compression_ratio(rgb_image, alpha_channel, 
                                                                      compression_code, False))
            
            decompressor.decompress(compression_code, decompression_net, False)
            time.sleep(0.250)
            err_corrected_image = image_utils.load_image("predicted/decompression_error_corrected.png")[0]


            psnrs[idx].append(get_psnr(rgb_image, err_corrected_image [:,:,:3]))      
      
        print("PSNRs and compression ratios with error correction quality " + str(quality) + " estimated")
        
    return psnrs, compression_ratios