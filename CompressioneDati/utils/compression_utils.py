import time
import numpy as np
import matplotlib.image as img
from . import custom_lzma 
from . import image_utils
from . import glob


                                 
# Compress image blocks through a compression network                                  
def compress_image(compression_net, image_blocks):
    compressed_blocks = []
    for i in range (0, len(image_blocks)):
        compressed_block= compression_net.predict(np.asarray([image_blocks[i]]))
        compressed_blocks.append(compressed_block)

    return compressed_blocks 


'''
Decompress image blocks through a decompression network. 
Blocks are then reshaped to form the decompressed image. Lastly, image padding is deleted.
'''  
def image_decompression_with_autoencoder(decompression_net, compressed_blocks,
                                         padding_info, width_immagine_originale,
                                         height_immagine_originale, p=True):
    
    width_decompressed_image = (padding_info.size_pad_left +
                                width_immagine_originale +
                                padding_info.size_pad_right -
                                glob.frame_size * 2)
                
    height_decompressed_image = (padding_info.size_pad_top +
                                 height_immagine_originale +
                                 padding_info.size_pad_bottom -
                                 glob.frame_size * 2)
    
    #Initialize empty decompressed_image    
    net_decompressed_image = np.zeros((height_decompressed_image,width_decompressed_image,3), dtype=np.float32)
    
    net_decompressed_image = predict_original_blocks(decompression_net, padding_info, 
                                                        compressed_blocks, net_decompressed_image)
            
    
    return net_decompressed_image


def predict_original_blocks(decompression_net, padding_info, compressed_blocks, decompressed_image):
    step = glob.block_size - (glob.frame_size * 2)
    #Predict original blocks and position them properly                             
    for row in range (0, padding_info.number_of_vertical_blocks):
        for col in range (0, padding_info.number_of_horizontal_blocks):
            
            to_predict = np.asarray(compressed_blocks[row * padding_info.number_of_horizontal_blocks + col])
            decompressed_block = decompression_net.predict(to_predict)                                      
            decompressed_block=decompressed_block[0]
            
            last_index = glob.block_size-glob.frame_size
            decompressed_block=decompressed_block[glob.frame_size:last_index, glob.frame_size:last_index,:]
            #Decompressed block is positioned inside decompressed image properly
            start_row_index= row * step
            start_col_index= col * step
            final_row_index = start_row_index+step
            final_col_index = start_col_index+step
            decompressed_image[start_row_index:final_row_index, start_col_index:final_col_index, :] = decompressed_block
            
    return decompressed_image


#Compute relevant errors
def compute_relevant_errors(immagine, autoencoder_decompressed_img, threshold, p=True):
    
    height = len(immagine)
    width = len(immagine[0])
    
    #First channel for R error value. Second channel for G error value. Third channel for B error value. 
    #Fourth channen for pixel's row index . Fifth channel for pixel's col index.
    error_matrix = np.zeros((height, width, 5), dtype=np.float32)
    
    #Error computed
    error_matrix[:,:,0:3] = autoencoder_decompressed_img - immagine

    #Decore matrix with row and col information
                                  
    #Row decoration
    channel_for_row_information= np.zeros((len(immagine), len(immagine[0])), dtype=int)
    t0= time.clock()

    for i in range(height):    
        i_row= np.full((width), i)
        channel_for_row_information[i, :] = i_row
   
    #Col decoration
    channel_for_col_information= np.zeros((len(immagine), len(immagine[0])), dtype=int)
    for j in range(width):    
        j_col= np.full(height, j)
        channel_for_col_information[:, j] = j_col
 
    error_matrix[:,:,3]= channel_for_row_information
    error_matrix[:,:,4]= channel_for_col_information
    
                         
    important_errors= []
    error_matrix=error_matrix.reshape(error_matrix.shape[0] * error_matrix.shape[1],5)
   
    r1= error_matrix [error_matrix[:,0] >= threshold]
    error_matrix=error_matrix [error_matrix[:,0] <= threshold]
    r2 = error_matrix [error_matrix[:,0] <= -threshold]
    error_matrix=error_matrix [error_matrix[:,0] >= -threshold]
    g1= error_matrix [error_matrix[:,1] >= threshold]
    error_matrix=error_matrix [error_matrix[:,1] <= threshold]
    g2 = error_matrix [error_matrix[:,1] <= -threshold]
    error_matrix=error_matrix [error_matrix[:,1] >= -threshold]
    b1= error_matrix [error_matrix[:,2] >= threshold]
    error_matrix=error_matrix [error_matrix[:,2] <= threshold]
    b2 = error_matrix [error_matrix[:,2] <= -threshold]

    important_errors= np.concatenate((r1, r2), axis=0)
    important_errors= np.concatenate((important_errors, g1), axis=0)
    important_errors= np.concatenate((important_errors, g2), axis=0)
    important_errors= np.concatenate((important_errors, b1), axis=0)
    important_errors= np.concatenate((important_errors, b2), axis=0)    
    
    return important_errors
 
                                  
#Error correction
def correct_errors(decompressed_image, important_errors):
    for i in range (0, len(important_errors)):
        row_index= int(important_errors[i][3])
        col_index= int(important_errors[i][4])   
        decompressed_image[row_index, col_index, :] -= important_errors[i,0:3]
    
    return decompressed_image


def get_treshold(im1, im2, quality, n_samples=10000):
    h = im1.shape[0]
    w = im1.shape[1]

    random_indices = list(zip(np.random.choice(h, n_samples), np.random.choice(w, n_samples)))

    errors = []
    for indices in random_indices:
        px_or = im1[indices[0], indices[1]]
        px_ou = im2[indices[0], indices[1]]
        diff = px_or - px_ou
        errors.extend(diff.tolist())
    errors = [abs(x) for x in errors]
    errors.sort()
    treshold = np.percentile(errors, quality)
    return treshold 