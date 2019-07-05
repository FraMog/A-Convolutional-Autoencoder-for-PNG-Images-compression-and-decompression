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
def decompress_image(decompression_net, compressed_blocks,
                     padding_info, width_immagine_originale,
                     height_immagine_originale, p=True):
    
    width_decompressed_image = (padding_info.size_pad_cols_left +
                                width_immagine_originale +
                                padding_info.size_pad_cols_right -
                                glob.frame_size * 2)
                
    height_decompressed_image = (padding_info.size_pad_cols_top +
                                 height_immagine_originale +
                                 padding_info.size_pad_cols_bottom -
                                 glob.frame_size * 2)
    
    #Initialize empty decompressed_image    
    decompressed_image=np.zeros((height_decompressed_image,width_decompressed_image,3), dtype=np.float32)
    passo= glob.block_size - (glob.frame_size * 2)
                                  
    for row in range (0, padding_info.number_of_vertical_blocks):
        for col in range (0, padding_info.number_of_horizontal_blocks):
            
            to_predict = np.asarray(compressed_blocks[row * padding_info.number_of_horizontal_blocks + col])
            decompressed_block = decompression_net.predict(to_predict)                                      
            decompressed_block=decompressed_block[0]
            
            last_index = glob.block_size-glob.frame_size
            decompressed_block=decompressed_block[glob.frame_size:last_index, glob.frame_size:last_index,:]
            #Decompressed block is positioned inside decompressed image properly
            start_row_index= row * passo
            start_col_index= col * passo
            final_row_index = start_row_index+passo
            final_col_index = start_col_index+passo
            decompressed_image[start_row_index:final_row_index, start_col_index:final_col_index, :] = decompressed_block
            
    #Decompressed image is unpadded
    h_decompressed_img = len(decompressed_image)
    w_decompressed_img = len(decompressed_image[0])
    first_top_idx = padding_info.size_pad_cols_top - glob.frame_size
    first_bottom_idx = h_decompressed_img - padding_info.size_pad_cols_bottom + glob.frame_size
    first_left_idx = padding_info.size_pad_cols_left - glob.frame_size
    first_right_idx = w_decompressed_img - padding_info.size_pad_cols_right + glob.frame_size
                                  
    decompressed_image=decompressed_image[first_top_idx:first_bottom_idx, first_left_idx:first_right_idx, :]
    
    return decompressed_image

                                  
                                  

#Compute relevant errors
def compute_relevant_errors(immagine, decompressed_image, threshold, p=True):
    
    height = len(immagine)
    width = len(immagine[0])
    
    #First channel for R error value. Second channel for G error value. Third channel for B error value. 
    #Fourth channen for pixel's row index . Fifth channel for pixel's col index.
    error_matrix = np.zeros((height, width, 5), dtype=np.float32)
    
    #Error computed
    error_matrix[:,:,0:3] = decompressed_image - immagine

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
def migliora_immagine(decompressed_image, important_errors):
    for i in range (0, len(important_errors)):
        row_index= int(important_errors[i][3])
        col_index= int(important_errors[i][4])   
        decompressed_image[row_index, col_index, :] -= important_errors[i,0:3]
    
    return decompressed_image


        
    
def evaluate_compression_ratio(image_path, encoder, decoder, quality, p=True):
    quality=100-quality
    t0= time.clock()
    #Carica l'immagine PNG da comprimere, separando valori RGB (normalizzati) ed alpha channel.
    
    immagine_rgb, alpha_channel=image_utils.load_image(image_path)
    if p:
        print("Elapsed: {}".format(time.clock()-t0))
    t0= time.clock()
    
    width_immagine_originale=len(immagine_rgb[0])
    height_immagine_originale=len(immagine_rgb)

    
    img_padded, padding_info = image_utils.pad_test_image(immagine_rgb)
    if p:
        print("Elapsed: {}".format(time.clock()-t0))
    t0= time.clock()
    
    #Recupera i blocchi del test set secondo la strategia indicata in get_test_blocks
    image_blocks =image_utils.get_test_blocks(img_padded, padding_info)
    if p:
        print("Elapsed: {}".format(time.clock()-t0))
    t0= time.clock()
    
    if alpha_channel is not None:
        #Compressione alpha channel attraverso LZMA
        alpha_channel_to_bytes= np.asarray(alpha_channel).tobytes()
        alpha_channel_lz = lzma.compress(
            alpha_channel_to_bytes,
            format=lzma.FORMAT_RAW,
            filters=[{'id': lzma.FILTER_LZMA2, 'preset': 9 | lzma.PRESET_EXTREME}]
        )
    else: 
        alpha_channel_lz=None

    #Compressione dei blocchi attraverso la rete
    compressed_blocks=compress_image(encoder, image_blocks)
    compressed_blocks_shape = np.asarray(compressed_blocks).shape
    if p:
        print("Elapsed: {}".format(time.clock()-t0))
    t0= time.clock() 
    
    # Ulteriore compressione dei blocchi attraverso LZMA
    compressed_blocks_to_bytes= np.asarray(compressed_blocks).tobytes()
    compressed_blocks_lz = lzma.compress(
        compressed_blocks_to_bytes,
        format=lzma.FORMAT_RAW,
        filters=[{'id': lzma.FILTER_LZMA2, 'preset': 9 | lzma.PRESET_EXTREME}]
    )
    if p:
        print("image_blocks_size " + str(number_of_bytes(image_blocks)) 
          + " compressed_blocks_comp " + str(len(compressed_blocks_lz)) )  
  

    
    #Decomprimo l'immagine per valutare l'errore di decompressione
    decompressed_image= decompress_image(decoder, compressed_blocks,
                                         padding_info, width_immagine_originale,
                                         height_immagine_originale, p)

    if p:
        print("Elapsed: {}".format(time.clock()-t0))
    t0= time.clock()
    
    #Rileva e raccoglie gli errori più elevati, ovvero quelli che hanno un maggiore impatto visivo
    treshold = get_treshold(immagine_rgb, decompressed_image, quality)
    important_errors = evaluate_error(immagine_rgb, decompressed_image, treshold, p)
    
    imp_errors_shape=important_errors.shape
    if p:
        print(important_errors)
        print("Fine evaluate error Elapsed: {}".format(time.clock()-t0))
    t0= time.clock()
       
    #  compression using lzma python library
    important_errors_bytes= important_errors.tobytes()
    important_errors_lz = lzma.compress(
        important_errors_bytes,
        format=lzma.FORMAT_RAW,
        filters=[{'id': lzma.FILTER_LZMA2, 'preset': 9 | lzma.PRESET_EXTREME}]
    ) 
    err_to_send= important_errors_lz
    
    

    


    if p:
        print("shape compressed_blocks " + str(np.asarray(compressed_blocks).shape))
        print(type(compressed_blocks[0][0][0][0][0]))
    size_compressed_blocks_lz= len(compressed_blocks_lz)
    if p:
        print("size_compressed_blocks_lz " + str(size_compressed_blocks_lz) +
          " shape compressed_blocks " + str(np.asarray(compressed_blocks).shape) )
        print()
    
    size_important_errors_lz= len(important_errors_lz)
    if p:
        print("size_important_errors " + str(size_important_errors_lz))
        print()
    
    if(alpha_channel is not None):
        size_alpha_channel= number_of_bytes(alpha_channel)
        if p:
            print("size_alpha_channel " + str(size_alpha_channel) + 
              " shape alpha_channel " + str(np.asarray(alpha_channel).shape) )
            print(type(alpha_channel[0][0][0]))
            print()
    else: 
        size_alpha_channel=0
        
        
    if (alpha_channel_lz is not None):
        size_alpha_channel_lz= len(size_alpha_channel_lz)
        if p:
            print("size_alpha_channel_lz " + str(size_alpha_channel_lz))
            print()
    else :
        size_alpha_channel_lz=0
    
    
    size_immagine_rgb= number_of_bytes(immagine_rgb)
    if p:
        print("size_immagine_rgb " + str(size_immagine_rgb) + 
          " shape immagine_rgb " + str(np.asarray(immagine_rgb).shape) )
        print(type(image_blocks[0][0][0][0]))
        print()
    
    
    
    compressed_size=size_compressed_blocks_lz  + size_important_errors_lz + size_alpha_channel_lz
    original_size=size_alpha_channel + size_immagine_rgb
    return original_size/compressed_size



def  number_of_bytes(data_structure):
    if (data_structure is None):
        return 0
    #Sono tutti float32
    size=1
    for i in range (0, len((np.asarray(data_structure)).shape)):
        size *= np.asarray(data_structure).shape[i]
    return 4 * size
  
    
def single_image_pred_error(net, image_path):
    image, alpha_channel=image_utils.load_image(image_path)
    image_width=len(image[0])
    image_height=len(image)
    
    img_padded = image_utils.pad_test_image(image,128)[0]
    
    
    

    #Organizzo i blocchi come impostato in get_test_blocks
    number_of_horizontal_blocks= int(img_padded.shape[1]/64) - 1
    number_of_vertical_blocks= int(img_padded.shape[0]/64) - 1 
    

    
    nz_img=img_padded
    test_blocks=image_utils.get_test_blocks(nz_img)

    predicted_image=np.zeros((img_padded.shape[0]-64,img_padded.shape[1]-64,3), dtype=np.float32)
  

    for row in range (0, number_of_vertical_blocks):
        for col in range (0, number_of_horizontal_blocks):
            start_row_index= row * 64 
            start_col_index= col * 64
            
            #Organizzo gli output della predizione del singolo blocco salvando solo il quadrarto 64x64x3 centrale, come indicato nella funzione
            #get_test_blocks
            pred= get_prediction_new_block(row * number_of_horizontal_blocks + col, net, test_blocks)
            #print("shape pred " + str(pred.shape))
            pred=pred[0]
            #print("shape pred " + str(pred.shape))
            pred=pred[32:96,32:96,:]
            predicted_image[start_row_index:start_row_index+64, start_col_index:start_col_index+64, :]=pred
            
    print("predicted_image shape " + str(predicted_image.shape))     
    '''Elimino il padding
    1) Cancello i 4 padding da 64 righe/colonne di pixel messi sopra/sotto/sinistra/destra dell'intera immagine nel metodo pad_image. 
    Da ognuno dei 4 lati ho già escluso (vedi il commento precedente) le 32 righe/colonne più esterni. 
    '''
    
    predicted_image= predicted_image[32:len(predicted_image)-32,32:len(predicted_image[0])-32,:]
      
    '''
    2) Cancello ora il padding concatenatoc da Giovanni a destra ed in fondo all'immagine 
    '''
    
   
    predicted_image =predicted_image[0:image_height,0:image_width, :]
   
    
    print(type(predicted_image[0][0][0]))
    
    img.imsave("predicted/error_image.png", predicted_image)
    error_image = predicted_image - image
    error_image=error_image.reshape(-1)
    
   
    #Valore atteso. l'ho calcolato una volta sui valori reali dell'errore (sia positivi che negativi), ed un'altra volta sul valori assoluti
    #dell'errore.
    mu=np.mean(error_image)
    
    #Deviazione standard. l'ho calcolata una volta sui valori reali dell'errore (sia positivi che negativi), ed un'altra volta sul valori assoluti
    #dell'errore.
    std= np.std(error_image)
    
    print(error_image)
    print("Expected Value dell'errore sulla singola immagine " + image_path + " " + str(mu))
    print("Standard deviation dell'errore sulla singola immagine " + image_path + " " + str(std))

    print()
    error_image_absolute=np.absolute(error_image)
    mu_absolute=np.mean(error_image_absolute)
    std_absolute= np.std(error_image_absolute)
    
    print(error_image_absolute)
    print("Expected Value dell'errore sulla singola immagine " + image_path + " " + str(mu_absolute))
    print("Standard deviation dell'errore sulla singola immagine " + image_path + " " + str(std_absolute))

    return error_image, error_image_absolute 
    

    
def validation_set_pred_error(net):
    error_vector = single_image_pred_error(conv2,"dataset2_preloaded/valid/0.png")[0]
    for i in range (1,100):
        i_error_vector=single_image_pred_error(conv2,"dataset2_preloaded/valid/" + str(i) +".png")
        error_vector = np.concatenate((error_vector, i_error_vector), axis=0)
        
    error_vector_absolute=np.absolute(error_vector)
    #Valore atteso
    mu=np.mean(error_vector)
    #Deviazione standard
    std= np.std(error_vector)
    print("Expected Value complessivo dell'errore " + str(mu))
    print("Standard deviation complessiva dell'errore " + str(std))
    print(error_vector.shape)
    return error_vector



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