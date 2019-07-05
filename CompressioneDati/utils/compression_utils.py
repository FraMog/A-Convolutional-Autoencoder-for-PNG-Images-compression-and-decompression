import time
import numpy as np
import matplotlib.image as img
import lzma 

from . import image_utils
from . import glob


def compressor(image_path, encoder, decoder, quality):
    t0= time.clock()
    #Carica l'immagine PNG da comprimere, separando valori RGB (normalizzati) ed alpha channel.
    
    immagine_rgb, alpha_channel=image_utils.load_image(image_path)
    print("Elapsed: {}".format(time.clock()-t0))
    t0= time.clock()
    
    width_immagine_originale=len(immagine_rgb[0])
    height_immagine_originale=len(immagine_rgb)

    
    img_padded, padding_info = image_utils.pad_test_image(immagine_rgb)
    print("Elapsed: {}".format(time.clock()-t0))
    t0= time.clock()
    
    #Recupera i blocchi del test set secondo la strategia indicata in get_test_blocks
    image_blocks =image_utils.get_test_blocks(img_padded, padding_info)
    print("Elapsed: {}".format(time.clock()-t0))
    t0= time.clock()
    #Compressione dei blocchi
    compressed_blocks=compress_image(encoder, image_blocks)
    compressed_blocks_shape = np.asarray(compressed_blocks).shape
    print("Elapsed: {}".format(time.clock()-t0))
    t0= time.clock() 
    
    # Block compression with LZMA
    compressed_blocks_to_bytes= np.asarray(compressed_blocks).tobytes()
    compressed_blocks_lz = lzma.compress(
        compressed_blocks_to_bytes,
        format=lzma.FORMAT_RAW,
        filters=[{'id': lzma.FILTER_LZMA2, 'preset': 9 | lzma.PRESET_EXTREME}]
    )
    print("image_blocks_size " + str(number_of_bytes(image_blocks)) 
          + " compressed_blocks_comp " + str(len(compressed_blocks_lz)) )  
  

    
    #Decomprimo l'immagine per valutare l'errore di decompressione
    decompressed_image= decompress_image(decoder, compressed_blocks,
                                         padding_info, width_immagine_originale,
                                         height_immagine_originale)
    
    

    print("Elapsed: {}".format(time.clock()-t0))
    t0= time.clock()
    
    #Rileva e raccoglie gli errori più elevati, ovvero quelli che hanno un maggiore impatto visivo
    treshold = get_treshold(immagine_rgb, decompressed_image, quality)
    important_errors =  evaluate_error(immagine_rgb, decompressed_image, treshold)
    imp_errors_shape=important_errors.shape
    print(important_errors)
    print("Fine evaluate error Elapsed: {}".format(time.clock()-t0))
    t0= time.clock()
       
    #  compression using lzma python library
    important_errors_bytes= important_errors.tobytes()
    important_errors_comp = lzma.compress(
        important_errors_bytes,
        format=lzma.FORMAT_RAW,
        filters=[{'id': lzma.FILTER_LZMA2, 'preset': 9 | lzma.PRESET_EXTREME}]
    ) 
    err_to_send= important_errors_comp
    print("important_errors_size " + str(number_of_bytes(important_errors)) 
          + " err_to_send bytes " + str(len(err_to_send)) )  

    '''
    # important error compression using custom LZ78 functions
    error_stream=preprocess_important_errors(important_errors)
    glob.lz_compressor.initialize()
    glob.lz_decompressor.initialize()    
    glob.lz_compressor.compress(error_stream)
    err_to_send=None
    print("important_errors_size " + str(number_of_bytes(important_errors)))
    '''
    
    
    
    print("Fine compressione LZ Elapsed: {}".format(time.clock()-t0))
    print("FINE COMPRESSORE")
    
    
    decompressor(compressed_blocks_lz, compressed_blocks_shape,
                 err_to_send, imp_errors_shape,
                 padding_info, width_immagine_originale,
                 height_immagine_originale, alpha_channel,
                 decoder)
   
    
    
def decompressor(compressed_blocks_lz, compressed_blocks_shape,
                 err_received, imp_errors_shape,
                 padding_info, width_immagine_originale,
                 height_immagine_originale, alpha_channel,
                 decoder):
    print("Inizio decompressore ")
    t0= time.clock()
    
    
    
    # Block decompression with LZMA
   
    # important error compression using lzma python library
    compressed_blocks_byte= lzma.decompress(
        compressed_blocks_lz,
        format=lzma.FORMAT_RAW,
        filters=[{'id': lzma.FILTER_LZMA2, 'preset': 9 | lzma.PRESET_EXTREME}]
    )
    
    compressed_blocks_float = np.frombuffer(compressed_blocks_byte, dtype=np.float32)
    compressed_blocks= np.reshape(compressed_blocks_float, compressed_blocks_shape)
    
    print("End block decompression with lzma {}".format(time.clock()-t0))
    t0= time.clock()
    #Decomprimo l'immagine
    
    decompressed_image= decompress_image(decoder, compressed_blocks, padding_info, 
                                         width_immagine_originale, height_immagine_originale)
    
    
    
    decompressed_image_only_network=decompressed_image
    #Aggiungo l'alpha channel all'immagine ottenuta dalla decompressione effettuata dalla rete...
    if(alpha_channel is not None):
        decompressed_image_only_network = np.concatenate((decompressed_image_only_network, alpha_channel), axis=2)

    
    #... e salvo l'immagine per valutare graficamente le performance di decompressione della rete
    img.imsave("predicted/output_network.png", decompressed_image_only_network)
    print("Fine decompressione rete: {}".format(time.clock()-t0))
    t0= time.clock()
        
    #Successivamente effettuo la error correction, correggendo gli errori più evidenti graficamente . Ciò 
    # porterà ad un'immagine molto più simile a quella originale. Nei piccoli dettagli composti da pochi pixel (zoommando sia 
    # decompressed_image_only_network che decompressed_image_migliorata è possibile notare una notevole differenza).
    
    important_errors_comp= err_received
    # important error compression using lzma python library
    imp_errs_decomp_byte= lzma.decompress(
        important_errors_comp,
        format=lzma.FORMAT_RAW,
        filters=[{'id': lzma.FILTER_LZMA2, 'preset': 9 | lzma.PRESET_EXTREME}]
    )
    
    imp_errs_decomp_float = np.frombuffer(imp_errs_decomp_byte, dtype=np.float32)
    important_errors= np.reshape(imp_errs_decomp_float, imp_errors_shape)
    
    '''
    # important error decompression using custom LZ78 functions
    imp_errs_decomp_str=glob.lz_decompressor.get_reconstructed_word()
    
    #postprocess
    '''
    
    
    print(important_errors)
    print("End decompressed important errors {}".format(time.clock()-t0))
    t0= time.clock()
    
    
    decompressed_image_migliorata = migliora_immagine(decompressed_image, important_errors)
    print("Fine miglioramento immagine: {}".format(time.clock()-t0))
    t0= time.clock()
   
    if(alpha_channel is not None):
        decompressed_image_migliorata = np.concatenate((decompressed_image_migliorata, alpha_channel), axis=2)

    
    img.imsave("predicted/decompression.png", decompressed_image_migliorata)
    
    print("Fine decompressore Elapsed: {}".format(time.clock()-t0))
    t0= time.clock()
    
    
def compress_image(compression_net, image_blocks):
    #I blocchi 128x128x3 sono compressi in 32x32x16
    compressed_blocks = []
    for i in range (0, len(image_blocks)):
        compressed_block= compression_net.predict(np.asarray([image_blocks[i]]))
        compressed_blocks.append(compressed_block)

    return compressed_blocks 


    
#USATO DAL COMPRESSORE PER STIMARE L'ERRORE DI DECOMPRESSIONE, E DAL DECOMPRESSORE per decomprimere
def decompress_image(decompression_net, compressed_blocks,
                     padding_info, width_immagine_originale,
                     height_immagine_originale):
    
    #Le prime/ultime 10 (ovvero size_cornice) righe/colonne non saranno incluse nell'array (sono certo che siano solo valori di padding). 
    # Le esludo a priori. Questo è il motivo del  -size_cornice * 2 che si ha nelle dimensioni.
    width_decompressed_image = (padding_info.size_pad_cols_left +
                                width_immagine_originale +
                                padding_info.size_pad_cols_right -
                                glob.frame_size * 2)
                
    height_decompressed_image = (padding_info.size_pad_cols_top +
                                 height_immagine_originale +
                                 padding_info.size_pad_cols_bottom -
                                 glob.frame_size * 2)
    
    #Inizializzo la matrice che conterrà l'output di tutte le decompressioni
    decompressed_image=np.zeros((height_decompressed_image,width_decompressed_image,3), dtype=np.float32)
 
    passo= glob.block_size - (glob.frame_size * 2)
    for row in range (0, padding_info.number_of_vertical_blocks):
        for col in range (0, padding_info.number_of_horizontal_blocks):
           
            to_predict = np.asarray(compressed_blocks[row * padding_info.number_of_horizontal_blocks + col])
            decompressed_block = decompression_net.predict(to_predict)
                                            
            decompressed_block=decompressed_block[0]
            
            #Organizzo gli output della predizione del singolo blocco salvando solo il quadrato 108x108x3 
            last_index = glob.block_size-glob.frame_size
            decompressed_block=decompressed_block[glob.frame_size:last_index, glob.frame_size:last_index,:]
        
            #Posiziono opportunamente all'interno dell'immagine il blocco 108x108x3 ottenuto dall'istruzione precedente
            # avanzo a scatti di passo (secondo la strategia spiegata in pad_test_image)
            start_row_index= row * passo
            start_col_index= col * passo
            final_row_index = start_row_index+passo
            final_col_index = start_col_index+passo
            decompressed_image[start_row_index:final_row_index, start_col_index:final_col_index, :] = decompressed_block
            
    #Tolgo i due pad, tenendo presente che le 10 righe/colonne iniziali/finali dell'immagine paddata sono state già escluse 
    # da  decompressed_block= decompression_net.predict(np.asarray(compressed_blocks[row * number_of_horizontal_blocks + col]))
    # e dunque non vanno escluse nuovamente. (Questa è il motivo di tutte le presenze di size_cornice nella istruzione seguente)
    h_decompressed_img = len(decompressed_image)
    w_decompressed_img = len(decompressed_image[0])
    first_top_idx = padding_info.size_pad_cols_top - glob.frame_size
    first_bottom_idx = h_decompressed_img - padding_info.size_pad_cols_bottom + glob.frame_size
    first_left_idx = padding_info.size_pad_cols_left - glob.frame_size
    first_right_idx = w_decompressed_img - padding_info.size_pad_cols_right + glob.frame_size
    
    
    decompressed_image=decompressed_image[first_top_idx:first_bottom_idx, first_left_idx:first_right_idx, :]
    return decompressed_image


# Valuta gli errori di compressione in ogni canale di ogni pixel. Crea tre array, uno per gli errori più rilevanti nel canale rosso,
# uno per gli errori più rilevanti del verde, uno per gli errori più rilevanti del blu.
def evaluate_error(immagine, decompressed_image, threshold):
    
    height = len(immagine)
    width = len(immagine[0])
    
    
    #First channel for R error value. Second channel for G error value. Third channel fo B error value. 
    #Fourth channen for pixel's row index . Fifth channel for pixel's col index.
    error_matrix = np.zeros((height, width, 5), dtype=np.float32)
    
    #Calcolo l'errore per ogni pixel dell'immagine
    error_matrix[:,:,0:3] = decompressed_image - immagine

    
    #Aggiungo all'error_matrix informazioni su riga e colonna.
    
    #Inizializzo il quarto channel, che conterrà, per ogni pixel, l'indice della riga al quale il pixel appartiene
    channel_for_row_information= np.zeros((len(immagine), len(immagine[0])), dtype=int)
    t0= time.clock()
    # Per l'iesima riga devo craere un array di size 
    for i in range(height):    
        i_row= np.full((width), i)
        channel_for_row_information[i, :] = i_row
    print("T elapsed = {}".format(time.clock() - t0))
    t0 = time.clock()
    
    #Inizializzo il quarto channel, che conterrà, per ogni pixel, l'indice della colonna al quale appartiene
    channel_for_col_information= np.zeros((len(immagine), len(immagine[0])), dtype=int)
    for j in range(width):    
        j_col= np.full(height, j)
        channel_for_col_information[:, j] = j_col
    print("T elapsed = {}".format(time.clock() - t0))
 
    
    #Inserisco i due canali costruiti
    error_matrix[:,:,3]= channel_for_row_information
    error_matrix[:,:,4]= channel_for_col_information
    
    #error_matrix ha ora shape n_rows, n_cols, n_channels. I canali sono ora 5: error_red, error_green, error_blue, row_index, col_index.                            
    important_errors= []
    
    error_matrix=error_matrix.reshape(error_matrix.shape[0] * error_matrix.shape[1],5)
   
    
    r1= error_matrix [error_matrix[:,0] >= threshold]
    error_matrix=error_matrix [error_matrix[:,0] < threshold]
    
    r2 = error_matrix [error_matrix[:,0] <= -threshold]
    error_matrix=error_matrix [error_matrix[:,0] > -threshold]
    
    g1= error_matrix [error_matrix[:,1] >= threshold]
    error_matrix=error_matrix [error_matrix[:,1] < threshold]
    
    g2 = error_matrix [error_matrix[:,1] <= -threshold]
    error_matrix=error_matrix [error_matrix[:,1] > -threshold]
    
    b1= error_matrix [error_matrix[:,2] >= threshold]
    error_matrix=error_matrix [error_matrix[:,2] < threshold]
    
    b2 = error_matrix [error_matrix[:,2] <= -threshold]

    important_errors= np.concatenate((r1, r2), axis=0)
    important_errors= np.concatenate((important_errors, g1), axis=0)
    important_errors= np.concatenate((important_errors, g2), axis=0)
    important_errors= np.concatenate((important_errors, b1), axis=0)
    important_errors= np.concatenate((important_errors, b2), axis=0)     
    
    print("Fine calcolo errore T elapsed = {}".format(time.clock() - t0))
    t0 = time.clock()
    
    
    
    print("Shape important errors " + str(np.asarray(important_errors).shape))
    return np.asarray(important_errors)
    
    
#Usato dal decompressore per la error_correction
def migliora_immagine(decompressed_image, important_errors):
    for i in range (0, len(important_errors)):
        row_index= int(important_errors[i][3])
        col_index= int(important_errors[i][4])
        
        decompressed_image[row_index, col_index, :] -= important_errors[i,0:3]
    
    return decompressed_image


        
def preprocess_important_errors(important_errors):
    error_stream = ""

    for idx, elem in enumerate(important_errors):
        
        current_string =('{:.2f}'.format(elem[0]) +"*" +
                        '{:.2f}'.format(elem[0]) +"*"+
                        '{:.2f}'.format(elem[0]) +"*"+
                        str(int(elem[3]))+"*"+str(int(elem[4])))
        
        error_stream += ('#'+current_string)
       
    return error_stream[1:]

    
def evaluate_compression_ratio(image_path, encoder, decoder):
    #Carica l'immagine PNG da comprimere, togliendo alpha channel.
    immagine_rgb, alpha_channel=image_utils.load_image(image_path)
    
    width_immagine_originale=len(immagine_rgb[0])
    height_immagine_originale=len(immagine_rgb)
    
    #Effettua il padding, ed ottiene infomarzioni sul numero di blocchi orizzontali
    #e verticali che suddividono la img_padded in una griglia. Ottiene inoltre informazioni sull'ampiezza
    # del padding in ognuno dei 4 lati.
    img_padded, padding_info = image_utils.pad_test_image(immagine_rgb, glob.block_size)
    
    
    image_blocks =image_utils.get_test_blocks(img_padded, padding_info)
    
    
    compressed_blocks=compress_image(encoder, image_blocks)
    
    
    #Decomprimo l'immagine per valutare l'errore di decompressione
    decompressed_image= decompress_image(decoder, compressed_blocks, 
                                         padding_info, width_immagine_originale, 
                                         height_immagine_originale)
    
    #Rileva e raccoglie gli errori più elevati, ovvero quelli che hanno un maggiore impatto visivo
    important_errors =  evaluate_error(immagine_rgb, decompressed_image)

    


    print("shape compressed_blocks " + str(np.asarray(compressed_blocks).shape))
    print(type(compressed_blocks[0][0][0][0][0]))
    size_compressed_blocks= number_of_bytes(compressed_blocks)
    print("size_compressed_blocks " + str(size_compressed_blocks) +
          " shape compressed_blocks " + str(np.asarray(compressed_blocks).shape) )
    print()
    
    size_important_errors= number_of_bytes(important_errors)
    print(type(important_errors[0,0]))
    print("size_important_errors " + str(size_important_errors) + 
          " shape important_errors " + str(np.asarray(important_errors).shape) )
    print()
    
    if(alpha_channel is not None):
        size_alpha_channel= number_of_bytes(alpha_channel)
        print("size_alpha_channel " + str(size_alpha_channel) + 
              " shape alpha_channel " + str(np.asarray(alpha_channel).shape) )
        print(type(alpha_channel[0][0][0]))
        print()
    else: 
        size_alpha_channel=0
    
    
    size_immagine_rgb= number_of_bytes(immagine_rgb)
    print("size_immagine_rgb " + str(size_immagine_rgb) + 
          " shape immagine_rgb " + str(np.asarray(immagine_rgb).shape) )
    print(type(image_blocks[0][0][0][0]))
    print()
    
    
    
    compressed_size=size_compressed_blocks + size_important_errors + size_alpha_channel
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


def get_treshold(im1, im2, quality, n_samples=10000)
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