import numpy as np

from . import glob

class LZ_78_decompressor:
    def __init__(self):
        return None
        
    def initialize(self):
        print("initialing LZ_78_decompressor")
        self.dict_size=8192  # uint8.
        self.dictionary= np.zeros((self.dict_size), dtype= object)
        self.dictionary [0] = "*"
        self.dictionary [1] = "-"
        self.dictionary [2] = "."
        self.dictionary [3] = "#"
        for i in range(4, self.dict_size):
            self.dictionary[i] = str(i-4)
        self.reconstructed_word=""
    
    #Uncompressed= stringa rappresentante important_errors preprocessato
    def decompress(self,match_index, character):
        word = self.dictionary[match_index] + character
       
        #Update heuristic
        self.dictionary[15:self.dict_size] = self.dictionary[14: self.dict_size-1]
        self.dictionary[14] = word
        
        
        self.reconstructed_word+=word
        
  
            
    def get_reconstructed_word(self):
        return self.reconstructed_word   
    
class LZ_78_compressor:

    def __init__(self):
        return None
    
    def initialize(self):
        print("initialing LZ_78_compressor")
        self.dict_size=8192  # uint8.
        self.dictionary= np.zeros((self.dict_size), dtype= object)
        self.dictionary [0] = "*"
        self.dictionary [1] = "-"
        self.dictionary [2] = "."
        self.dictionary [3] = "#"
        for i in range(4, self.dict_size):
            self.dictionary[i] = str(i-4)
        self.reconstructed_word="" 
    
    #Uncompressed= stringa rappresentante important_errors preprocessato
    def compress(self, uncompressed):
        tot = 0
        t=0
        w = ""
        
        compressed_values=0
        times=0
        l=len(uncompressed)
        
        for c in uncompressed:
         
            wc = w + c               
            if wc in self.dictionary:
                w = wc
            else:
                tot=tot+1
                #print ("w è " +  w + " con size " + str(len(w))  + " c è " + str(c))
                g=len(w)
                if (g>=2):
                    t=t+1
                times+=1
                compressed_values+=g+1
                
                if(times%1000==0):
                    print("TIMES " + str(times) + " ho compresso " + str(compressed_values) + " su " + str(l))
                match_index= np.where(self.dictionary == w)[0][0]
                
                #Send to decompressor.
                glob.lz_decompressor.decompress(match_index, c)
                
                
                #Update heuristic
                self.dictionary[15:self.dict_size] = self.dictionary[14: self.dict_size-1]
                self.dictionary[14] = wc
                w = ""

            
        if w:
            glob.lz_decompressor.decompress(np.where(self.dictionary == w)[0][0], "")
        print("dizionario finale di size " + str(len(self.dictionary)) + " " + str(self.dictionary) )
        print("Il numero di frasi di lunghezza > 2 è " +  str(t) 
              + " su " + str(tot))
        
        
        print(str(self.dictionary[:20]))
        print()
        print (str(uncompressed))
        