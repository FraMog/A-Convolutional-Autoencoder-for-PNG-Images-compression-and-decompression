import lzma
import numpy as np


def compression(array):
    
    byte_array = np.asarray(array).tobytes()
    byte_array_lz = lzma.compress(
                    byte_array,
                    format=lzma.FORMAT_RAW,
                    filters=[{'id': lzma.FILTER_LZMA2, 'preset': 9 | lzma.PRESET_EXTREME}]
                    )
    return byte_array_lz   


def decompression(byte_array_lz, original_shape):
    
    byte_array= lzma.decompress(
                                byte_array_lz,
                                format=lzma.FORMAT_RAW,
                                filters=[{'id': lzma.FILTER_LZMA2, 'preset': 9 | lzma.PRESET_EXTREME}]
                                )
        
    float_array = np.frombuffer(byte_array, dtype=np.float32)
    
    return float_array