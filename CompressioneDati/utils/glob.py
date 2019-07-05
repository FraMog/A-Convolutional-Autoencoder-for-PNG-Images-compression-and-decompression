from .LZ_78 import LZ_78_compressor
from .LZ_78 import LZ_78_decompressor

global block_size
global frame_size
global lz_decompressor
global lz_compressor

block_size = 128
frame_size = 15
lz_decompressor= LZ_78_decompressor()
lz_compressor= LZ_78_compressor()



