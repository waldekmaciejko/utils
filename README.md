# utils
Various scripts for machine learning

## extract_mfcc_libri_pools.py
function for Libri speech audio pcm data base that creates dict:
keys - speaker label,
value - path to flac file
then creates callable class; during calling instance estimate MFCC
using librosa lib and does some other minor preprocessing steps like:
convert to mono, resampling to 16 kHz, trmming
then using multiprocessing lib start process-based parallelism


## pca_definition.py
function creates PCA algorithm using theoretical description



