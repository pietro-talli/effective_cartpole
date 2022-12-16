# Semantic and Effective Communication with Dynamic Feature Compression

This repo containes the code of the work "Semantic and Effective Communication with Dynamic Feature Compression".

## Code usage

### Obtain the ensamble of VQ-VAEs
Run:
```
python vq_vae_training.py
```
to collect the observation dataset and create a VQ-VAE. The codebook obtained will contain 64 codewords.

To obtain multiple quantizers from an existing Encoder-Decoder pair run:
```
python new_quantizer.py
```
This will create a se of new quantizer with respectively 32, 16, 8, 4 and 2 codewords. 

