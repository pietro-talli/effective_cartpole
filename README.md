# Semantic and Effective Communication with Dynamic Feature Compression

This repo containes the code of the work "Semantic and Effective Communication with Dynamic Feature Compression". [`Quantizer.py`](https://pages.github.com/)

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

### Train the regressor (semantic task)
Run:
```
python train_the_regressor.py
```
to train a regressor to estimate the physical state values. A regression network is obtained for every quantizer. 

### Train the Controller (effective task)
To train the controller to learn a policy based on the encoded observations use:
```

```
