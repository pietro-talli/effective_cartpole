# Titolo del paper

This repo contains the code of the work "Titolo del paper". [Quantizer.py](https://github.com/pietro-talli/effective_cartpole/blob/main/Quantizer.py)
contains the main Neural Network models. While in [utils.py](https://github.com/pietro-talli/effective_cartpole/blob/main/utils.py) there are the main functions to train the models. In particular
- `train_sensor_policy_PSNR`: trains the Sensor policy with respect to the technical problem
- `train_sensor_policy_semantically`: trains the Sensor policy for the semantic problem 
- `train_sensor_policy_effectively`: trains the Sensor policy for the effectiveness problem 

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
This will create a set of new quantizer with respectively 32, 16, 8, 4 and 2 codewords. 

### Train the regressor (semantic task)
Run:
```
python train_the_regressor.py
```
to train a regressor to estimate the physical state values. A regression network is obtained for every quantizer. 

### Train the Controller (effective task)
To train the controller to learn a policy based on the encoded observations use:
```
python train_the_control_policy.py
```
This script trains a different policy for all the different number of codewords of the quantizers. 

### Train the Sensor Policy
To train the Sensor policy three examples are reported in [sensor_training_test.py](https://github.com/pietro-talli/effective_cartpole/blob/main/sensor_training_test.py). Run:
```
python sensor_training_test.py
```
to train the Sensor policy.
