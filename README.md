This is the repository for the paper [Shear Wave Splitting Analysis using Deep Learning (SWSNet)](http://dx.doi.org/10.22541/essoar.168691708.89597483/v2)

Here we introduce a baseline deep learning model SWSNet that has the potential to replace grid search methods used by previous studies to find splitting parameters 
for a waveform. Due to the dearth of labelled real data we train the model on synthetic data and use a novel deconvolution approach to minimise the difference between 
real and synthetic data.

The codes for simulating synthetic data for a single anisotropic layer can be found in simulation.py and for a 2-layer setting can be found in simulation_2_layer.py.
The codes for data deconvolution are there in deconvolution.py

Example notebooks can be found in the notebooks folder. 

The keras model for SWSNet is saved in h5 format in the models folder.
