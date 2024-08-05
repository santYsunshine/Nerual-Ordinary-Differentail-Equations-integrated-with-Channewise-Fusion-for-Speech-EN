The santy_node.py is the about integrating the neural ordinary differential concept into the Speech denoising project.
TCN is the main core of the speech denoiser, it used a dilated 1-D convolution blocks to create a masking.
The Neural Ordinary differential Equations(NODEs) takes the TCN created mask as the input and process it.
