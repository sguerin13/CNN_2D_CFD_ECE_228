# ECE-228-Project

Group Project for UCSD ECE 228 - Predicting 2D Steady state fluid flow fields with Convolutional Neural Networks.

Three different styles of architectures were examined:

- Feed Forward Encoder-Decoder:

- ResNet:

- U-Net:

The network models were trained on data generated using the PyLBM library and consisted of open fluid domains with elementary objects included in the fluid domain. 

To run the Encoder-Decoder model, download the steady state data from https://drive.google.com/file/d/1KBQAjNUeEm7JcbpA7uK18aAsL7nDr75P/view?usp=sharing. 
Put the data in the same directory as the symmetrical_ae.py and run the script.

To run the UNET model, uncompress UNET.7z, mode the data into that folder, and run the train.py script inside.
To run the Resnet-18 model, uncompress ResNet.zip, mode the data into that folder, and run the train.py script inside.
