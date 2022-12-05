# Label-Propagation-ssl
Mnist get_data file randomly takes 10,000 samples as train data. 100 samples(10 for each class) are only used as annotated initial samples. For rest of 9900 samples we use semi-supervised label propagation to generate Labels(called pseudo labels).
lp_main takes the data and performs semi-supervised label propagation.
If you dont have enough resources(GPU/RAM) reduce the no of images in mnist_get_data.py
This is a simplified approach implemented using tensorflow 
For detailed implementation please see original code--> (Code for CVPR 2019 paper Label Propagation for Deep Semi-supervised Learning https://github.com/ahmetius/LP-DeepSSL)
