# Identification of Glomerulosclerosis Using IBM Watson And Shallow Neural Networks
Repo hosting the source code for the paper "Identification of Glomerulosclerosis Using IBM Watson And Shallow Neural Networks"

This directory contains:
1. _getROIs.m_: Script for creating the knowledge base: staring from svs files, positive and negative folders with samples are created based on the annotations contained in the original file.
2. _textureFeatures.m_: Script for extracting textural features from the processed images.
3. _createBowmanMaskNoCentroids.m_: Script for creating the masks of the bowman capsule.
4. _extractBowmanFeatures.m_: Script for creating features from the bowman capsule.
5. _kfoldtrainingsingle.m_: Script for loading the dataset, filter the features in the PCA space, train the ANN with the optimal naumber of neurons in the hidden layer.

The remaining 3 Matlab files are functions called by the previous scripts.
