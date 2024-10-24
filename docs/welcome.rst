Welcome to MEEGNet!
===================

.. image:: https://img.shields.io/pypi/v/meegnet.svg
   :target: https://pypi.org/project/meegnet/
   
.. image:: https://img.shields.io/pypi/pyversions/meegnet.svg
   :target: https://pypi.org/project/meegnet/

MEEGNet is an open-source Python toolbox for neuroscientists interested in using Artificial Neural Networks (ANNs) and more specifically Convolutional Neural Networks (CNNs) for Magnetoencephalography (MEG) and Electroencephalography (EEG) data analysis. Our library focuses on providing tools for interpretability and visualization of latent space, making ANNs more transparent.

- **Source:** https://github.com/arthurdehgan/meegnet
- **Bug Reports:** https://github.com/arthurdehgan/meegnet/issues


Key Features
============

* Dataset Management: Easily manage and preprocess MEG and EEG datasets using our custom dataset objects.
* Model Management: Easily define, train, and evaluate CNN models for MEG and EEG data using our custom model object.
* Model Explainability: Use our library to generate explanations for your CNN models, including saliency maps and feature importance metrics.
* Latent Space Visualization: Visualize and explore the latent space of your CNN models using our custom visualization tools.
* Pre-trained Architectures: Access pre-trained CNN architectures through Hugging Face and easily fine-tune them for your specific use case.
* Tutorials and Examples: Learn how to use the library with our extensive suite of tutorials and example scripts.

Future Features
---------------

* BIDS compatibility
* LF-CNN 
* VAR-CNN 

MeegNet Architecture
--------------------

.. image:: https://github.com/arthurdehgan/meegnet/blob/master/architecture.png

Other Available Architectures
-----------------------------

The package currently supports the following architectures: 

* EEGNet 
* VGG16
* MLP
* Custom CNN architectures

Alternatives
============

Maybe this package doesn’t suit your needs, in which case we can recommend similar packages with similar goals: 

* https://mneflow.readthedocs.io/en/latest/
* https://braindecode.org/stable/index.html
