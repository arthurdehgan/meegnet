Welcome to MEEGNet!
===================

.. image:: https://img.shields.io/pypi/v/meegnet.svg
   :target: https://pypi.org/project/meegnet/

.. image:: https://img.shields.io/pypi/pyversions/meegnet.svg
   :target: https://pypi.org/project/meegnet/

MEEGNet is an open-source Python toolbox for neuroscientists interested in using Artificial Neural Networks (ANNs) and more specifically Convolutional Neural Networks (CNNs) for Magnetoencephalography (MEG) and Electroencephalography (EEG) data analysis. Our library focuses on providing tools for interpretability and visualization of latent space, making ANNs more transparent.

- **Source:** https://github.com/arthurdehgan/meegnet
- **Bug Reports:** https://github.com/arthurdehgan/meegnet/issues
- **Documentation:** https://meegnet.readthedocs.io/en/latest/index.html


Installation
============

.. code-block:: bash

   pip install meegnet

More installation options can be found in the `online documentation <https://meegnet.readthedocs.io/en/latest/index.html>`_.

Tutorials and Examples
======================

Prepare your data by following the instructions
`here <https://github.com/arthurdehgan/meegnet/blob/master/notebooks/prepare_data.ipynb>`__

Learn the basics of how to train and evaluate using a pre-made network
`here <https://github.com/arthurdehgan/meegnet/blob/master/notebooks/train_network.ipynb>`__

Generate saliency maps for your network
`here <https://github.com/arthurdehgan/meegnet/blob/master/notebooks/visu_saliency.ipynb>`__

Visualize latent space with Gradcam
`here <https://github.com/arthurdehgan/meegnet/blob/master/notebooks/visu_gradcam.ipynb>`__

Learn about your model using filter visualizations
`here <https://github.com/arthurdehgan/meegnet/blob/master/notebooks/visu_filters.ipynb>`__

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
* VAR-CNN and LF-CNN from Zubarev et al. (2019)

MEEGNet Architecture
--------------------

.. image:: https://github.com/arthurdehgan/meegnet/blob/master/architecture.png

Other Available Architectures
-----------------------------

The package currently supports the following architectures:

* MLP
* VGG-16
* EEGNet
* MEEGNet

License Information
===================

MEEGNet is released under the MIT License. This license is permissive, allowing you to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software. The full text of the license can be found in the `LICENSE` file in the repository.

By using MEEGNet, you agree to the terms of the MIT License. In summary, the MIT License allows you to:

- Use the software for any purpose, including commercial use.
- Modify and distribute the software, as long as the original copyright notice and license notice are included in all copies or substantial portions of the software.

For more details, please refer to the `LICENSE` file or visit the Open Source Initiative's `MIT License page <https://opensource.org/licenses/MIT>`_.

Alternatives
============

Maybe this package doesn’t suit your needs, in which case we can recommend similar packages with similar goals:

* https://mneflow.readthedocs.io/en/latest/
* https://braindecode.org/stable/index.html

References
==========

MEEGNet
-------

Work in Progress

LF-CNN or VAR-CNN
-----------------

Zubarev I, Zetter R, Halme HL, Parkkonen L. Adaptive neural network
classifier for decoding MEG signals. Neuroimage. 2019 May 4;197:425-434.
`link <https://www.sciencedirect.com/science/article/pii/S1053811919303544?via%3Dihub>`__

::

   @article{Zubarev2019AdaptiveSignals.,
       title = {{Adaptive neural network classifier for decoding MEG signals.}},
       year = {2019},
       journal = {NeuroImage},
       author = {Zubarev, Ivan and Zetter, Rasmus and Halme, Hanna-Leena and Parkkonen, Lauri},
       month = {5},
       pages = {425--434},
       volume = {197},
       url = {https://linkinghub.elsevier.com/retrieve/pii/S1053811919303544 http://www.ncbi.nlm.nih.gov/pubmed/31059799},
       doi = {10.1016/j.neuroimage.2019.04.068},
       issn = {1095-9572},
       pmid = {31059799},
       keywords = {Brain–computer interface, Convolutional neural network, Magnetoencephalography}
   }

EEGNet
------

::

   @article{Lawhern2018,
     author={Vernon J Lawhern and Amelia J Solon and Nicholas R Waytowich and Stephen M Gordon and Chou P Hung and Brent J Lance},
     title={EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces},
     journal={Journal of Neural Engineering},
     volume={15},
     number={5},
     pages={056013},
     url={http://stacks.iop.org/1741-2552/15/i=5/a=056013},
     year={2018}
   }
