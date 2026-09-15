Welcome to MEEGNet!
===================

.. image:: https://img.shields.io/pypi/v/meegnet.svg
   :target: https://pypi.org/project/meegnet/

.. image:: https://img.shields.io/pypi/pyversions/meegnet.svg
   :target: https://pypi.org/project/meegnet/

MEEGNet is an open-source Python toolbox for neuroscientists interested in using Artificial Neural Networks (ANNs) and more specifically Convolutional Neural Networks (CNNs) for Magnetoencephalography (MEG) and Electroencephalography (EEG) data analysis. Our library focuses on providing tools for interpretability and visualization of latent space, making ANNs more transparent.

If you use MEEGNet in your research, please cite:
`Dehgan et al. (2025) <https://doi.org/10.1101/2025.03.20.644276>`_

- **Source:** https://github.com/arthurdehgan/meegnet
- **Bug Reports:** https://github.com/arthurdehgan/meegnet/issues
- **Documentation:** https://meegnet.readthedocs.io/en/latest/index.html


Installation
============

MEEGNet requires Python 3.12 or higher. Install the released version from PyPI:

.. code-block:: bash

   pip install meegnet

To install the latest development version from source:

.. code-block:: bash

   git clone https://github.com/arthurdehgan/meegnet.git
   cd meegnet
   poetry install

More installation options can be found in the `online documentation <https://meegnet.readthedocs.io/en/latest/index.html>`_.

Quick Start
===========

Once your MEG/EEG data has been preprocessed into trials (see the
`prepare_data.ipynb <https://github.com/arthurdehgan/meegnet/blob/master/notebooks/prepare_data.ipynb>`__
tutorial), loading data and training a network is done in a few lines:

.. code-block:: python

   from meegnet.dataloaders import EpochedDataset
   from meegnet.network import Model

   # Load preprocessed data (EpochedDataset expects data already cut into
   # trials; the RestDataset class creates the trials for you instead).
   dataset = EpochedDataset(
       sfreq=500,
       n_subjects=100,
       n_samples=100,
       sensortype='ALL',  # MAG GRAD GRAD
       lso=True,          # leave-subject-out data split
   )
   dataset.load('/path/to/data')

   # Create and train a model.
   # Architectures: 'eegnet', 'meegnet', 'vgg16', 'mlp'
   model = Model(
       'my_model',       # model name, also used as output file prefix
       'eegnet',
       input_size=(3, 102, 400),  # (channels, sensors, time points)
       n_outputs=2,
       save_path='./outputs',
   )
   model.train(dataset, max_epoch=15, verbose=1)

   # Inspect training curves and evaluate on the held-out set.
   model.plot_accuracy()
   model.test(dataset)

   # Load a pre-trained network for interpretability analysis.
   model.load('./outputs/my_model.pt')   # or model.from_pretrained()

   from meegnet.viz import compute_saliency_maps

   compute_saliency_maps(
       dataset, model.net, './outputs/saliency_maps',
       labels=dataset.target_labels, epoched=True,
   )

Grad-CAM and additional visualization tools are available in
`meegnet.viz`.

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

Reproducing the Paper Results
=============================

The step-by-step pipeline used to produce the paper results — data preparation,
training, and the subject-size experiment — is documented in
`scripts/README.rst <scripts/README.rst>`_ and run from ``scripts/``.

The precomputed saliency maps, pre-trained models, and per-experiment
participant tables used to generate the paper figures are published on
`Figshare <https://doi.org/10.6084/m9.figshare.33806332>`_. The figure
notebooks (``notebooks/visu_*``) can reproduce the paper figures without any
retraining; see the *Reproducing Figures Without Retraining* section of
`scripts/README.rst <scripts/README.rst>`_ for the exact steps.

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

.. image:: https://raw.githubusercontent.com/arthurdehgan/meegnet/master/architecture.png

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

Dehgan A, Pascarella A, Harel Y, Rish I, Jerbi K. MEEGNet: an open source python library for the application of convolutional neural networks to MEG. bioRxiv. 2025.
`link <https://doi.org/10.1101/2025.03.20.644276>`__

::

   @article{Dehgan2025,
       title = {{MEEGNet}: an open source python library for the application of convolutional neural networks to {MEG}},
       author = {Dehgan, Arthur and Pascarella, Annalisa and Harel, Yann and Rish, Irina and Jerbi, Karim},
       year = {2025},
       journal = {bioRxiv},
       doi = {10.1101/2025.03.20.644276}
   }

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