Welcome to MEEGNet!
===================

.. image:: https://img.shields.io/pypi/v/meegnet.svg
   :target: https://pypi.org/project/meegnet/
   
.. image:: https://img.shields.io/pypi/pyversions/meegnet.svg
   :target: https://pypi.org/project/meegnet/

This is a suite of tools made available for neuroscience researchers to use a variety of M/EEG neural network architectures on their data.

- **Source:** https://github.com/byuccl/meegnet
- **Bug Reports:** https://github.com/byuccl/meegnet/issues


MeegNet Architecture
--------------------

.. image:: https://github.com/arthurdehgan/meegnet/blob/master/architecture.png

Other Available Architectures
-----------------------------

The package currently supports the following architectures: 

* LF-CNN 
* VAR-CNN 
* EEGNet 

Features
========

Work in Progress

Tutorials
=========

Jupyter Notebook tutorials:
---------------------------

Prepare your data by following the instructions
`here <https://github.com/arthurdehgan/meegnet/blob/master/notebooks/Prepare%20Data%20Tutorial.ipynb>`__

Learn the basics of how to train and evaluate using a pre-made network
`here <https://github.com/arthurdehgan/meegnet/blob/master/notebooks/Meegnet%20Network%20Training%20Basic%20Tutorial.ipynb>`__

Alternatives
============

Maybe this package doesn’t suit your needs, in which case I can
recommend similar packages with similar goals: -
https://mneflow.readthedocs.io/en/latest/ -
https://braindecode.org/stable/index.html

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
