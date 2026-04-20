Reproduce Paper Results
=======================

This folder holds scripts for reproducing paper results.
Run workflow from ``scripts/`` in this order:

1. Prepare CamCAN data.
2. Train network with paper config.
3. Switch to notebooks for visualizations.

Requirements
============

* Python environment from repo setup.
* Local CamCAN copy with ``cc700`` and ``dataman`` folders.
* Raw data already accessible at a path you can pass to preprocessing.

The preprocessing script expects this raw tree layout:

.. code-block:: text

   <raw-path>/cc700
   <raw-path>/dataman

Data Preparation
================

Run preprocessing from ``scripts/``:

.. code-block:: bash

   cd scripts
   python prepare_data.py --config default.ini --raw-path /path/to/camcan --save-path /path/to/output

Notes:

* Use ``prepare_data_parallel.py`` for parallel preprocessing on same input layout.
* Script writes processed ``.npy`` files, participant tables, and logs under ``--save-path``.
* Event-based runs use ``dataset = passive`` in config.
* Resting-state runs use ``dataset = rest``.

Training
========

Train model after preprocessing, still from ``scripts/``:

.. code-block:: bash

   python train_net.py --config eventclf.ini

For subject classification, use:

.. code-block:: bash

   python train_net.py --config subclf.ini

Config files in this folder control model, sampling, and output paths:

* ``eventclf.ini`` for event classification.
* ``subclf.ini`` for subject classification.

Expected Outputs
================

After training, expect model checkpoints, logs, and evaluation results in paths defined by config. Keep same ``save-path`` and ``model-name`` if you want outputs to match paper run layout.

Visualization Notebooks
=======================

Switch to notebooks after training to reproduce figures and interpretability plots.

Recommended notebooks:

* ``notebooks/visu_saliency.ipynb`` for saliency maps.
* ``notebooks/visu_gradcam.ipynb`` for Grad-CAM visualizations.
* ``notebooks/visu_filters.ipynb`` for learned filter inspection.
* ``notebooks/visu_erps.ipynb`` for ERP-style summaries.
* ``notebooks/visu_saliency_paper_figure.ipynb`` for paper-ready saliency figure generation.

Open those notebooks only after preprocessing and training completed, since they load saved outputs from prior steps.
