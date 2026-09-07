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

Subject Size Experiment
=======================

To reproduce the effect of training set size on performance, train one model per subject count.
Run from ``scripts/``:

.. code-block:: bash

   for max_subj in 25 50 100 150 200 300 450; do
     poetry run python train_net.py \
       --config eventclf.ini \
       --save-path /workspace/camcan/eventclf \
       --model-name "eventclf_meegnet_42_ALL_${max_subj}" \
       --max-subj "$max_subj"
   done

Each run produces a separate model under ``--save-path``, named with the subject count.
Compare test metrics across runs to reproduce the subject-size scaling curve from the paper.

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
