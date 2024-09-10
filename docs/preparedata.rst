Prepare data Tutorial:
======================

The goal of this tutorial is to guide the user through the steps on how to get your dataset ready for the meegnet library. In order for the dataset class to work flawlessly, only two major elements are necessary. The first one is the data folder that will be in a folder inside the dataset path. The second one is a csv file that contains the information about the labels for each subject matrix data.

The data folder
---------------

The data folder *MUST* be named:

.. code-block:: bash
    
   downsampled_[sfreq]

where sfreq is the sampling frequency of the data inside the folder. For example if the data has been sampled at 1000Hz, the folder should be named:

.. code-block:: bash

   downsampled_1000

The data folder should contain numpy-compatible files (see the `numpy doc <https://numpy.org/devdocs/reference/generated/numpy.lib.format.html>`_). These files must start with the subject/participant ID in their name, followed by an underscore ("_") and any other information you need/want.

Here is an example of a valid name for a single participant file:

.. code-block:: bash

   CC12345_funny_useful_information.npy

One single file should be of the shape:

.. code-block:: bash

   (n_trials, n_sensors, n_samples)

or of the shape:

.. code-block:: bash

   (n_trials, n_sensor_type, n_sensors, n_samples)

In the case of MEG data for example where there are multiple sensors for a given location (Magnetometers and gradiometers).

An example of a valid shape for a MEG file where only the magnetometers are selected:

.. code-block:: bash

   (245, 1, 102, 2000)

These numbers mean we have:

* 245 trials
* with only one sensor type (Magnetometers)
* 102 locations
* 2000 time samples (2s of 1000Hz signal for example)

The same file could also be saved with a shape of:

.. code-block:: bash

   (245, 102, 2000)

Since there is only one sensor type, it is not necessary to explicitely have it in the shape.

The participants_info.csv file
------------------------------

The csv file will be inside the main dataset folder alongside the data folder.

By default, the meegnet.Dataset object will look for a "participants_info.csv" file but the csv file can be manually fed if it has a different name or if multiple copies are created with different labels for example.

The csv file must have a "sub" column and a "labels" column.

Each row of the csv will therefore contain the subject ID that must be the same as the subject ID given for the file name in the data folder. And contains the labels information about the subject.

The label can either be a string or an integer or a list if strings or integers. If there is only one element il the labels column for a given subject's row, the library will assume the label is the same for all trials of the subject. If a list is given, it must be of the same length as the subject data array (245 in our previous example).

Recap
-----

Inside the path you will provide when creating an instance of the meegnet.Dataset, there must be a "downsampled_[sfreq]" folder and a "participants_info.csv" file.

Inside the "downsampled_[sfreq]" folder there must be files in the format:

.. code-block:: bash
   SUBJECTID_anything.npy


Jupyter notebook version of this tutorial can be found `here <https://github.com/arthurdehgan/meegnet/blob/master/notebooks/Prepare%20Data%20Tutorial.ipynb>`__
