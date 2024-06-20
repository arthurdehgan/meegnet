.. _INSTALL.rst:

Installation
============

Create a clean environment
--------------------------

MEEGNet requires Python 3.6 or newer. Please ensure that this prerequisite has been met before proceeding.

We recommend creating a fresh python environment in order to use this packages and assiciated scrips:

.. code:: bash

   VENV_PATH="/path/to/environment/"
   python -m venv $VENV_PATH\meegnet
   source $VENV_PATH\meegnet/bin/activate

For all ``pip`` installations, make sure ``pip`` is up-to-date by executing the following command::

    > python -m pip install -U pip

Install the package
-------------------

The stable release of MEEGNet can be installed using ``pip``::

    > pip install meegnet
    
It can be installed using the ``--user`` to avoid permission issues::

    > pip install --user meegnet
    
It can be to upgraded to a newer release by using the ``--upgrade`` flag::

    > pip install --upgrade meegnet
    
It can also be installed from within the directory of its repository after cloning from `the repo <www.github.com/arthurdehgan/meegnet>`_::

    > pip install .
    
or::

    > python setup.py install