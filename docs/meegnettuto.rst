Train Network Basics:
=====================

You can find a jupyter notebook version of this tutorial `here <https://github.com/arthurdehgan/meegnet/blob/master/notebooks/Meegnet%20Network%20Training%20Basic%20Tutorial.ipynb>`__

How to use the dataset class
----------------------------

Loading a dataset requires the data to be in the correct format (see `Prepare data tutorial <https://meegnet.readthedocs.io/en/latest/tutorials.html>`_). Just create the dataset object and use the load method:

.. code-block:: python

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )


    from meegnet.dataloaders import RestDataset

    data_path = "/home/arthur/data/camcan/subclf"

    # use Dataset class for data that has already been cut into trials
    # else, use RestDataset with additional parameters of window and overlap to create trials.
    dataset = RestDataset(
        window=4, # window size of 4 seconds
        overlap=0.25, # 25% overlap between windows
        sfreq=200, # sampling frequency of 200 
        n_subjects=100, # only load 100 subjects
        n_samples=100, # limit the number of samples for each subject to 100
        sensortype="GRAD", # only use gradiometers
        lso=False, # do not use leave subject oout for data splits
    )

    dataset.load(data_path)

.. code-block:: bash

    05/14/2024 10:31:58 AM Logging subjects and labels from /home/arthur/data/camcan/subclf...
    05/14/2024 10:31:58 AM Found 100 subjects to load.

.. code-block:: python

    print(len(dataset))
    print(dataset.data.shape)
    print(dataset.labels[[1, 500, 800]])

.. code-block:: bash

    10000
    torch.Size([10000, 2, 102, 600])
    tensor([84.,  5., 80.])

We have loaded 100 subjects of the resting-state dataset located in data_path. There are 100 examples per subject so 10000 data examples total. With only gradiometers selected with sensors="GRAD", we only have 2 channels. The length of each time segment is 4 seconds at 200Hz which is why they are 800 time points of size.
How to use the network class

Create the model object instance of the Model class and then use the train method with the dataset previously created.

.. code-block:: python

    from meegnet.network import Model

    save_path = data_path
    net_option = "best_net"
    input_size = dataset.data[0].shape
    n_outputs = 100 # Here we have 100 possible outputs as we have 1 label per subject and 100 subjects
    name = "my_cool_network"

    my_model = Model(name, net_option, input_size, n_outputs, save_path)

    print(my_model.net)

    my_model.train(dataset)

.. code-block:: bash

    05/14/2024 10:36:41 AM Creating DataLoaders...

    my_net(
    (maxpool): MaxPool2d(kernel_size=(1, 20), stride=1, padding=0, dilation=1, ceil_mode=False)
    (feature_extraction): Sequential(
        (0): Conv2d(2, 100, kernel_size=(102, 1), stride=(1, 1))
        (1): ReLU()
        (2): Conv2d(100, 200, kernel_size=(1, 9), stride=(1, 1))
        (3): MaxPool2d(kernel_size=(1, 20), stride=1, padding=0, dilation=1, ceil_mode=False)
        (4): ReLU()
        (5): Conv2d(200, 200, kernel_size=(1, 9), stride=(1, 1))
        (6): MaxPool2d(kernel_size=(1, 20), stride=1, padding=0, dilation=1, ceil_mode=False)
        (7): ReLU()
        (8): Conv2d(200, 100, kernel_size=(1, 9), stride=(1, 1))
        (9): MaxPool2d(kernel_size=(1, 20), stride=1, padding=0, dilation=1, ceil_mode=False)
        (10): ReLU()
        (11): Flatten()
        (12): Dropout(p=0.5, inplace=False)
    )
    (classif): Sequential(
        (0): Linear(in_features=51900, out_features=1000, bias=True)
        (1): Linear(in_features=1000, out_features=100, bias=True)
    )
    )

    05/14/2024 10:36:41 AM Starting Training with:
    05/14/2024 10:36:41 AM batch size: 128
    05/14/2024 10:36:41 AM learning rate: 1e-05
    05/14/2024 10:36:41 AM patience: 20
    05/14/2024 10:36:43 AM Epoch: 1 // Batch 1/63 // loss = 4.60947
    [...]
    05/14/2024 10:37:16 AM Epoch: 1
    05/14/2024 10:37:16 AM  [LOSS] TRAIN 4.592482872009278 / VALID 4.600228523254395
    05/14/2024 10:37:16 AM  [ACC] TRAIN 0.03325 / VALID 0.018000000037252904
    05/14/2024 11:53:15 AM Epoch: 159 // Batch 1/63 // loss = 0.00072
    [...]
    05/14/2024 11:53:39 AM Epoch: 159
    05/14/2024 11:53:39 AM  [LOSS] TRAIN 0.0018134863618761302 / VALID 0.3479517550468445
    05/14/2024 11:53:39 AM  [ACC] TRAIN 1.0 / VALID 0.9129999990463257

It is always possible to access the network inside the Model object if we want to perform single trial predictions for a figure for example:

.. code-block:: python 

    import numpy as np
    import torch

    random_sample = 555
    data_example = dataset.data[random_sample][np.newaxis] # need to add a new axis to respect expected shapes, not needed if using multiple examples.

    pred = my_model.net.forward(torch.Tensor(data_example).cuda())

    print(f"predicted label: {np.argmax(pred.detach().cpu().numpy())}, original label: {dataset.labels[random_sample]}")

.. code-block:: bash

    predicted label: 5, original label: 5.0

