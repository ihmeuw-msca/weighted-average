Quickstart
==========

The Basics
----------

The `weave` package has four main components:

1. The :doc:`Smoother <../api_reference/weave.smoother>` class, which computes
   weighted averages across data.
2. The :doc:`Dimension <../api_reference/weave.smoother>` class, where you
   specify how weights are computed for each dimension in your data.
3. The :doc:`kernel <../api_reference/weave.kernels>` functions, which compute
   weights given the distances between points.
4. The :doc:`distance <../api_reference/weave.distance>` functions, which
   compute the distances between points.

In this tutorial, we will smooth noisy data across one dimension. We will start
by importing all of the packages we will be using.

.. code-block::

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    from weave.dimension import Dimension
    from weave.smoother import Smoother

Data
^^^^

The `weave` package works with data in the form of
`Pandas <https://pandas.pydata.org/>`_ data frames. The following code 
generates and plots our data set.

.. code-block::

    n_obs = 50
    x_val = np.linspace(0, 2*np.pi, n_obs)
    y_true = np.sin(x_val)
    y_obs = y_true + 0.3*np.random.normal(size=n_obs)

    data = pd.DataFrame({
        'x_id': np.arange(n_obs),
        'x_val': x_val,
        'y_true': y_true,
        'y_obs': y_obs
    })

    plt.plot(x_val, y_true, label='Truth')
    plt.plot(x_val, y_obs, '.', label='Observed')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

.. image:: images/raw_data.png
   :align: center

Dimension
^^^^^^^^^

To smooth data using `weave`, we first need to specify the dimensions in our
data by creating an instance of the ``Dimension`` class. This object contains
information about how weights are computed. Many dimension parameters have
defaults, but at the minimum we need to include ``name``. This corresponds to
the column in our data frame containing unique point IDs.

.. code-block::

    age = Dimension('x_id')

This dimension object above will have the default identity kernel (i.e.,
weights are equal to distances) and Euclidean distance. Weights are computed
using data from the column of our data frame corresponding to the dimension
attribute ``coordinates``, which is automatically assigned to the ``name``
parameter if not specified.

To compute weights based on a column other than 'x_id', we can include the
``coordinates`` parameter. Values in this column do not need to be unique,
unlike those corresponding to ``name``.

.. code-block::

    age = Dimension('x_id', 'x_val')

All kernels other than identity kernel have additional parameters. For this
tutorial, we will use the exponential kernel which requires a kernel radius.
The following code creates our smoothing dimension with default Euclidean
distance.

.. code-block::

    dim = Dimension(
        name='x_id',
        coordinates='x_val',
        kernel='exponential',
        radius=0.5
    )

Smoother
^^^^^^^^

Our next step is to create an instance of the ``Smoother`` class using the
dimension object we just defined. While this tutorial only uses one dimension,
you can also input multiple dimensions in a list.

.. code-block::

    smoother = Smoother(dim)

To smooth our noisy data, we simply provide our data frame and the name of the
column or columns we would like to smooth. The output is a copy of input data
frame with appended column(s) containing the smoothed data. For ``columns``
'col1', 'col2', etc., these smoothed columns will be labeled 'col1_smooth',
'col2_smooth', etc. The following code smooths our data and plots the results.

.. code-block::

    result = smoother(data, 'y_obs')

    plt.plot(x_val, y_true, label='Truth')
    plt.plot(x_val, y_obs, '.', label='Observed')
    plt.plot(result['x_val'], result['y_obs_smooth'], '.', label='Smoothed')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

.. image:: images/smooth_data.png
   :align: center
