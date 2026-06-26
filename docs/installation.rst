Installation
============

From PyPI
---------

.. code-block:: bash

   pip install photonfdtd

From a checkout
---------------

.. code-block:: bash

   git clone https://github.com/ngpaladi/photonfdtd
   cd photonfdtd
   pip install -e .

Optional extras
---------------

.. code-block:: bash

   pip install -e ".[test]"   # pytest
   pip install -e ".[docs]"   # sphinx + furo, to build these docs

photonfdtd requires Python 3.10+ and depends on NumPy, SciPy, and Matplotlib.
The ``gdsfactory`` adapter and the optional Numba acceleration are imported
lazily, so neither is required for a base install.
