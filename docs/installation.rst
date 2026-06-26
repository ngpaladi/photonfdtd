Installation
============

From source
-----------

photonfdtd is not on PyPI yet, so ``pip install photonfdtd`` does not work.
Install it from source instead — either directly from GitHub:

.. code-block:: bash

   pip install "git+https://github.com/ngpaladi/photonfdtd"

or from a checkout:

.. code-block:: bash

   git clone https://github.com/ngpaladi/photonfdtd
   cd photonfdtd
   pip install -e .

Once a release is published to PyPI, ``pip install photonfdtd`` will work too.

Optional extras
---------------

.. code-block:: bash

   pip install -e ".[test]"   # pytest
   pip install -e ".[docs]"   # sphinx + furo, to build these docs

photonfdtd requires Python 3.10+ and depends on NumPy, SciPy, and Matplotlib.
The ``gdsfactory`` adapter and the optional Numba acceleration are imported
lazily, so neither is required for a base install.
