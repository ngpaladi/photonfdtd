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

GPU acceleration
----------------

Passing ``use_gpu=True`` to a simulation runs the time-stepping through
`CuPy <https://cupy.dev>`_. Only generic CuPy array operations are used, so
either GPU vendor works — install the CuPy build that matches your hardware:

.. code-block:: bash

   pip install cupy-cuda12x    # NVIDIA / CUDA GPU
   pip install cupy-rocm-5-0   # AMD / ROCm GPU

CuPy is optional and imported lazily; with none installed, ``use_gpu=True``
raises a clear error and the NumPy core still runs. (CuPy's ROCm support is
itself experimental, so match its version to your ROCm install.)
