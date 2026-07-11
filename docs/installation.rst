Installation
============

From source
-----------

photonfdtd isn't on PyPI yet, so ``pip install photonfdtd`` won't get you
anywhere. Install it from source instead — either straight from GitHub:

.. code-block:: bash

   pip install "git+https://github.com/ngpaladi/photonfdtd"

or from a checkout, which is what you want if you plan to poke at the code:

.. code-block:: bash

   git clone https://github.com/ngpaladi/photonfdtd
   cd photonfdtd
   pip install -e .

Once there's a release on PyPI, plain ``pip install photonfdtd`` will work too.

Optional extras
---------------

.. code-block:: bash

   pip install -e ".[test]"   # pytest
   pip install -e ".[docs]"   # sphinx + the Read the Docs theme, to build these docs

photonfdtd needs Python 3.10+ and leans on NumPy, SciPy, and Matplotlib, and
nothing else for the core. The ``gdsfactory`` adapter and the optional Numba
acceleration are imported lazily, so you don't need either one for a base
install — they only cost you anything if you actually reach for them.

GPU acceleration
----------------

The GPU story is JAX. Pass ``use_jax=True`` to a simulation and, once a
CUDA-enabled ``jaxlib`` is installed, the forward run *and* the differentiable /
reversible adjoints run on the GPU through XLA — no code change on your end:

.. code-block:: bash

   pip install "jax[cuda12]"   # NVIDIA / CUDA 12 GPU

This is validated on an NVIDIA RTX 4080: the GPU forward run matches the NumPy
reference to floating-point rounding, and both adjoints run on-device. If JAX
quietly falls back to CPU with a "cuSPARSE not found"-style error, the CUDA
libraries just aren't on the loader path yet; point ``LD_LIBRARY_PATH`` at the
``nvidia/*/lib`` directories the ``jax[cuda12]`` install dropped in your
site-packages and it'll find them.

There's also a legacy `CuPy <https://cupy.dev>`_ backend behind
``use_gpu=True``. It's superseded by JAX for GPU work (and deprecated), but it's
kept around for two things JAX can't do: AMD/ROCm hardware, and streaming
disk-backed tiles through the GPU for a run that's larger than your VRAM. If you
want either, install the CuPy build that matches your card:

.. code-block:: bash

   pip install cupy-cuda12x    # NVIDIA / CUDA GPU
   pip install cupy-rocm-5-0   # AMD / ROCm GPU

CuPy is optional and imported lazily; with neither it nor a CUDA ``jaxlib``
installed, the plain NumPy core still runs everything on the CPU.

Which backend actually runs
---------------------------

You don't have to think about any of this most of the time. By default
(``backend="auto"``) a simulation picks its own engine: if JAX is installed and
the run is big enough that compiling it pays off, it uses the JAX backend (on the
GPU, if there's one) — and otherwise it stays on the plain NumPy core, which is
instant and needs no extra dependency. On a large 3-D forward run that hands you
roughly a 4x speedup on CPU and ~40x on GPU without changing a line of code. If
you'd rather pin it, pass ``backend="numpy"`` (deterministic, dependency-light)
or ``backend="jax"`` outright. Out-of-core runs always take the NumPy tiling
path, since JAX can't stream to disk.
