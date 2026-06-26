# Configuration file for the Sphinx documentation builder.
#
# Full reference: https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
from importlib import metadata

# Make the package importable for autodoc. The source lives under ../src.
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "photonfdtd"
author = "ngpaladi"
copyright = "2026, ngpaladi"

try:
    # Prefer the installed package's version so there is a single source of truth.
    release = metadata.version("photonfdtd")
except metadata.PackageNotFoundError:  # pragma: no cover - fallback for bare checkouts
    import photonfdtd

    release = photonfdtd.__version__
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",      # NumPy-style docstrings
    "sphinx.ext.viewcode",      # "[source]" links
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",       # math in docstrings
    "sphinx_rtd_theme",         # Read the Docs HTML theme
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Optional / heavy dependencies that autodoc must not need to import.
autodoc_mock_imports = ["gdsfactory", "numba"]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_title = f"photonfdtd {release}"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}
