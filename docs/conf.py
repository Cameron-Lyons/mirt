"""Sphinx configuration for MIRT documentation."""

import sys
from pathlib import Path

# Add the src directory to the path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Project information
project = "MIRT"
copyright = "2024, Cameron Lyons"
author = "Cameron Lyons"

# Get version from package
try:
    from mirt._version import __version__

    release = __version__
    version = ".".join(release.split(".")[:2])
except ImportError:
    version = "dev"
    release = "dev"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autosummary_generate = True

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Numpydoc settings
numpydoc_show_class_members = False

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# HTML output options
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"MIRT {release}"

# Create _static directory if it doesn't exist
Path(__file__).parent.joinpath("_static").mkdir(exist_ok=True)
