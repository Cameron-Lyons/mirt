"""Sphinx configuration for MIRT documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

project = "MIRT"
copyright = "2026, Cameron Lyons"
author = "Cameron Lyons"

try:
    from mirt._version import __version__

    release = __version__
    version = ".".join(release.split(".")[:2])
except ImportError:
    version = "dev"
    release = "dev"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"MIRT {release}"

Path(__file__).parent.joinpath("_static").mkdir(exist_ok=True)
