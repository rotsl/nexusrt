# NexusRT Sphinx configuration.
import os, sys
sys.path.insert(0, os.path.abspath(".."))

project = "NexusRT"
author = "NexusRT Contributors"
copyright = "2026, NexusRT Contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]
myst_enable_extensions = ["colon_fence", "deflist", "table"]
source_suffix = [".rst", ".md"]
master_doc = "index"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable/", None),
}
