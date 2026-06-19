# NexusRT Sphinx configuration.
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(".."))

project = "NexusRT"
author = "NexusRT Contributors"
copyright = "2026, NexusRT Contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"] if Path("_static").exists() else []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]
