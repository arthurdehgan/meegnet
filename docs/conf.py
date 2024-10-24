# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../meegnet/"))

project = "MEEGNet"
copyright = "2024, Arthur Dehgan"
author = "Arthur Dehgan"
release = "0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["nbsphinx", "sphinx.ext.autodoc", "guzzle_sphinx_theme", "sphinx.ext.napoleon"]

# generate autosummary pages
autosummary_generate = True

# The master toctree document.
master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
modindex_common_prefix = ["meegnet."]
napoleon_use_param = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import guzzle_sphinx_theme

html_theme_path = guzzle_sphinx_theme.html_theme_path()
html_theme = "guzzle_sphinx_theme"
html_sidebars = {
    "**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"]
}

# Register the theme as an extension to generate a sitemap.xml
extensions.append("guzzle_sphinx_theme")

html_theme_options = {
    # Set the name of the project to appear in the sidebar
    "project_nav_name": "MEEGNet",
}
