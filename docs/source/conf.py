# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations
import os
import sys
package_path = os.path.abspath('../../')
sys.path.insert(0, package_path)
package_path = os.path.abspath('../../pyQSM')
sys.path.insert(0, package_path)
print('############### ', package_path)
my_path =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))+'/pyQSM/'
sys.path.insert(0,my_path)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "pyQSM"
copyright = "2024, Collin Wischmeyer"
authors = "Collin Wischmeyer"
release = "0.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    # 'm2r',
]

templates_path = ["_templates"]
exclude_patterns = ["*.py"]

# -- Options for myst_parser  -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/markdown.html
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}
myst_enable_extensions = [
    # "attrs_inline",
    # "colon_fence",
    "deflist",
    # "dollarmath",
    # "fieldlist",
    "html_admonition",
    "html_image",
    # "linkify",
    "smartquotes",
    "substitution",
    # "tasklist",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "includehidden": False,
    "titles_only": True,
    "prev_next_buttons_location": "both",
}
