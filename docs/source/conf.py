# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../.."))

project = "Radius Clustering"
copyright = "2024, Haenn Quentin, Chardin Brice, Baron Mickaël"
author = "Haenn Quentin, Chardin Brice, Baron Mickaël"
release = "1.3.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_prompt",
    "sphinx.ext.napoleon",
    "sphinxcontrib.sass",
    "sphinx_remove_toctrees",
    "sphinxcontrib.email",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx_design",
]

master_doc = "index"

# Specify how to identify the prompt when copying code snippets
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_exclude = "style"

# Conf of numpydoc
numpydoc_class_members_toctree = False

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
# html_static_path = ['_static']

html_logo = "./images/logo-lias.jpg"

html_short_title = "Radius Clustering"

html_sidebars = {"**": []}

html_theme_options = {
    "icon_links_label": "Icon Links",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/quentinhaenn",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
    "show_prev_next": False,
    "search_bar_text": "Search the docs ...",
    "navigation_with_keys": False,
    "navbar_align": "left",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links", "version-switcher"],
    "navbar_persistent": ["search-button"],
    "article_footer_items": ["prev-next"],
    "footer_start": ["copyright"],
    "footer_center": [],
    "footer_end": [],
}

# Compile scss files into css files using sphinxcontrib-sass
sass_src_dir, sass_out_dir = "scss", "_static/styles"
sass_targets = {
    f"{file.stem}.scss": f"{file.stem}.css"
    for file in Path(sass_src_dir).glob("*.scss")
}

html_static_path = ["_static"]
# Additional CSS files, should be subset of the values of `sass_targets`
html_css_files = ["styles/custom.css"]

sg_examples_dir = "../../examples"
sg_gallery_dir = "auto_examples"
sphinx_gallery_conf = {
    "doc_module": "radius_clustering",
    "backreferences_dir": os.path.join("modules", "generated"),
    "show_memory": False,
    "examples_dirs": [sg_examples_dir],
    "gallery_dirs": [sg_gallery_dir],
    # avoid generating too many cross links
    "inspect_global_variables": False,
    "remove_config_comments": True,
    "plot_gallery": "True",
    "recommender": {"enable": True, "n_examples": 4, "min_df": 12},
    "reset_modules": ("matplotlib", "seaborn"),
}
