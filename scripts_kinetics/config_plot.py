"""Configuration for standalone fitting + plotting pipeline.

Edit SOURCE_PATH / OUTPUT_PATH for your own data, or set the
KINETICS_PLOT_SRC / KINETICS_PLOT_OUT environment variables.
"""

import os

SOURCE_PATH = os.environ.get("KINETICS_PLOT_SRC", "./kinetics_results/")
OUTPUT_PATH = os.environ.get("KINETICS_PLOT_OUT", "./kinetics_results/plot")

x_max_plot         = 30.0
n_bins             = 50
bin_spacing_factor = 0.5
base_fontsize      = 16
do_plot            = True
