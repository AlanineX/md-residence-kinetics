"""Configuration for standalone additive-attribution plotting."""

SOURCE_PATH = "/path/to/results"
OUTPUT_PATH = "/path/to/results/plots"

X_MAX_PLOT = 5.0
N_BINS = 25
BIN_SPACING_FACTOR = 0.8
BASE_FONTSIZE = 16
TITLE = ""

# Ranking: "residue_id", "input", "alphabetical", "initial_percentage",
# or "integrated_contribution". MANUAL_ORDER overrides the selected method.
STACK_ORDER = "residue_id"
MANUAL_ORDER = None
LARGEST_AT_BOTTOM = True
RANK_HORIZON_NS = 5.0

# Coloring: "component_identity" keeps colors attached to residue/component
# identity; "stack_position" keeps the warm-to-cool progression along the stack.
COLOR_METHOD = "component_identity"
PALETTE = "harmonic"
COMPONENT_COLORS = {}
