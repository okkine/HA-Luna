"""Constants for the Luna integration."""

DOMAIN = "luna"

elevation_step = 0.5
azimuth_step = 1

DEBUG_ATTRIBUTES = False
# Debug flags for sensors
DEBUG_ELEVATION_SENSOR = False
DEBUG_AZIMUTH_SENSOR = False
# Add more as needed for other sensors

ELEVATION_TOLERANCE = 1  # seconds

# Azimuth search tolerance
AZIMUTH_DEGREE_TOLERANCE = 0.001 # degrees - azimuth precision tolerance

# Azimuth reversal detection
AZIMUTH_REVERSAL_SEARCH_MAX_ITERATIONS = 5000

# Azimuth ternary search iteration limit
AZIMUTH_TERNARY_SEARCH_MAX_ITERATIONS = 5000  # Maximum iterations for binary search refinement
