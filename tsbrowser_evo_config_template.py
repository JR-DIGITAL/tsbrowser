import matplotlib.dates as mdates

# set name of the interpreter
interpreter = "Joe Doe"

# set point sample vector file path
path = "//digs110/fer/HR-VPP2/calibration/testing/test_setup_points.shp"

# define required attribute names
legacy_mode = False  # if True, the program expects pre-processed image chips and all
# parameters regarding on-the-fly quality evaluation are ignored
attr_id = "id_str"
attr_q_loc = "data_loc"  # location of quality raster files for the feature
attr_i_loc = "data_loc"  # location of image raster files for the feature

# set quality raster file search options
q_pattern = "L2A*scl.vrt"
q_recursive = False
q_depth = 1

# set image raster file search options
i_pattern = "L2A*[ab].vrt"  # '*[CKU].[tv][ir][ft]'
i_recursive = False
i_depth = 1

# set options for retrieving the image acquisition timestamp
t_mode = "filename"  # possible values: metadata, filename
t_slice = slice(4, 19)  # used if t_mode == filename
t_format = "%Y%m%dT%H%M%S"  # used if t_mode == filename

# set quality evaluation parameters
q_mode = "threshold_gt"  # possible values: classes, threshold_lt, threshold_gt
masking_classes = (0, 3, 8, 9, 11)  # classes to mask (set to false) used if mode == 'classes'
valid_classes = None  # classes to keep (set to true), mutually exclusive with masking_classes used if mode == 'classes'
q_band = 3  # used if mode == 'threshold_gt'
threshold = 250  # used if mode in ('threshold_lt', 'threshold_gt')
overall_valid_ratio = 0.6
specific_radius = 9
specific_valid_ratio = 0.8

# define the flag label set with a right mouse button click
default_flag_label = "1"
add_flag_labels = "a b c d e f g h".split()

# set an output directory to store flag files (pickle format)
# if None, write to the shapefile directory
flag_dir = "//digs110/fer/HR-VPP2/calibration/testing/flags"

# assign names to input bands
layermap = {"B02": 1, "B03": 2, "B04": 3, "B08": 4, "B11": 9}

# Indices which should be calculated on the fly
# only simple arithmetic operations are supported (+,-,*,/)
indices = {
    "NDMI": "(B08 - B11) / (B08 + B11)",  # Normalized Difference Moisture Index
    #    "NDVI": "(B08 - B04) / (B08 + B04)",  # Normalized Difference Vegetation Index
    #    "NBR": "(B08 - B11) / (B08 + B11)",  # Normalized Burn Ratio
    #    "TCW": " 0.0315*B02 + 0.2021*B03 + 0.3102*B04 + 0.1594*B08 - 0.6806*B11 - 0.6109*B12"
}

# configure which bands to display in the 3 time series sub-plots
timeseries = {"ts_B1": "NDMI", "ts_B2": "B08", "ts_B3": "B11"}

# configure which band combination to display in the 2 image sub-plots
images = {"img_L": ("B04", "B03", "B02"), "img_R": ("B11", "B08", "B04")}

# initial image chip size [m]
chip_width = 1600
chip_height = 1200

# image pixel value scale factor handling
apply_metadata_zscale = False
zscale = 1.0

# define band-specific min/max contrast stretch (absolute reflectance, e.g. 0.1 means 10% reflectance)
contrast = {
    "B02": (0.0, 1800.0),
    "B03": (0.0, 2000.0),
    "B04": (0.0, 2200.0),
    "B08": (0.0, 6000.0),
    "B11": (0.0, 5000.0),
}

# contrast = {'B02': 'mean_stddev',
# 'B03': 'mean_stddev',
# 'B04': 'mean_stddev',
# 'B08': 'mean_stddev',
# 'B11': 'mean_stddev'}

# contrast = {'B02': 'median_mad',
# 'B03': 'median_mad',
# 'B04': 'median_mad',
# 'B08': 'median_mad',
# 'B11': 'median_mad'}

# contrast = {'B02': 'pct_clip',
# 'B03': 'pct_clip',
# 'B04': 'pct_clip',
# 'B05': 'pct_clip',
# 'B08': 'pct_clip',
# 'B11': 'pct_clip'
# }

pct_min = 0.3  # for stretch mode pct_clip
pct_max = 98.0  # for stretch mode pct_clip

std_factor = 2.5  # for stretch mode mean_stddev and median_mad

# format time series plots
ylim = ["auto", "auto"]  # use min, max, or numeric values. Not used with --semilogy
ytick_multiples = (
    100  # Round y tick labels to multiples of this value. Not used with --semilogy
)
monthLoc = mdates.MonthLocator(bymonth=range(1, 13, 6))
monthFmt = mdates.DateFormatter("%Y %b %d")

# VHR module options
vhr_zoom = 17
remove_duplicates = True
