import os
from pathlib import Path

# ===========================
#        CONFIG
# ===========================

# -------- Input data --------
FILE_PATH = (
    "C:\\Users\\gurja\\OneDrive\\Desktop\\EB_consumption\\data\\LD2011_2014.txt"
)  # Raw electricity consumption file

TIME_COL = "Timestamp"   # Timestamp column (auto-detect if different)
DELIM = ";"              # File delimiter

# -------- Output paths --------
output_path = "C:\\Users\\gurja\\OneDrive\\Desktop\\EB_consumption\\outputs"

# Aggregated daily data
dalily_agg_filepath = os.path.join(output_path, "daily_data_agg.csv")

# Final model-ready dataset
model_data_filepath = os.path.join(output_path, "model_data.parquet")

# Base directory for model artifacts
base = Path(output_path) / "model_dev"
model_dir = base / "models"
res_dir = base / "results"

# Create directories if they don't exist
model_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)

# -------- Train / Test time split --------
training_time_period = ("2012-07-01", "2014-06-30")
test_time_period = ("2014-07-01", "2014-12-31")

# ===========================
#     MODELING PARAMETERS
# ===========================

# -------- Feature selection thresholds --------

# Drop features with missing % greater than this
missing_perc_threshold = 0.5

# Drop features with variance lower than this
variance_threshold = 0.01

# Minimum absolute correlation with target
correlation_with_target_threshold = 0.01

# Maximum allowed correlation between features
correlation_wd_other_features_cutoff = 0.7

# Information Value threshold (if used)
IV_threshold = 0.02

# Cramér’s V cutoff for categorical features
creamers_V_cutoff = 0.1

# Maximum features allowed per feature category
max_features_from_a_category = 2
