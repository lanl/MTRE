#!/bin/bash
module load python  # if your system uses modules

# Make sure Conda commands are available
source $(conda info --base)/etc/profile.d/conda.sh

# Activate your environment
conda activate /usr/projects/unsupgan/geigh_env

# Check Python version
echo "Python version:"
python --version

# Check for OpenCV (cv2)
echo "Testing for cv2..."
python - <<EOF
try:
    import cv2
    print("✅ OpenCV is installed. Version:", cv2.__version__)
except ImportError:
    print("❌ OpenCV (cv2) is NOT installed in this environment.")
EOF
