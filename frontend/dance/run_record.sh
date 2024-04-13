#!/bin/bash

# Check if record_data.py exists
if [ -f "record_data.py" ]; then
    echo "Running record_data.py..."
    python record_data.py
else
    echo "Error: record_data.py not found!"
    exit 1
fi
