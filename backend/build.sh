#!/bin/bash
set -e

# Install system dependencies
apt-get update && apt-get install -y python3.9 python3.9-dev python3.9-venv gcc

# Create and activate virtual environment
python3.9 -m venv /vercel/.venv
source /vercel/.venv/bin/activate

# Install Python packages
pip install --upgrade pip setuptools wheel
pip install -e .