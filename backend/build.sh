#!/bin/bash
set -e

# Install system dependencies
apt-get update && apt-get install -y python3-dev gcc

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt