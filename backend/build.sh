#!/bin/bash
set -e

# Install system dependencies
apt-get update && apt-get install -y python3-dev gcc

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install Python packages
pip install -r requirements.txt