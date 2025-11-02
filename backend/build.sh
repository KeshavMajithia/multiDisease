#!/bin/bash
set -e

# Install system dependencies
apt-get update && apt-get install -y python3.9 python3.9-dev python3.9-venv gcc

# Set Python 3.9 as default
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install Python packages
pip3 install --upgrade pip setuptools wheel
pip3 install -r requirements.txt