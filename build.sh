#!/usr/bin/env bash
# Build script for Render deployment
# Installs system dependencies for MP3 support

set -o errexit

echo "Installing system dependencies for MP3 support..."

# Check if we're on a Debian/Ubuntu system (Render uses Debian)
if command -v apt-get &> /dev/null; then
    # Update package list
    apt-get update -qq
    
    # Install ffmpeg and audio libraries (non-interactive)
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libsndfile1 \
        2>&1 | grep -v "debconf: delaying package configuration"
    
    echo "System dependencies installed successfully!"
else
    echo "Warning: apt-get not available. MP3 support may be limited."
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

