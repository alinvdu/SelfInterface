#!/bin/bash
set -e

# Step 1: Build the UI
echo "Building self-ui..."
cd self-ui
npm install
npm run build
cd ..

# Step 2: Copy the build output into your FastAPI backend directory
echo "Copying built files to backend..."

# Ensure the destination directory exists
mkdir -p fast-api/frontend

# Remove any old files in the destination folder
rm -rf fast-api/frontend/*

# Copy the contents of the build folder into backend/frontend
# Adjust the source folder name if needed (e.g., 'build' vs. 'dist')
cp -r self-ui/build/* fast-api/frontend/
