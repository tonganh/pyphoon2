#!/bin/bash

# Check if the folder exists
if [ ! -d "test_data_files/image/200801" ]; then
  echo "Directory 'test_data_files/image/200801' does not exist. Downloading required files..."

  # Download the ZIP file
  wget -O test_data_files.zip https://minio.hisoft.com.vn/anhtn/test_data_files.zip

  # Check if the download was successful
  if [ $? -ne 0 ]; then
    echo "Failed to download the file. Exiting."
    exit 1
  fi

  # Unzip the file to the specified directory
  unzip test_data_files.zip -d test_data_files

  # Check if unzip was successful
  if [ $? -ne 0 ]; then
    echo "Failed to unzip the file. Exiting."
    exit 1
  fi

  echo "Files downloaded and extracted successfully."
else
  echo "Directory 'test_data_files/image/200801' already exists. Skipping download."
fi

python3 -m unittest test_DigitalTyphoonDataset.TestDigitalTyphoonDataset
python3 -m unittest test_DigitalTyphoonImage.TestDigitalTyphoonImage
python3 -m unittest test_DigitalTyphoonSequence.TestDigitalTyphoonSequence
python3 -m unittest test_DigitalTyphoonUtils.TestDigitalTyphoonUtils
