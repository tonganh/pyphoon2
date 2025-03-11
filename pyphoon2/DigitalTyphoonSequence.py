import os
import warnings
from datetime import datetime

from typing import Callable

from pathlib import Path
import numpy as np
from typing import List, Dict
import pandas as pd

from pyphoon2.DigitalTyphoonImage import DigitalTyphoonImage
from pyphoon2.DigitalTyphoonUtils import parse_image_filename, is_image_file, TRACK_COLS, parse_common_image_filename


class DigitalTyphoonSequence:

    def __init__(self, seq_str: str, start_season: int, num_images: int, transform_func=None,
                 spectrum='Infrared', verbose=False):
        """
        Class representing one typhoon sequence from the DigitalTyphoon dataset

        :param seq_str: str, sequence ID as a string
        :param start_season: int, the season in which the typhoon starts in
        :param num_images: int, number of images in the sequence
        :param transform_func: this function will be called on each image before saving it/returning it.
                            It should take and return a np array
        :param spectrum: str, specifies the spectrum of the images (e.g., 'Infrared')
        :param verbose: bool, if True, additional information and warnings will be printed during processing
        """
        self.verbose = verbose

        self.sequence_str = seq_str  # sequence ID string
        self.season = start_season
        self.num_track_entries = 0
        self.num_original_images = num_images
        self.track_data = np.array([])
        self.img_root = None  # root path to directory containing image files
        self.track_path = None  # path to track file data
        self.transform_func = transform_func
        self.spectrum = spectrum

        # Ordered list containing image objects with metadata
        self.images: List[DigitalTyphoonImage] = list()

        # Dictionary mapping datetime to Image objects
        self.datetime_to_image: Dict[datetime, DigitalTyphoonImage] = {}

        # TODO: add synthetic image value into metadata and consistency check within dataset loader

    def get_sequence_str(self) -> str:
        """
        Returns the sequence ID as a string

        :return: string sequence ID
        """
        return self.sequence_str

    def process_seq_img_dir_into_sequence(self, directory_path: str,
                                          load_imgs_into_mem=False,
                                          ignore_list=None,
                                          spectrum=None,
                                          filter_func: Callable[[DigitalTyphoonImage], bool] = lambda img: True) -> None:
        """
        Processes the image directory and stores the images found within the sequence.

        :param directory_path: Path of directory containing images of this sequence
        :param load_imgs_into_mem: Boolean for whether the image data should be stored in mem
        :param ignore_list: Set of image filenames to ignore
        :param spectrum: string, name of which spectrum the image is from, IR is default
        :param filter_func: function that takes in a DigitalTyphoonImage and returns bool, whether that img should be kept
        :return: None
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(
                f"{directory_path} is not a valid directory.")

        if ignore_list is None:
            ignore_list = set([])

        if spectrum is None:
            spectrum = self.spectrum

        self.set_images_root_path(directory_path)

        files_to_parse = []
        for root, dirs, files in os.walk(directory_path, topdown=True):
            for file in files:
                if is_image_file(file):
                    path_of_file = os.path.join(root, file)
                    files_to_parse.append(path_of_file)

        # Using sorted makes it deterministic
        for file_path in sorted(files_to_parse):
            try:
                # Extract metadata from file, important is sequence ID and datetime of image
                file_basename = os.path.basename(file_path)
                metadata = parse_image_filename(file_basename)
                file_sequence, file_datetime, _ = metadata

                if file_basename not in ignore_list:
                    # Check if this sequence and ID matches; sometimes track data for 1 typhoon is stored with data of another
                    # In which case, the sequence string and date matches
                    # Do another check for match between sequence stored in filename and object, sequence idx is used to verify
                    if file_sequence == self.get_sequence_str():
                        if file_datetime not in self.datetime_to_image:
                            # Create digital typhoon image object
                            self.datetime_to_image[file_datetime] = DigitalTyphoonImage(
                                file_basename, file_path, sequence_id=self.get_sequence_str(),
                                transform_func=self.transform_func, spectrum=self.spectrum)
                        else:
                            # If object already exists, just set image path (case of getting multi-spectrum images)
                            self.datetime_to_image[file_datetime].set_image_data(
                                file_basename, file_path, load_img_into_mem=load_imgs_into_mem, spectrum=spectrum)

                        # Add image to running list only if it passes the filter
                        if filter_func(self.datetime_to_image[file_datetime]):
                            if self.datetime_to_image[file_datetime] not in self.images:
                                self.images.append(
                                    self.datetime_to_image[file_datetime])

            except Exception as e:
                if self.verbose:
                    warnings.warn(str(e))

        # Make sure image dataset is ordered w.r.t. time
        try:
            self.images.sort(key=lambda x: x.get_datetime())
        except Exception as e:
            pass

        # Print warning if there is an inconsistency between number of found images and expected number
        if self.verbose:
            if not self.num_images_match_num_expected():
                warnings.warn(f'The number of images ({len(self.images)}) does not match the '
                              f'number of expected images ({self.num_original_images}) from metadata. If this is expected, ignore this warning.')

            if len(self.images) < self.num_track_entries:
                warnings.warn(
                    f'Only {len(self.images)} of {self.num_track_entries} track entries have images.')

    def process_seq_img_dirs_into_sequence(self, directory_paths: List[str],
                                           common_image_names: List[str],
                                           load_imgs_into_mem=False,
                                           ignore_list=None,
                                           filter_func: Callable[[DigitalTyphoonImage], bool] = lambda img: True,
                                           spectrum=None) -> None:
        """
        Process images from multiple directories into a sequence, combining them as multiple channels.
        
        :param directory_paths: List of paths to the directories containing images
        :param common_image_names: List of common image names across directories
        :param load_imgs_into_mem: Bool representing if images should be loaded into memory
        :param ignore_list: list of image filenames to ignore
        :param filter_func: function that accepts an image and returns True/False if it should be included
        :param spectrum: string representing what spectrum the image lies in
        :return: None
        """
        print(f"\nProcessing sequence: {self.get_sequence_str()}")
        print(f"  - Directory paths: {directory_paths}")
        print(f"  - Common image names: {len(common_image_names)} images")
        if common_image_names:
            print(f"  - First few image names: {common_image_names[:3]}")
        
        # Input validation
        valid_dirs = []
        for directory_path in directory_paths:
            if os.path.isdir(directory_path):
                valid_dirs.append(directory_path)
            else:
                print(f"  - Warning: {directory_path} is not a valid directory, skipping.")
        
        if not valid_dirs:
            print(f"  - Error: No valid directories found for sequence {self.get_sequence_str()}")
            return
        
        directory_paths = valid_dirs
        
        if not common_image_names:
            print(f"  - Error: No common image names provided for sequence {self.get_sequence_str()}")
            return
        
        if ignore_list is None:
            ignore_list = set([])

        if spectrum is None:
            spectrum = self.spectrum
        
        # Set the root path to the first directory for reference
        self.set_images_root_path(directory_paths[0])
        
        # Dictionary to store filepath lists for each common image name
        filepath_for_common_image_names = {}
        
        # Collect all filepaths for each common image name across all directories
        print("  - Looking for image files in directories...")
        files_found = 0
        
        for directory_path in directory_paths:
            dir_files_found = 0
            
            # Get all files in this directory
            try:
                all_files_in_dir = os.listdir(directory_path)
                for file in all_files_in_dir:
                    file_path = os.path.join(directory_path, file)
                    if not os.path.isfile(file_path):
                        continue
                        
                    # Check if this file matches any common image name
                    for common_image_name in common_image_names:
                        if common_image_name in file and is_image_file(file):
                            if common_image_name not in filepath_for_common_image_names:
                                filepath_for_common_image_names[common_image_name] = []
                            filepath_for_common_image_names[common_image_name].append(file_path)
                            dir_files_found += 1
                
                files_found += dir_files_found
            except Exception as e:
                print(f"  - Error reading directory {directory_path}: {str(e)}")
        
        # Count how many common images have matching files
        images_with_files = sum(1 for img_name in filepath_for_common_image_names if filepath_for_common_image_names[img_name])
        
        # Detailed debugging for no images found
        if images_with_files == 0:
            print("  - ERROR: No image files found for any common image names!")
            print("  - Sample directory contents:")
            for directory_path in directory_paths[:2]:  # Show only first two to avoid excessive output
                try:
                    all_files = os.listdir(directory_path)
                    image_files = [f for f in all_files if is_image_file(f)]
                except Exception as e:
                    print(f"    Error listing {directory_path}: {str(e)}")
            return
        
        # Parse metadata from common image names and sort chronologically
        print("  - Parsing image metadata...")
        common_name_name_with_metadata = []
        for common_image_name in common_image_names:
            try:
                metadata = parse_common_image_filename(common_image_name)
                common_name_name_with_metadata.append((common_image_name,) + metadata)
            except ValueError as e:
                print(f"  - Warning: Skipping {common_image_name}: {e}")
                continue
            
        common_name_name_with_metadata.sort(key=lambda x: x[2])  # Sort by datetime
        
        # Process each common image
        print("  - Processing individual images...")
        images_processed = 0
        for common_image_name, file_sequence, common_image_date, _ in common_name_name_with_metadata:
            if common_image_name not in ignore_list:
                filepaths = filepath_for_common_image_names.get(common_image_name, [])
                
                # Only process if we have filepaths for this image
                if filepaths:
                    try:
                        if common_image_date not in self.datetime_to_image:
                            # Create a new image entry if it doesn't exist
                            self.datetime_to_image[common_image_date] = DigitalTyphoonImage(
                                None, None, sequence_id=self.get_sequence_str(),
                                transform_func=self.transform_func,
                                spectrum=self.spectrum)
                        
                        # Set image data with all filepaths (multiple channels)
                        self.datetime_to_image[common_image_date].set_image_datas(
                            image_filepaths=filepaths, 
                            load_imgs_into_mem=load_imgs_into_mem, 
                            spectrum=spectrum)
                        
                        # Apply filter and add to sequence if it passes
                        if filter_func(self.datetime_to_image[common_image_date]):
                            if self.datetime_to_image[common_image_date] not in self.images:
                                self.images.append(self.datetime_to_image[common_image_date])
                                images_processed += 1
                    except Exception as e:
                        print(f"  - Error processing image {common_image_name}: {str(e)}")
                        import traceback
                        traceback.print_exc()
        
        # Make sure images are sorted by time
        try:
            self.images.sort(key=lambda x: x.get_datetime())
        except Exception as e:
            pass
        
        
        if self.verbose:
            if not self.num_images_match_num_expected():
                warnings.warn(f'The number of images ({len(self.images)}) does not match the '
                             f'number of expected images ({self.num_original_images}) from metadata. If this is expected, ignore this warning.')

            if len(self.images) < self.num_track_entries:
                warnings.warn(
                    f'Only {len(self.images)} of {self.num_track_entries} track entries have images.')

    def get_start_season(self) -> int:
        """
        Get the start season of the sequence

        :return: int, the start season
        """
        return self.season

    def get_num_images(self) -> int:
        """
        Gets the number of images in the sequence

        :return: int
        """
        return len(self.images)

    def get_num_original_images(self) -> int:
        """
        Get the number of images in the sequence

        :return: int, the number of images
        """
        return self.num_original_images

    def has_images(self) -> bool:
        """
        Returns true if the sequence currently holds images (or image filepaths). False otherwise.

        :return: bool
        """
        return len(self.images) != 0

    def process_track_data(self, track_filepath: str, csv_delimiter=',') -> None:
        """
        Takes in the track data for the sequence and processes it into the images for the sequence.

        :param track_filepath: str, path to track csv
        :param csv_delimiter: delimiter for the csv file
        :return: None
        """
        if not os.path.exists(track_filepath):
            raise FileNotFoundError(
                f"The track file {track_filepath} does not exist.")

        df = pd.read_csv(track_filepath, delimiter=csv_delimiter)
        data = df.to_numpy()
        for row in data:
            row_datetime = datetime(int(row[TRACK_COLS.YEAR.value]), int(row[TRACK_COLS.MONTH.value]),
                                    int(row[TRACK_COLS.DAY.value]), int(row[TRACK_COLS.HOUR.value]))
            self.datetime_to_image[row_datetime] = DigitalTyphoonImage(None, row, sequence_id=self.get_sequence_str(),
                                                                       transform_func=self.transform_func,
                                                                       spectrum=self.spectrum)
            self.num_track_entries += 1

    def add_track_data(self, filename: str, csv_delimiter=',') -> None:
        """
        Reads and adds the track data to the sequence.

        :param filename: str, path to the track data
        :param csv_delimiter: char, delimiter to use to read the csv
        :return: None
        """
        df = pd.read_csv(filename, delimiter=csv_delimiter)
        self.track_data = df.to_numpy()

    def set_track_path(self, track_path: str) -> None:
        """
        Sets the path to the track data file

        :param track_path: str, filepath to the track data
        :return: None
        """
        if not self.track_path:
            self.track_path = track_path

    def get_track_path(self) -> str:
        """
        Gets the path to the track data file

        :return: str, the path to the track data file
        """
        return self.track_path

    def get_track_data(self) -> np.ndarray:
        """
        Returns the track csv data as a numpy array, with each row corresponding to a row in the CSV.

        :return: np.ndarray
        """
        return self.track_data

    def get_image_at_idx(self, idx, spectrum=None) -> DigitalTyphoonImage:
        """
        Gets the image at the specified index.
        
        :param idx: The index of the image to retrieve
        :param spectrum: Optional spectrum to use (defaults to sequence's spectrum)
        :return: DigitalTyphoonImage object
        """
        try:
            if idx < 0 or idx >= len(self.images):
                if self.verbose:
                    print(f"Warning: Image index {idx} out of range [0, {len(self.images) - 1}]")
                # Return a minimal DigitalTyphoonImage object
                return DigitalTyphoonImage("", np.array([]), self.get_sequence_str(), 
                                          spectrum=spectrum or self.spectrum, 
                                          verbose=self.verbose, transform_func=self.transform_func)
            
            # Get the existing image
            image = self.images[idx]
            
            # Ensure the image has a full path
            if image.image_filepath and not os.path.isabs(image.image_filepath) and os.path.exists(image.image_filepath):
                image.image_filepath = os.path.abspath(image.image_filepath)
                
            # Also fix multi-channel paths if present
            if image.image_filepaths:
                abs_image_filepaths = []
                for path in image.image_filepaths:
                    if path and not os.path.isabs(path) and os.path.exists(path):
                        abs_image_filepaths.append(os.path.abspath(path))
                    else:
                        abs_image_filepaths.append(path)
                image.image_filepaths = abs_image_filepaths
                
            return image
            
        except Exception as e:
            if self.verbose:
                print(f"Error retrieving image at index {idx}: {str(e)}")
            # Return a minimal DigitalTyphoonImage object
            return DigitalTyphoonImage("", np.array([]), self.get_sequence_str(), 
                                      spectrum=spectrum or self.spectrum,
                                      verbose=self.verbose, transform_func=self.transform_func)

    def get_image_at_idx_as_numpy(self, idx: int, spectrum=None) -> np.ndarray:
        """
        Gets the idx'th image in the sequence as a numpy array. Raises an exception if the idx is outside of the
        sequence's range.

        :param idx: int, idx to access
        :param spectrum: str, spectrum of the image
        :return: np.ndarray, image as a numpy array with shape of the image dimensions
        """
        if spectrum is None:
            spectrum = self.spectrum
        return self.get_image_at_idx(idx, spectrum=spectrum).image()

    def get_all_images_in_sequence(self) -> List[DigitalTyphoonImage]:
        """
        Returns all of the image objects (DigitalTyphoonImage) in the sequence in order.

        :return: List[DigitalTyphoonImage]
        """
        return self.images

    def return_all_images_in_sequence_as_np(self, spectrum=None) -> np.ndarray:
        """
        Returns all the images in a sequence as a numpy array of shape (num_images, image_shape[0], image_shape[1])

        :param spectrum: str, spectrum of the image
        :return: np.ndarray of shape (num_image, image_shape[0], image_shape[1])
        """
        if spectrum is None:
            spectrum = self.spectrum
        return np.array([image.image(spectrum=spectrum) for image in self.images])

    def num_images_match_num_expected(self) -> bool:
        """
        Returns True if the number of image filepaths stored matches the number of images stated when initializing
        the sequence object. False otherwise.

        :return: bool
        """
        return len(self.images) == self.num_original_images

    def get_image_filepaths(self) -> List[str]:
        """
        Returns a list of the filenames of the images (without the root path)

        :return: List[str], list of the filenames
        """
        return [image.filepath() for image in self.images]

    def set_images_root_path(self, images_root_path: str) -> None:
        """
        Sets the root path of the images.

        :param images_root_path: str, the root path
        :return: None
        """
        self.img_root = Path(images_root_path)

    def get_images_root_path(self) -> str:
        """
        Gets the root path to the image directory

        :return: str, the root path
        """
        return str(self.img_root)

    def add_image_path(self, image_filepath, image_filepaths=None, track_entry=None, verbose=False, ignore_list=None):
        """
        Adds an image to this sequence

        :param image_filepath: str, path to image h5 file
        :param image_filepaths: List[str], optional list of paths for multi-channel images
        :param track_entry: np.ndarray, track entry for this image
        :param verbose: bool, flag for verbose output
        :param ignore_list: Set of filenames to ignore
        :return: none
        """
        try:
            # Get just the basename for ignore list checking
            filename = os.path.basename(image_filepath)
            
            # Check if this file should be ignored
            if ignore_list and filename in ignore_list:
                if verbose or self.verbose:
                    print(f"Skipping ignored file: {filename}")
                return
            
            # Ensure we always store absolute paths
            if image_filepath and not os.path.isabs(image_filepath):
                # Only convert to absolute if it's a relative path that exists
                if os.path.exists(image_filepath):
                    image_filepath = os.path.abspath(image_filepath)
            
            # Also ensure absolute paths for multi-channel images
            if image_filepaths:
                abs_image_filepaths = []
                for path in image_filepaths:
                    if path and not os.path.isabs(path):
                        if os.path.exists(path):
                            abs_image_filepaths.append(os.path.abspath(path))
                        else:
                            abs_image_filepaths.append(path)
                    else:
                        abs_image_filepaths.append(path)
                image_filepaths = abs_image_filepaths
                
            # Create the image object with the full path
            image = DigitalTyphoonImage(
                image_filepath, track_entry, self.get_sequence_str(), 
                spectrum=self.spectrum, image_filepaths=image_filepaths,
                verbose=verbose,
                transform_func=self.transform_func
            )
            self.images.append(image)
            
        except Exception as e:
            if self.verbose or verbose:
                print(f"Error adding image path {image_filepath}: {str(e)}")
            # Continue processing other images
