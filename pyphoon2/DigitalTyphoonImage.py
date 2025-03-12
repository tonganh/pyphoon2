import os
import h5py
import numpy as np
from typing import List
import pandas as pd
from datetime import datetime

from pyphoon2.DigitalTyphoonUtils import TRACK_COLS


class DigitalTyphoonImage:
    def __init__(self, image_filepath: str, track_data=None, sequence_id=None, load_imgs_into_mem=False,
                 transform_func=None, spectrum=None, image_filepaths=None, verbose=False):
        """
        Class for one image with metadata for the DigitalTyphoonDataset

        Does NOT check for file existence until accessing the image.

        :param image_filepath: str, path to image file
        :param track_data: np.ndarray, track data for this image (coordinates, etc.)
        :param sequence_id: str, sequence identifier this image belongs to
        :param load_imgs_into_mem: bool, flag indicating whether images should be loaded into memory
        :param spectrum: str, default spectrum to read the image in
        :param transform_func: function to transform image arrays
        :param image_filepaths: list of image file paths for multi-channel images
        :param verbose: bool, flag for verbose output
        """
        self.sequence_str = sequence_id
        self.verbose = verbose

        self.load_imgs_into_mem = load_imgs_into_mem
        self.spectrum = spectrum
        self.transform_func = transform_func


        self.image_filepath = image_filepath
        self.image_filepaths = image_filepaths
        self.image_array = None

        # Check image_filepath is exists
        if self.image_filepath and not os.path.exists(self.image_filepath):
            raise FileNotFoundError(f"Image file does not exist: {self.image_filepath}")
        if self.verbose:
            print(f"Creating DigitalTyphoonImage: filepath={image_filepath}, load_imgs_into_mem={load_imgs_into_mem}")
        
        if self.image_filepath and os.path.exists(self.image_filepath):
            if self.verbose:
                print(f"Image file exists: {self.image_filepath}")
                
            # If loading into memory is requested, immediately load the image
            if self.load_imgs_into_mem:
                if self.verbose:
                    print(f"Loading image into memory: {self.image_filepath}")
                try:
                    self.image_array = self._get_h5_image_as_numpy(self.image_filepath, self.spectrum)
                    if self.verbose:
                        if self.image_array.size > 0:
                            print(f"Successfully loaded image, shape={self.image_array.shape}")
                        else:
                            print(f"Warning: Loaded image is empty")
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to load image: {str(e)}")
                        import traceback
                        traceback.print_exc()
        elif self.image_filepath and self.verbose:
            print(f"Warning: Image file does not exist: {self.image_filepath}")
        
        # Initialize track_data as empty array if None is provided
        if track_data is None:
            self.track_data = np.array([])
        elif isinstance(track_data, np.ndarray):
            self.track_data = track_data
        elif isinstance(track_data, str):
            # This means track_data was mistakenly passed as a string
            # Store it properly and initialize an empty track array
            if self.verbose:
                print(f"Warning: track_data was passed as a string: {track_data}")
            self.track_data = np.array([])
        else:
            # Try to convert to numpy array, or use empty array if it fails
            try:
                self.track_data = np.array(track_data)
            except:
                if self.verbose:
                    print(f"Warning: Could not convert track_data to numpy array")
                self.track_data = np.array([])
        if self.image_filepath is not None and self.load_imgs_into_mem:
            self.set_image_data(
                self.image_filepath, load_imgs_into_mem=self.load_imgs_into_mem)

    def image(self, spectrum=None) -> np.ndarray:
        """
        Gets this image as a numpy array.

        If an image is already loaded, it is returned. Otherwise, the image is loaded from the h5 file.

        If `self.transform_func` is not None, it is applied before returning the image.

        :return: this image as a numpy array
        """
        if self.image_array is not None:
            image_array = self.image_array
        else:
            # Load the image from file
            if not self.image_filepath:
                if hasattr(self, 'verbose') and self.verbose:
                    print(f"Warning: No image filepath set, returning empty array")
                return np.array([], dtype=np.float64)

            try:
                if spectrum is None:
                    spectrum = self.spectrum
                image_array = self._get_h5_image_as_numpy(self.image_filepath, spectrum)

                if image_array.size == 0:
                    if hasattr(self, 'verbose') and self.verbose:
                        print(f"Warning: Image loaded from {self.image_filepath} is empty")
                    return image_array
                    
                # If we should keep images in memory, store the loaded array
                if self.load_imgs_into_mem:
                    self.image_array = image_array
                    if hasattr(self, 'verbose') and self.verbose:
                        print(f"Loaded image into memory: {self.image_filepath}")
                        
            except Exception as e:
                if hasattr(self, 'verbose') and self.verbose:
                    print(f"Error loading image from {self.image_filepath}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                return np.array([], dtype=np.float64)

        # Apply transform function if defined
        if self.transform_func is not None:
            try:
                image_array = self.transform_func(image_array)
                # If transform changes the array and we're keeping in memory, update stored array
                if self.load_imgs_into_mem:
                    self.image_array = image_array
            except Exception as e:
                if hasattr(self, 'verbose') and self.verbose:
                    print(f"Error applying transform function: {str(e)}")

        return image_array

    def sequence_id(self) -> str:
        """
        Returns the sequence ID this image belongs to

        :return: str sequence str
        """
        return self.sequence_str

    def track_array(self) -> np.ndarray:
        """
        Returns the csv row for this image

        :return: nparray containing the track data
        """
        if self.track_data is None:
            # Return an empty array if no track data exists
            return np.array([])
        return self.track_data

    def value_from_string(self, label):
        """
        Returns the image's value given the label as a string. e.g. value_from_string('month') -> the month

        :return: the element
        """
        try:
            label_name = TRACK_COLS.str_to_value(label)
            value = self.track_array()[label_name]
            # For commonly expected numeric values, try to convert
            if label in ['year', 'month', 'day', 'hour', 'grade', 'pressure', 'wind']:
                try:
                    return float(value) if label not in ['year', 'month', 'day', 'hour', 'grade'] else int(value)
                except (ValueError, TypeError):
                    # Return sensible defaults for common numeric fields
                    if label == 'grade':
                        return 0
                    elif label in ['year', 'month', 'day', 'hour']:
                        return 0 if label == 'year' or label == 'hour' else 1
                    else:
                        return 0.0
            return value
        except (IndexError, ValueError, TypeError, AttributeError):
            # Return sensible defaults for common fields
            if label in ['grade', 'year', 'month', 'day', 'hour']:
                return 0 if label == 'grade' or label == 'year' or label == 'hour' else 1
            elif label in ['pressure', 'wind', 'lat', 'lng', 'dir50', 'long50', 'short50', 'dir30', 'long30', 'short30']:
                return 0.0
            return 0

    def year(self) -> int:
        """
        Returns the year the image was taken

        :return: int, the year
        """
        try:
            return int(self.track_data[TRACK_COLS.YEAR.value])
        except (ValueError, TypeError, IndexError, AttributeError):
            # Return a default value if track data is invalid or unavailable
            return 0

    def month(self) -> int:
        """
        Returns the month the image was taken

        :return: int, the month (1-12)
        """
        try:
            return int(self.track_data[TRACK_COLS.MONTH.value])
        except (ValueError, TypeError, IndexError, AttributeError):
            return 1

    def day(self) -> int:
        """
        Returns the day the image was taken (number 1-31)

        :return: int the day
        """
        try:
            return int(self.track_data[TRACK_COLS.DAY.value])
        except (ValueError, TypeError, IndexError, AttributeError):
            return 1

    def hour(self) -> int:
        """
        Returns the hour the image was taken

        :return: int, the hour
        """
        try:
            return int(self.track_data[TRACK_COLS.HOUR.value])
        except (ValueError, TypeError, IndexError, AttributeError):
            return 0

    def datetime(self) -> datetime:
        """
        Returns a datetime object of when the image was taken.
        If track data is not available or has invalid values, attempts to parse from the filename.

        :return: datetime
        """
        try:
            # Try to get from track data first
            return datetime(self.year(), self.month(), self.day(), self.hour())
        except (ValueError, TypeError, IndexError, AttributeError):
            # If track data fails, try to extract from filename if available
            if hasattr(self, 'image_filepath') and self.image_filepath:
                try:
                    # Try to extract date from filename (assuming format like YYYYMMDDHH-*)
                    filename = os.path.basename(self.image_filepath)
                    date_part = filename.split('-')[0]
                    if len(date_part) >= 10:  # Should be at least 10 chars for YYYYMMDDHH
                        year = int(date_part[0:4])
                        month = int(date_part[4:6])
                        day = int(date_part[6:8])
                        hour = int(date_part[8:10])
                        return datetime(year, month, day, hour)
                except (ValueError, IndexError):
                    pass
            
            # Last resort: return a default datetime to avoid sorting errors
            return datetime(1970, 1, 1)  # Use Unix epoch as default

    def get_datetime(self) -> datetime:
        """
        Compatibility method that calls datetime()
        
        :return: datetime
        """
        return self.datetime()

    def grade(self) -> int:
        """
        Returns the grade for the typhoon at this image, at int in the range 1-6.

        :return: int between 1-6
        """
        try:
            return int(self.track_data[TRACK_COLS.GRADE.value])
        except (ValueError, TypeError, IndexError):
            # Return a default value without logging to reduce noise
            return 0  # Return a default value that won't filter out images

    def lat(self) -> float:
        """
        Gets latitude of typhoon at the time of this image

        :return: latitude as a float
        """
        try:
            if self.track_data is None or len(self.track_data) == 0:
                return 0.0
                
            # Check if we can access LAT.value
            if hasattr(TRACK_COLS, 'LAT') and hasattr(TRACK_COLS.LAT, 'value'):
                lat_idx = TRACK_COLS.LAT.value
                if lat_idx < len(self.track_data):
                    # Try to convert the value to float
                    try:
                        return float(self.track_data[lat_idx])
                    except (ValueError, TypeError):
                        # If conversion fails, return default
                        if self.verbose:
                            print(f"Warning: Could not convert lat value '{self.track_data[lat_idx]}' to float")
                        return 0.0
            
            # Fallback to checking if track_data has keys
            if hasattr(self.track_data, 'keys') and 'lat' in self.track_data:
                try:
                    return float(self.track_data['lat'])
                except (ValueError, TypeError):
                    pass
                    
            return 0.0
        except Exception as e:
            if self.verbose:
                print(f"Error getting latitude: {str(e)}")
            return 0.0

    def long(self) -> float:
        """
        Gets longitude of typhoon at the time of this image

        :return: longitude as a float
        """
        try:
            if self.track_data is None or len(self.track_data) == 0:
                return 0.0
                
            # Check if we can access LONG.value
            if hasattr(TRACK_COLS, 'LNG') and hasattr(TRACK_COLS.LNG, 'value'):
                long_idx = TRACK_COLS.LNG.value
                if long_idx < len(self.track_data):
                    # Try to convert the value to float
                    try:
                        return float(self.track_data[long_idx])
                    except (ValueError, TypeError):
                        # If conversion fails, return default
                        if self.verbose:
                            print(f"Warning: Could not convert long value '{self.track_data[long_idx]}' to float")
                        return 0.0
            
            # Fallback to checking if track_data has keys
            if hasattr(self.track_data, 'keys') and 'long' in self.track_data:
                try:
                    return float(self.track_data['long'])
                except (ValueError, TypeError):
                    pass
                    
            return 0.0
        except Exception as e:
            if self.verbose:
                print(f"Error getting longitude: {str(e)}")
            return 0.0

    def pressure(self) -> float:
        """
        Returns the pressure in # TODO: units? probably hg

        :return: float
        """
        return float(self.track_data[TRACK_COLS.PRESSURE.value])

    def wind(self) -> float:
        """
        Returns the wind speed in # TODO: units?

        :return: float
        """
        try:
            if self.track_data is None or len(self.track_data) == 0:
                return 0.0
                
            # Check if we can access WIND.value
            if hasattr(TRACK_COLS, 'WIND') and hasattr(TRACK_COLS.WIND, 'value'):
                wind_idx = TRACK_COLS.WIND.value
                if wind_idx < len(self.track_data):
                    # Try to convert the value to float
                    try:
                        return float(self.track_data[wind_idx])
                    except (ValueError, TypeError):
                        # If conversion fails, return default
                        if self.verbose:
                            print(f"Warning: Could not convert wind value '{self.track_data[wind_idx]}' to float")
                        return 0.0
            
            # Fallback to checking if track_data has keys
            if hasattr(self.track_data, 'keys') and 'wind' in self.track_data:
                try:
                    return float(self.track_data['wind'])
                except (ValueError, TypeError):
                    pass
                    
            return 0.0
        except Exception as e:
            if self.verbose:
                print(f"Error getting wind speed: {str(e)}")
            return 0.0

    def dir50(self) -> float:
        """
        # TODO: what is this?

        :return: float
        """
        try:
            if self.track_data is None or len(self.track_data) == 0:
                return 0.0
                
            if hasattr(TRACK_COLS, 'DIR50') and hasattr(TRACK_COLS.DIR50, 'value'):
                idx = TRACK_COLS.DIR50.value
                if idx < len(self.track_data):
                    try:
                        return float(self.track_data[idx])
                    except (ValueError, TypeError):
                        if self.verbose:
                            print(f"Warning: Could not convert DIR50 value to float")
                        return 0.0
            return 0.0
        except Exception as e:
            if self.verbose:
                print(f"Error getting DIR50: {str(e)}")
            return 0.0

    def long50(self) -> float:
        """
        # TODO: what is this?

        :return: float
        """
        try:
            if self.track_data is None or len(self.track_data) == 0:
                return 0.0
                
            if hasattr(TRACK_COLS, 'LONG50') and hasattr(TRACK_COLS.LONG50, 'value'):
                idx = TRACK_COLS.LONG50.value
                if idx < len(self.track_data):
                    try:
                        return float(self.track_data[idx])
                    except (ValueError, TypeError):
                        if self.verbose:
                            print(f"Warning: Could not convert LONG50 value to float")
                        return 0.0
            return 0.0
        except Exception as e:
            if self.verbose:
                print(f"Error getting LONG50: {str(e)}")
            return 0.0

    def short50(self) -> float:
        """
        # TODO: what is this?

        :return: float
        """
        try:
            if self.track_data is None or len(self.track_data) == 0:
                return 0.0
                
            if hasattr(TRACK_COLS, 'SHORT50') and hasattr(TRACK_COLS.SHORT50, 'value'):
                idx = TRACK_COLS.SHORT50.value
                if idx < len(self.track_data):
                    try:
                        return float(self.track_data[idx])
                    except (ValueError, TypeError):
                        if self.verbose:
                            print(f"Warning: Could not convert SHORT50 value to float")
                        return 0.0
            return 0.0
        except Exception as e:
            if self.verbose:
                print(f"Error getting SHORT50: {str(e)}")
            return 0.0

    def dir30(self) -> float:
        """
        # TODO: what is this?

        :return: float
        """
        try:
            if self.track_data is None or len(self.track_data) == 0:
                return 0.0
                
            if hasattr(TRACK_COLS, 'DIR30') and hasattr(TRACK_COLS.DIR30, 'value'):
                idx = TRACK_COLS.DIR30.value
                if idx < len(self.track_data):
                    try:
                        return float(self.track_data[idx])
                    except (ValueError, TypeError):
                        if self.verbose:
                            print(f"Warning: Could not convert DIR30 value to float")
                        return 0.0
            return 0.0
        except Exception as e:
            if self.verbose:
                print(f"Error getting DIR30: {str(e)}")
            return 0.0

    def long30(self) -> float:
        """
        # TODO: what is this?

        :return: float
        """
        try:
            if self.track_data is None or len(self.track_data) == 0:
                return 0.0
                
            if hasattr(TRACK_COLS, 'LONG30') and hasattr(TRACK_COLS.LONG30, 'value'):
                idx = TRACK_COLS.LONG30.value
                if idx < len(self.track_data):
                    try:
                        return float(self.track_data[idx])
                    except (ValueError, TypeError):
                        if self.verbose:
                            print(f"Warning: Could not convert LONG30 value to float")
                        return 0.0
            return 0.0
        except Exception as e:
            if self.verbose:
                print(f"Error getting LONG30: {str(e)}")
            return 0.0

    def short30(self) -> float:
        """
        # TODO: what is this?

        :return: float
        """
        try:
            if self.track_data is None or len(self.track_data) == 0:
                return 0.0
                
            if hasattr(TRACK_COLS, 'SHORT30') and hasattr(TRACK_COLS.SHORT30, 'value'):
                idx = TRACK_COLS.SHORT30.value
                if idx < len(self.track_data):
                    try:
                        return float(self.track_data[idx])
                    except (ValueError, TypeError):
                        if self.verbose:
                            print(f"Warning: Could not convert SHORT30 value to float")
                        return 0.0
            return 0.0
        except Exception as e:
            if self.verbose:
                print(f"Error getting SHORT30: {str(e)}")
            return 0.0

    def landfall(self) -> float:
        """
        # TODO: what is this?

        :return: float
        """
        try:
            if self.track_data is None or len(self.track_data) == 0:
                return 0.0
                
            if hasattr(TRACK_COLS, 'LANDFALL') and hasattr(TRACK_COLS.LANDFALL, 'value'):
                idx = TRACK_COLS.LANDFALL.value
                if idx < len(self.track_data):
                    try:
                        return float(self.track_data[idx])
                    except (ValueError, TypeError):
                        if self.verbose:
                            print(f"Warning: Could not convert LANDFALL value to float")
                        return 0.0
            return 0.0
        except Exception as e:
            if self.verbose:
                print(f"Error getting LANDFALL: {str(e)}")
            return 0.0

    def interpolated(self) -> bool:
        """
        Returns whether this entry is interpolated or not

        :return: bool
        """
        try:
            if self.track_data is None or len(self.track_data) == 0:
                return False
                
            if hasattr(TRACK_COLS, 'INTERPOLATED') and hasattr(TRACK_COLS.INTERPOLATED, 'value'):
                idx = TRACK_COLS.INTERPOLATED.value
                if idx < len(self.track_data):
                    try:
                        return bool(self.track_data[idx])
                    except (ValueError, TypeError):
                        if self.verbose:
                            print(f"Warning: Could not convert INTERPOLATED value to bool")
                        return False
            return False
        except Exception as e:
            if self.verbose:
                print(f"Error getting INTERPOLATED: {str(e)}")
            return False

    def filepath(self) -> str:
        """
        Returns the filepath to the image

        :return: str
        """
        return str(self.image_filepath)

    def mask_1(self) -> float:
        """
        Returns number of pixels in the image that are corrupted

        :return: float the number of pixels
        """
        return float(self.track_data[TRACK_COLS.MASK_1.value])

    def mask_1_percent(self) -> float:
        """
        Returns percentage of pixels in the image that are corrupted

        :return: float the percentage of pixels
        """
        return float(self.track_data[TRACK_COLS.MASK_1_PERCENT.value])

    def set_track_data(self, track_entry: np.ndarray) -> None:
        """
        Sets the track entry

        :param track_entry: numpy array representing one entry of the track csv
        :return: None
        """
        # if len(track_entry) != len(TRACK_COLS):
        #     raise ValueError(f'Number of columns in the track entry ({len(track_entry)}) is not equal '
        #                      f'to expected amount ({len(TRACK_COLS)})')
        self.track_data = track_entry

    def set_image_data(self, image_filepath: str, load_imgs_into_mem=False, spectrum=None) -> None:
        """
        Sets the image file data

        :param load_imgs_into_mem: bool, whether to load images into memory
        :param spectrum: str, spectrum to open h5 images with
        :param image_filepath: string to image
        :return: None
        """
        self.load_imgs_into_mem = load_imgs_into_mem
        if spectrum is None:
            spectrum = self.spectrum

        self.image_filepath = image_filepath
        self.image_filepaths = None
        self.image_array = None

        # Check if file exists, but don't raise an exception
        # Just warn if verbose is enabled
        if self.image_filepath and not os.path.exists(self.image_filepath):
            if self.verbose:
                print(f"Warning: Image file does not exist: {self.image_filepath}")

        if self.load_imgs_into_mem:
            # Load the image on instantiation if load_imgs_into_mem is set to True
            self.image()

    def set_image_datas(self, image_filepaths: List[str], load_imgs_into_mem=False, spectrum=None) -> None:
        """
        Sets multiple image filepaths for multi-channel loading.
        
        :param image_filepaths: List of paths to h5 image files (one per channel)
        :param load_imgs_into_mem: Bool indicating if the images should be loaded into memory
        :param spectrum: String indicating what spectrum the images are in
        :return: None
        """
        if spectrum is not None:
            self.spectrum = spectrum
        
        # Validate inputs
        if not image_filepaths:
            print(f"Error: No image filepaths provided for sequence {self.sequence_str}")
            return
        
        # Check that all files exist
        missing_files = []
        for filepath in image_filepaths:
            if not os.path.exists(filepath) or not os.path.isfile(filepath):
                missing_files.append(filepath)
        
        if missing_files:
            print(f"Warning: {len(missing_files)} files don't exist: {missing_files[:2]}...")
        
        # Keep only files that exist
        valid_filepaths = [f for f in image_filepaths if os.path.exists(f) and os.path.isfile(f)]
        
        if not valid_filepaths:
            print(f"Error: No valid image files found from {len(image_filepaths)} provided paths")
            return
        
        # Store the filepaths
        self.image_filepaths = valid_filepaths
        self.load_imgs_into_mem = load_imgs_into_mem
        
        # Reset the image array to force reloading
        self.image_array = None
        
        # Load all channels into memory if requested
        if load_imgs_into_mem:
            try:
                self.image_array = self.image()
                print(f"Loaded {len(valid_filepaths)} channels into memory, shape: {self.image_array.shape}")
            except Exception as e:
                print(f"Error loading images into memory: {str(e)}")
                # Reset the array since loading failed
                self.image_array = None

    def get_multiimage_data(self, image_filepaths: List[str], load_imgs_into_mem=False, spectrum=None):
        self.load_imgs_into_mem = load_imgs_into_mem
        if spectrum is None:
            spectrum = self.spectrum
        self.image_filepaths = image_filepaths
        if self.load_imgs_into_mem:
            self.image()

    def _get_h5_image_as_numpy(self, image_filepath: str, spectrum=None) -> np.ndarray:
        """
        Reads a single h5 image at the specified filepath as a numpy array.
        
        :param image_filepath: str, path to h5 image file
        :param spectrum: The spectrum (channel) to read from multi-channel images
        :return: np.ndarray of the image data
        """
        if spectrum is None:
            spectrum = self.spectrum

        # Handle multi-channel case (list of filepaths)
        if self.image_filepaths is not None and len(self.image_filepaths) > 0:
            try:
                multi_channel_data = []
                
                for filepath in self.image_filepaths:
                    # Read each file as a separate channel
                    if not os.path.exists(filepath):
                        continue
                        
                    with h5py.File(filepath, 'r') as h5file:
                        # For h5 files, get the data from the first dataset
                        keys = list(h5file.keys())
                        if not keys:
                            continue
                            
                        dataset_name = keys[0]
                        channel_data = np.array(h5file[dataset_name])
                        multi_channel_data.append(channel_data)
                        
                if not multi_channel_data:
                    return np.array([])
                    
                # Stack along channel dimension
                result = np.stack(multi_channel_data, axis=0)
                return result
            except Exception as e:
                # Only print errors for serious issues
                print(f"Error loading multi-channel image: {str(e)}")
                return np.array([])
        
        # Handle single-channel case
        if image_filepath is None:
            return np.array([])
        
        if not os.path.exists(image_filepath):
            return np.array([])
        
        try:
            with h5py.File(image_filepath, 'r') as h5file:
                # For h5 files, get the data from the first dataset
                keys = list(h5file.keys())
                if not keys:
                    return np.array([])
                    
                dataset_name = keys[0]
                result = np.array(h5file[dataset_name])
                return result
        except Exception as e:
            # Only print errors for serious issues if in verbose mode
            if hasattr(self, 'verbose') and self.verbose:
                print(f"Error loading image {os.path.basename(image_filepath)}: {str(e)}")
            return np.array([])

    def debug_track_data(self) -> None:
        """
        Print detailed diagnostic information about the track data.
        Useful for debugging missing or incorrect values.
        
        :return: None
        """
        print("\n=== TRACK DATA DIAGNOSTICS ===")
        print(f"Sequence ID: {self.sequence_str}")
        print(f"Track data type: {type(self.track_data)}")
        print(f"Track data length: {len(self.track_data) if hasattr(self.track_data, '__len__') else 'N/A'}")
        print(f"Track data content: {self.track_data}")
        
        print("\nImage Data Status:")
        print(f"  Image filepath: {self.image_filepath}")
        print(f"  Image exists: {os.path.exists(self.image_filepath) if self.image_filepath else False}")
        print(f"  load_imgs_into_mem setting: {self.load_imgs_into_mem}")
        print(f"  Image preloaded: {self.image_array is not None}")
        if self.image_array is not None:
            print(f"  Image array shape: {self.image_array.shape}")
            print(f"  Image array dtype: {self.image_array.dtype}")
            print(f"  Image array non-zero: {np.any(self.image_array != 0)}")
        
        if len(self.track_data) > 0:
            print("\nTrack column values:")
            for col_name in dir(TRACK_COLS):
                if col_name.startswith('_') or not col_name.isupper():
                    continue
                
                col = getattr(TRACK_COLS, col_name)
                if hasattr(col, 'value'):
                    col_idx = col.value
                    print(f"  {col_name} (index {col_idx}):", end=" ")
                    
                    try:
                        if col_idx < len(self.track_data):
                            value = self.track_data[col_idx]
                            print(f"{value} (type: {type(value)})")
                        else:
                            print("INDEX OUT OF BOUNDS")
                    except Exception as e:
                        print(f"ERROR: {str(e)}")
        
        print("\nMethod outputs:")
        try:
            print(f"  year(): {self.year()}")
        except Exception as e:
            print(f"  year() ERROR: {str(e)}")
            
        try:
            print(f"  month(): {self.month()}")
        except Exception as e:
            print(f"  month() ERROR: {str(e)}")
            
        try:
            print(f"  wind(): {self.wind()}")
        except Exception as e:
            print(f"  wind() ERROR: {str(e)}")
            
        try:
            print(f"  long(): {self.long()}")
        except Exception as e:
            print(f"  long() ERROR: {str(e)}")
            
        print("\nImage information:")
        print(f"  Has image path: {self.image_filepath is not None}")
        if self.image_filepath:
            print(f"  Image path: {self.image_filepath}")
            print(f"  Image exists: {os.path.exists(self.image_filepath)}")
        print("==========================\n")
