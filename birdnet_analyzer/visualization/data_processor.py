"""
DataProcessor class for handling and transforming prediction data.

This module defines the DataProcessor class, which processes prediction data
from one or multiple files, prepares a consolidated DataFrame, and provides
methods for filtering that data.
"""

import os
import datetime
import re
from typing import Dict, List, Optional, Tuple

import warnings
import numpy as np
import pandas as pd
import gradio as gr  # Add this import

from birdnet_analyzer.evaluation.preprocessing.utils import (
    extract_recording_filename,
    extract_recording_filename_from_filename,
    read_and_concatenate_files_in_directory,
)


class DataProcessor:
    """
    Processor for handling and transforming prediction data.

    This class loads prediction files (either a single file or all files in
    a specified directory), prepares them into a unified DataFrame, and
    provides methods to filter the prediction data by recording, class,
    or confidence.
    """

    # Default column mappings for predictions
    DEFAULT_COLUMNS_PREDICTIONS = {
        "Start Time": "Start Time",
        "End Time": "End Time",
        "Class": "Class",
        "Recording": "Recording",
        "Duration": "Duration",
        "Confidence": "Confidence",
    }

    def __init__(
        self,
        prediction_directory_path: str,
        prediction_file_name: Optional[str] = None,
        columns_predictions: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initializes the DataProcessor by loading and preparing prediction data.

        Args:
            prediction_directory_path (str): Path to the folder containing prediction files.
            prediction_file_name (Optional[str]): Name of a single prediction file to process.
                If None, all `.csv` or `.tsv` files in the directory will be loaded.
            columns_predictions (Optional[Dict[str, str]], optional): Custom column mappings for
                prediction files (e.g., {"Start Time": "begin", "End Time": "end"}). If None,
                default mappings are used.
        """
        # Paths and filenames
        self.prediction_directory_path: str = prediction_directory_path
        self.prediction_file_name: Optional[str] = prediction_file_name

        # Use provided column mappings or defaults
        self.columns_predictions: Dict[str, str] = (
            columns_predictions if columns_predictions is not None
            else self.DEFAULT_COLUMNS_PREDICTIONS.copy()
        )

        # Internal DataFrame to hold all predictions
        self.predictions_df: pd.DataFrame = pd.DataFrame()

        # Metadata DataFrame
        self.metadata_df: Optional[pd.DataFrame] = None

        # Validate column mappings
        self._validate_columns()

        # Load and prepare data
        self.load_data()
        self.predictions_df = self._prepare_dataframe(self.predictions_df)

        # Ensure that the confidence column is numeric.
        conf_col = self.get_column_name("Confidence")
        if conf_col in self.predictions_df.columns:
            self.predictions_df[conf_col] = self.predictions_df[conf_col].astype(float)

        # Gather unique classes (if "Class" column exists)
        class_col = self.get_column_name("Class")
        if class_col in self.predictions_df.columns:
            self.classes = tuple(
                sorted(self.predictions_df[class_col].dropna().unique())
            )
        else:
            self.classes = tuple()

    def _validate_columns(self) -> None:
        """
        Validates that essential columns are provided in the prediction column mappings.

        Raises:
            ValueError: If required columns are missing or have None values.
        """
        # Required columns for predictions
        required_columns = ["Start Time", "End Time", "Class"]

        missing_pred_columns = [
            col
            for col in required_columns
            if col not in self.columns_predictions or self.columns_predictions[col] is None
        ]
        if missing_pred_columns:
            raise ValueError(f"Missing or None prediction columns: {', '.join(missing_pred_columns)}")

    def load_data(self) -> None:
        """
        Loads the prediction data into a DataFrame.

        - If `prediction_file_name` is None, all CSV/TSV files in `prediction_directory_path`
          are concatenated.
        - Otherwise, only the specified file is read.
        """
        if self.prediction_file_name is None:
            # Load all files in the directory
            self.predictions_df = read_and_concatenate_files_in_directory(
                self.prediction_directory_path
            )
        else:
            # Load a single specified file
            full_path = os.path.join(self.prediction_directory_path, self.prediction_file_name)
            # Attempt TSV read first; if it fails, try CSV
            try:
                self.predictions_df = pd.read_csv(full_path, sep="\t")
            except pd.errors.ParserError:
                self.predictions_df = pd.read_csv(full_path)

        # Ensure 'source_file' column exists for traceability
        if "source_file" not in self.predictions_df.columns:
            # If a single file was loaded, each row is from that file
            default_source = self.prediction_file_name if self.prediction_file_name else ""
            self.predictions_df["source_file"] = default_source

    def _extract_datetime_from_filename(self, filename: str) -> Tuple[str, datetime.datetime, str]:
        """
        Extracts site name and datetime from filename using strict format:
        SITE_SAMPLERATE_SITEID_YYYYMMDD_HHMMSS.wav
        
        Returns: (site_name, datetime_obj, original_filename)
        """
        if not isinstance(filename, str):
            return ("", None, str(filename))
            
        # Strict pattern matching the required format
        pattern = r"^([^_]+)_([^_]+)_([^_]+)_(\d{8})_(\d{6})(?:\.wav)?$"
        match = re.match(pattern, filename)
        if not match:
            return ("", None, filename)
        
        _, _, site_id, date_str, time_str = match.groups()
        try:
            date_time = datetime.datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            return (site_id, date_time, filename)
        except ValueError:
            return ("", None, filename)

    def _add_metadata_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds metadata information (latitude, longitude) to the DataFrame."""
        if self.metadata_df is None or df.empty:
            return df

        # Create a copy and extract site names
        df = df.copy()
        df['site_name'] = df['recording_filename'].apply(
            lambda x: self._extract_datetime_from_filename(x)[0]
        )

        # Get valid sites and find missing ones
        valid_sites = set(self.metadata_df['site_name'].unique())
        found_sites = set(df['site_name'].unique())
        missing_sites = found_sites - valid_sites
        
        if missing_sites:
            missing_sites_str = sorted(list(missing_sites))
            gr.Warning(f"Site IDs not found in metadata: {missing_sites_str}")

        # Filter to valid sites
        df = df[df['site_name'].isin(valid_sites)]

        if df.empty:
            return df

        # Merge with metadata
        df = pd.merge(
            df,
            self.metadata_df[['site_name', 'latitude', 'longitude']],
            on='site_name',
            how='inner'  # Only keep records with matching site names
        )

        return df

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced preparation of DataFrame including datetime processing.
        """
        # First do the original preparation
        recording_col = self.get_column_name("Recording")
        if recording_col in df.columns:
            df["recording_filename"] = extract_recording_filename(df[recording_col])
            # Add filename column as an alias for compatibility
            df["filename"] = df["recording_filename"]
        else:
            if "source_file" in df.columns:
                df["recording_filename"] = extract_recording_filename_from_filename(df["source_file"])
                df["filename"] = df["recording_filename"]
            else:
                df["recording_filename"] = ""
                df["filename"] = ""

        # Extract datetime information
        datetime_info = df['recording_filename'].apply(self._extract_datetime_from_filename)
        df['site_name'] = datetime_info.apply(lambda x: x[0])
        df['recording_datetime'] = datetime_info.apply(lambda x: x[1])
        
        # Calculate actual prediction times
        start_time_col = self.get_column_name("Start Time")
        if start_time_col in df.columns and 'recording_datetime' in df.columns:
            df['prediction_time'] = df.apply(
                lambda row: row['recording_datetime'] + 
                          datetime.timedelta(seconds=float(row[start_time_col]))
                if pd.notnull(row['recording_datetime']) else None,
                axis=1
            )
        
        # Add metadata information
        df = self._add_metadata_info(df)
        
        return df

    def set_metadata(self, metadata_df: pd.DataFrame, 
                    site_col: str = 'Site',
                    lat_col: str = 'Latitude',
                    lon_col: str = 'Longitude') -> None:
        """
        Sets the metadata DataFrame with standardized column names.
        """
        # Ensure the source columns exist
        if not all(col in metadata_df.columns for col in [site_col, lat_col, lon_col]):
            missing = [col for col in [site_col, lat_col, lon_col] if col not in metadata_df.columns]
            raise ValueError(f"Missing columns in metadata: {missing}")

        # Create a copy to avoid modifying the original
        self.metadata_df = metadata_df.copy()
        
        # Rename columns
        column_mapping = {
            site_col: 'site_name',
            lat_col: 'latitude',
            lon_col: 'longitude'
        }
        self.metadata_df = self.metadata_df.rename(columns=column_mapping)
        
        # Apply coordinate conversion if numeric
        self.metadata_df['latitude'] = pd.to_numeric(self.metadata_df['latitude'], errors='coerce')
        self.metadata_df['longitude'] = pd.to_numeric(self.metadata_df['longitude'], errors='coerce')

    def get_column_name(self, field_name: str, prediction: bool = True) -> str:
        """
        Retrieves the appropriate column name for the specified field.

        Args:
            field_name (str): The name of the field (e.g., "Class", "Start Time").
            prediction (bool): Whether to fetch from predictions mapping (True)
                             or annotations mapping (False). 
                             In visualization, this parameter is ignored since we only 
                             have prediction data.

        Returns:
            str: The column name corresponding to the field.

        Raises:
            TypeError: If field_name is None.
        """
        if field_name is None:
            raise TypeError("field_name cannot be None.")

        if field_name in self.columns_predictions and self.columns_predictions[field_name] is not None:
            return self.columns_predictions[field_name]

        return field_name

    def get_data(self) -> pd.DataFrame:
        """
        Retrieves a copy of the prediction DataFrame.

        Returns:
            pd.DataFrame: A copy of the `predictions_df`.
        """
        return self.predictions_df.copy()

    def filter_data(
        self,
        selected_recordings: Optional[List[str]] = None,
        selected_classes: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Returns a filtered version of the prediction DataFrame.

        Args:
            selected_recordings (List[str], optional): A list of recording filenames to include.
                If None, no filtering by recording is applied.
            selected_classes (List[str], optional): A list of classes to include.
                If None, no filtering by class is applied.
            min_confidence (float, optional): Minimum confidence threshold for inclusion.
                If None, no filtering by confidence is applied.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        df = self.get_data()  # Work on a copy

        # Filter by recordings
        if selected_recordings is not None and "recording_filename" in df.columns:
            df = df[df["recording_filename"].isin(selected_recordings)]

        # Filter by classes
        class_col = self.get_column_name("Class")
        if selected_classes is not None and class_col in df.columns:
            df = df[df[class_col].isin(selected_classes)]

        # Filter by confidence
        confidence_col = self.get_column_name("Confidence")
        if min_confidence is not None and confidence_col in df.columns:
            df = df[df[confidence_col] >= min_confidence]

        return df

    def get_aggregated_locations(self, 
                               selected_classes: Optional[List[str]] = None) -> pd.DataFrame:
        """Returns aggregated prediction counts by location and class."""
        df = self.get_data()
        
        # Apply metadata and remove invalid records
        df = self._add_metadata_info(df)
        if df.empty:
            raise ValueError("No valid predictions with matching site IDs found")

        class_col = self.get_column_name("Class")
        if selected_classes:
            df = df[df[class_col].isin(selected_classes)]
            
        # Group by location and class, count occurrences
        agg_df = df.groupby([
            'site_name',
            'latitude',
            'longitude',
            class_col
        ]).size().reset_index(name='count')
        
        return agg_df
