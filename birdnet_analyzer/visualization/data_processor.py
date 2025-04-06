"""
DataProcessor class for handling and transforming prediction data.

This module defines the DataProcessor class, which processes prediction data
from one or multiple files, prepares a consolidated DataFrame, and provides
methods for filtering that data.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

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

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares a DataFrame by extracting recording filenames and adding them as columns.
        """
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

        return df

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
