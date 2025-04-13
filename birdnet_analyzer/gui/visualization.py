import json
import os
import shutil
import tempfile
import typing
import io
from pathlib import Path
import datetime

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer.visualization.data_processor import DataProcessor
from birdnet_analyzer.visualization.plotting.confidences import ConfidencePlotter
from birdnet_analyzer.visualization.utils.coordinates import process_coordinates


class ProcessorState(typing.NamedTuple):
    """State of the DataProcessor."""
    processor: DataProcessor
    prediction_dir: str
    metadata_dir: str
    color_map: typing.Optional[typing.Dict[str, str]] = None


def get_date_range(df: pd.DataFrame) -> tuple:
    """Get the earliest and latest dates from predictions."""
    try:
        if 'prediction_time' not in df.columns or df.empty:
            return None, None
        
        min_date = df['prediction_time'].min()
        max_date = df['prediction_time'].max()
        
        if pd.isna(min_date) or pd.isna(max_date):
            return None, None
            
        # Set time to start and end of day
        start_date = min_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = max_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        return start_date, end_date
    except Exception:
        return None, None


def convert_timestamp_to_datetime(timestamp):
    """Convert Gradio DateTime timestamp to pandas datetime."""
    if timestamp is None:
        return None
    try:
        # Convert timestamp to pandas datetime
        return pd.to_datetime(timestamp, unit='s')
    except:
        return None


def apply_datetime_filters(df, date_range_start, date_range_end, 
                         time_start_hour, time_start_minute, 
                         time_end_hour, time_end_minute):
    """Apply date and time filters to DataFrame."""
    if df.empty or 'prediction_time' not in df.columns:
        return df

    # Create a copy to avoid modifying original
    filtered_df = df.copy()
    
    # Apply date range filter if dates are provided
    if date_range_start is not None and date_range_end is not None:
        try:
            filtered_df = filtered_df[filtered_df['prediction_time'].notna()]
            start_date = convert_timestamp_to_datetime(date_range_start)
            end_date = convert_timestamp_to_datetime(date_range_end)
            
            if start_date and end_date:
                # Convert to pandas datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(filtered_df['prediction_time']):
                    filtered_df['prediction_time'] = pd.to_datetime(filtered_df['prediction_time'])
                
                # Fix timezone issues by using normalized dates
                start_date = pd.Timestamp(start_date.date()) 
                end_date = pd.Timestamp(end_date.date()) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                
                # Filter by date range
                filtered_df = filtered_df[filtered_df['prediction_time'].dt.normalize().between(start_date, end_date)]
        
        except Exception as e:
            print(f"Date filtering error: {e}")
            return filtered_df

    # Apply time range filter if all time components are provided
    if all(x is not None for x in [time_start_hour, time_start_minute, time_end_hour, time_end_minute]):
        try:
            # Convert times to datetime.time objects
            start_time = datetime.time(int(time_start_hour), int(time_start_minute), 0, 0)
            end_time = datetime.time(int(time_end_hour), int(time_end_minute), 59, 999999)
            
            # Filter by time of day if prediction_time is datetime
            if pd.api.types.is_datetime64_any_dtype(filtered_df['prediction_time']):
                filtered_df = filtered_df[filtered_df['prediction_time'].dt.time.between(start_time, end_time)]
                
        except Exception as e:
            print(f"Time filtering error: {e}")
            return filtered_df

    return filtered_df


def build_visualization_tab():
    """
    Builds a Gradio tab for loading and plotting prediction data only,
    using ConfidencePlotter. Annotation logic and metric calculations
    have been removed.
    """

    # Default columns for predictions (kept for user convenience)
    prediction_default_columns = {
        "Start Time": "Begin Time (s)",
        "End Time": "End Time (s)",
        "Class": "Common Name",
        "Recording": "Begin File",
        "Confidence": "Confidence",
    }

    # Default columns for metadata
    metadata_default_columns = {
        "Site": "Site",
        "Country": "country",
        "X": "N",
        "Y": "W",
    }

    localized_column_labels = {
        "Start Time": loc.localize("eval-tab-column-start-time-label"),
        "End Time": loc.localize("eval-tab-column-end-time-label"),
        "Class": loc.localize("eval-tab-column-class-label"),
        "Recording": loc.localize("eval-tab-column-recording-label"),
        "Confidence": loc.localize("eval-tab-column-confidence-label"),
        "Site": loc.localize("viz-tab-column-site-label"),
        "Country": loc.localize("viz-tab-column-country-label"),
        "X": loc.localize("viz-tab-column-x-label"),
        "Y": loc.localize("viz-tab-column-y-label"),
    }

    def get_columns_from_uploaded_files(files):
        """
        Reads the header row of each file to discover available columns.
        """
        columns = set()
        if files:
            for file_obj in files:
                try:
                    df = pd.read_csv(file_obj, sep=None, engine="python", nrows=0)
                    columns.update(df.columns)
                except Exception as e:
                    print(f"Error reading file {file_obj}: {e}")
                    gr.Warning(f"{loc.localize('eval-tab-warning-error-reading-file')} {file_obj}")
        return sorted(list(columns))

    def save_uploaded_files(files):
        """
        Saves uploaded files into a temporary directory and returns its path.
        """
        if not files:
            return None
        temp_dir = tempfile.mkdtemp()
        for file_obj in files:
            dest_path = os.path.join(temp_dir, os.path.basename(file_obj))
            shutil.copy(file_obj, dest_path)
        return temp_dir

    def initialize_processor(
        prediction_files,
        pred_start_time,
        pred_end_time,
        pred_class,
        pred_confidence,
        pred_recording,
        prediction_dir=None,
    ):
        """Creates a simplified DataProcessor for predictions only."""
        if not prediction_files:
            return [], [], None, None

        try:
            # Initialize processor
            if prediction_dir is None:
                prediction_dir = save_uploaded_files(prediction_files)

            # Debug just number of files
            print(f"\nProcessing {len(prediction_files)} prediction files")
            print(f"Using prediction directory: {prediction_dir}")

            # Set up column mappings with proper fallbacks
            cols_pred = {}
            for key, default in prediction_default_columns.items():
                if key == "Start Time":
                    cols_pred[key] = pred_start_time or default
                elif key == "End Time":
                    cols_pred[key] = pred_end_time or default
                elif key == "Class":
                    cols_pred[key] = pred_class or default
                elif key == "Confidence":
                    cols_pred[key] = pred_confidence or default
                elif key == "Recording":
                    cols_pred[key] = pred_recording or default

            print("Using column mappings:", cols_pred)

            proc = DataProcessor(
                prediction_directory_path=prediction_dir,
                prediction_file_name=None,
                columns_predictions=cols_pred,
            )
            
            # Get data and extract unique values
            df = proc.get_data()
            
            # Debug dataframe shape and columns
            print(f"\nLoaded DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {df.columns}")
            
            avail_classes = list(proc.classes) 
            print(f"\nFound {len(avail_classes)} unique classes")
            
            # Get and clean up recording names consistently (strip and lowercase)
            recordings = df["recording_filename"].dropna().unique()
            avail_recordings = []
            for rec in recordings:
                if isinstance(rec, str):
                    clean_name = os.path.splitext(os.path.basename(rec.strip()))[0].lower()
                    if clean_name and clean_name not in avail_recordings:
                        avail_recordings.append(clean_name)
            avail_recordings.sort()
            print(f"Found {len(avail_recordings)} unique recordings")

            return avail_classes, avail_recordings, proc, prediction_dir

        except Exception as e:
            print(f"Error in initialize_processor: {e}")
            raise gr.Error(f"Error initializing processor: {str(e)}")

    def update_prediction_columns(uploaded_files):
        """
        Called when user selects prediction files. Reads headers and updates dropdowns.
        """
        cols = get_columns_from_uploaded_files(uploaded_files)
        cols = [""] + cols
        updates = []
        for label in ["Start Time", "End Time", "Class", "Confidence", "Recording"]:
            default_val = prediction_default_columns.get(label)
            val = default_val if default_val in cols else None
            updates.append(gr.update(choices=cols, value=val))
        return updates

    def update_metadata_columns(uploaded_files):
        """
        Called when user selects metadata files. Reads headers and updates dropdowns.
        """
        cols = set()
        if uploaded_files:
            for file_obj in uploaded_files:
                try:
                    # Try reading with CSV engine first
                    df = pd.read_csv(file_obj, nrows=0)
                    cols.update(df.columns)
                except Exception:
                    try:
                        # Fallback to python engine with automatic delimiter detection
                        df = pd.read_csv(file_obj, sep=None, engine="python", nrows=0)
                        cols.update(df.columns)
                    except Exception as e:
                        print(f"Error reading file {file_obj}: {e}")
                        gr.Warning(f"{loc.localize('eval-tab-warning-error-reading-file')} {file_obj}")
        
        cols = [""] + sorted(list(cols))
        updates = []
        for label in ["Site", "Country", "X", "Y"]:
            default_val = metadata_default_columns.get(label)
            val = default_val if default_val in cols else None
            updates.append(gr.update(choices=cols, value=val))
        return updates

    def update_selections(
        prediction_files,
        metadata_files,
        pred_start_time,
        pred_end_time,
        pred_class,
        pred_confidence,
        pred_recording,
        meta_site,
        meta_country,
        meta_x,
        meta_y,
        coordinate_format,
        current_classes,
        current_recordings,
    ):
        """
        Called to create or update the processor based on current file uploads and column selections.
        Preserves any selected classes/recordings that remain valid.
        """
        prediction_dir = save_uploaded_files(prediction_files)
        metadata_dir = save_uploaded_files(metadata_files)
        
        avail_classes, avail_recordings, proc, prediction_dir = initialize_processor(
            prediction_files,
            pred_start_time,
            pred_end_time,
            pred_class,
            pred_confidence,
            pred_recording,
            prediction_dir,
        )
        
        state = ProcessorState(proc, prediction_dir, metadata_dir) if proc else None

        # Keep current selections if they exist in available options
        new_classes = []
        new_recordings = []

        if current_classes:
            new_classes = [c for c in current_classes if c in avail_classes]
        if current_recordings:
            normalized_current = [os.path.splitext(os.path.basename(r.strip()))[0].lower() 
                               for r in current_recordings if isinstance(r, str)]
            new_recordings = [r for r in normalized_current if r in avail_recordings]

        # Default to all available if no valid selections remain
        if not new_classes:
            new_classes = avail_classes
        if not new_recordings:
            new_recordings = avail_recordings

        return (
            gr.update(choices=avail_classes, value=new_classes),
            gr.update(choices=avail_recordings, value=new_recordings),
            state,
        )

    def update_datetime_defaults(processor_state):
        """Update the default date range based on available predictions."""
        if not processor_state or not processor_state.processor:
            return [gr.update()] * 6  # Updated for 6 outputs (2 dates + 4 time dropdowns)
        
        df = processor_state.processor.get_data()
        start_date, end_date = get_date_range(df)
        
        return [
            gr.update(value=start_date),
            gr.update(value=end_date),
            gr.update(value="00"),  # Start hour
            gr.update(value="00"),  # Start minute
            gr.update(value="23"),  # End hour
            gr.update(value="59"),  # End minute
        ]

    def combine_time_components(hour, minute):
        """Combine hour and minute components into a time string."""
        return f"{hour}:{minute}"

    def plot_predictions_action(
        proc_state: ProcessorState,
        selected_classes_list,
        selected_recordings_list,
        confidence_threshold: float,
        date_range_start,
        date_range_end,
        time_start_hour,
        time_start_minute,
        time_end_hour,
        time_end_minute,
    ):
        """Uses ConfidencePlotter to plot confidence distributions for the selected classes."""
        if not proc_state or not proc_state.processor:
            raise gr.Error(loc.localize("eval-tab-error-calc-metrics-first"))

        df = proc_state.processor.get_data()
        if df.empty:
            raise gr.Error("No predictions to show.")

        # Apply class and recording filters first
        col_class = proc_state.processor.get_column_name("Class")
        conf_col = proc_state.processor.get_column_name("Confidence")
        
        if selected_classes_list:
            df = df[df[col_class].isin(selected_classes_list)]
        if selected_recordings_list:
            selected_recordings_list = [rec.lower() for rec in selected_recordings_list]
            df["recording_filename"] = df["recording_filename"].apply(lambda x: os.path.splitext(os.path.basename(x.strip()))[0].lower() if isinstance(x, str) else x)
            df = df[df["recording_filename"].isin(selected_recordings_list)]

        # Apply date and time filters
        df = apply_datetime_filters(
            df, 
            date_range_start, 
            date_range_end, 
            time_start_hour, 
            time_start_minute, 
            time_end_hour, 
            time_end_minute
        )
        
        if df.empty:
            raise gr.Error("No predictions match the selected date/time filters.")

        # Create histogram plot (using Plotly) with fixed 10 bins (handled in the method)
        plotter = ConfidencePlotter(
            data=df,
            class_col=col_class,
            conf_col=conf_col
        )

        try:
            # Remove unsupported 'nbins' keyword; the method always uses 10 bins
            fig_hist = plotter.plot_histogram_plotly(title="Histogram of Confidence Scores by Class")
            new_state = ProcessorState(
                processor=proc_state.processor,
                prediction_dir=proc_state.prediction_dir,
                metadata_dir=proc_state.metadata_dir,
                color_map=proc_state.color_map
            )
            return [new_state, gr.update(visible=True, value=fig_hist)]
        except Exception as e:
            raise gr.Error(f"Error creating plots: {str(e)}")

    def plot_spatial_distribution(
        proc_state: ProcessorState,
        meta_country,
        meta_x,
        meta_y,
        coordinate_format,
        meta_site,
        confidence_threshold: float,
        date_range_start,
        date_range_end,
        time_start_hour,
        time_start_minute,
        time_end_hour,
        time_end_minute,
    ):
        """Plot spatial distribution of predictions by class."""
        if not proc_state or not proc_state.processor:
            raise gr.Error("Please load predictions first")
            
        try:
            # Read metadata file from the provided directory
            meta_files = list(Path(proc_state.metadata_dir).glob("*.csv"))
            if not meta_files:
                raise gr.Error("No metadata files found")
            metadata_df = pd.read_csv(meta_files[0])
            
            if coordinate_format != "UTM":
                # Ensure expected source columns exist
                if meta_x not in metadata_df.columns or meta_y not in metadata_df.columns:
                    raise gr.Error("Metadata file missing expected coordinate columns.")
                # Convert columns and create standardized 'latitude' and 'longitude'
                metadata_df = metadata_df.copy()
                metadata_df['latitude'] = pd.to_numeric(metadata_df[meta_x], errors='coerce')
                metadata_df['longitude'] = pd.to_numeric(metadata_df[meta_y], errors='coerce')
            else:
                metadata_df = process_coordinates(
                    metadata_df,
                    country_col=meta_country,
                    easting_col=meta_x,
                    northing_col=meta_y
                )
            
            # Set metadata into processor
            proc_state.processor.set_metadata(metadata_df, site_col=meta_site, lat_col='latitude', lon_col='longitude')
            
            # Get prediction data and apply filters
            df = proc_state.processor.get_data()
            
            # Apply confidence threshold filter
            conf_col = proc_state.processor.get_column_name("Confidence")
            df = df[df[conf_col] >= confidence_threshold]
            
            # Apply date and time filters
            df = apply_datetime_filters(
                df, 
                date_range_start, 
                date_range_end, 
                time_start_hour, 
                time_start_minute, 
                time_end_hour, 
                time_end_minute
            )
            
            if df.empty:
                raise gr.Error("No predictions match the selected filters.")
            
            class_col = proc_state.processor.get_column_name("Class")
            # Ensure that latitude and longitude exist in the data after merge
            for col in ['latitude', 'longitude']:
                if col not in df.columns:
                    raise gr.Error(f"Column '{col}' is missing after merging metadata.")
            
            agg_df = df.groupby(['site_name', 'latitude', 'longitude', class_col]).size().reset_index(name='count')
            if agg_df.empty:
                raise gr.Error("No predictions with valid locations found")
            
            sorted_classes = sorted(agg_df[class_col].unique())
            base_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
            colors = base_colors * (1 + len(sorted_classes) // len(base_colors))
            color_map = {cls: colors[i] for i, cls in enumerate(sorted_classes)}
            
            fig = px.scatter_mapbox(
                agg_df,
                lat='latitude',
                lon='longitude',
                size='count',
                color=class_col,
                category_orders={class_col: sorted_classes},
                color_discrete_map=color_map,
                hover_data=['site_name', 'count'],
                size_max=50,
                zoom=10,
                height=600,
                title="Spatial Distribution of Predictions by Class"
            )
            max_count = max(agg_df['count'])
            size_scale = 50
            sizeref = 2.0 * max_count / (size_scale**2)
            fig.update_traces(marker=dict(sizemin=3, sizemode='area', sizeref=sizeref, opacity=0.8))
            fig.update_layout(
                mapbox_style='open-street-map',
                margin={"r":0,"t":30,"l":0,"b":0},
                legend_title="Class",
                showlegend=True,
                mapbox=dict(center=dict(lat=agg_df['latitude'].mean(), lon=agg_df['longitude'].mean()), zoom=10)
            )
            fig.update_traces(
                hovertemplate=(
                    "Site: %{customdata[0]}<br>"
                    "Count: %{customdata[1]}<br>"
                    "Latitude: %{lat:.2f}<br>"
                    "Longitude: %{lon:.2f}<br>"
                    "<extra></extra>"
                )
            )
            return gr.update(value=fig, visible=True)
        except Exception as e:
            raise gr.Error(f"Error creating map: {str(e)}")

    def get_selection_tables(directory):
        """Reads prediction txt files and metadata csv files from directory."""
        from pathlib import Path
        directory = Path(directory)
        files = list(directory.glob("*.txt")) + list(directory.glob("*.csv"))
        return files

    with gr.Tab(loc.localize("visualization-tab-title")):
        gr.Markdown(
            """
            <style>
            .custom-checkbox-group {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                grid-gap: 8px;
            }
            </style>
            """
        )

        # States
        processor_state = gr.State()
        prediction_files_state = gr.State()
        metadata_files_state = gr.State()

        # File Selection UI
        with gr.Row():
            with gr.Column():
                prediction_select_directory_btn = gr.Button(loc.localize("eval-tab-prediction-selection-button-label"))
                prediction_directory_input = gr.Matrix(
                    interactive=False,
                    headers=[loc.localize("eval-tab-selections-column-file-header")],
                )
            with gr.Column():
                metadata_select_directory_btn = gr.Button(loc.localize("viz-tab-metadata-selection-button-label"))
                metadata_directory_input = gr.Matrix(
                    interactive=False,
                    headers=[loc.localize("eval-tab-selections-column-file-header")],
                )

        # Prediction columns box (using gr.Group as in evaluation tab)
        with gr.Group(visible=False) as prediction_group:
            with gr.Accordion(loc.localize("eval-tab-prediction-col-accordion-label"), open=True):
                with gr.Row():
                    prediction_columns: dict[str, gr.Dropdown] = {}
                    for col in ["Start Time", "End Time", "Class", "Confidence", "Recording"]:
                        prediction_columns[col] = gr.Dropdown(choices=[], label=localized_column_labels[col])

        # Metadata columns box
        with gr.Group(visible=False) as metadata_group:
            with gr.Accordion(loc.localize("viz-tab-metadata-col-accordion-label"), open=True):
                with gr.Row():
                    metadata_columns: dict[str, gr.Dropdown] = {}
                    for col in ["Site", "Country", "X", "Y"]:
                        label = localized_column_labels[col]
                        if col == "X":
                            label += " (Longitude/Easting)"
                        elif col == "Y":
                            label += " (Latitude/Northing)"
                        metadata_columns[col] = gr.Dropdown(choices=[], label=label)
                
                with gr.Row():
                    coordinate_format = gr.Radio(
                        choices=["GCS", "UTM"],
                        value="GCS",
                        label=loc.localize("viz-tab-coordinate-format-label"),
                        info=loc.localize("viz-tab-coordinate-format-info")
                    )

        # Class and Recording Selection Box
        with gr.Group(visible=True) as class_recording_group:
            with gr.Accordion(loc.localize("viz-tab-class-recording-accordion-label"), open=False):
                with gr.Row():
                    with gr.Column():
                        select_classes_checkboxgroup = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label=loc.localize("viz-tab-classes-label"),
                            info=loc.localize("viz-tab-classes-info"),
                            interactive=True,
                            elem_classes="custom-checkbox-group",
                        )
                    with gr.Column():
                        select_recordings_checkboxgroup = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label=loc.localize("viz-tab-recordings-label"),
                            info=loc.localize("viz-tab-recordings-info"),
                            interactive=True,
                            elem_classes="custom-checkbox-group",
                        )

        # Parameters Box
        with gr.Group():
            with gr.Accordion(loc.localize("viz-tab-parameters-accordion-label"), open=False):
                with gr.Row():
                    confidence_threshold = gr.Slider(
                        minimum=0.01,
                        maximum=0.99,
                        value=0.10,
                        step=0.01,
                        label=loc.localize("viz-tab-confidence-threshold-label"),
                        info=loc.localize("viz-tab-confidence-threshold-info")
                    )
                    
                with gr.Row():
                    date_range_start = gr.DateTime(
                        label=loc.localize("viz-tab-date-range-start-label"),
                        info=loc.localize("viz-tab-date-range-start-info"),
                        interactive=True,
                        show_label=True,
                        include_time=False
                    )
                    date_range_end = gr.DateTime(
                        label=loc.localize("viz-tab-date-range-end-label"),
                        info=loc.localize("viz-tab-date-range-end-info"),
                        interactive=True,
                        show_label=True,
                        include_time=False
                    )

                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            time_start_hour = gr.Dropdown(
                                choices=[f"{i:02d}" for i in range(24)],
                                value="00",
                                label=loc.localize("viz-tab-start-time-label-hour"),
                                interactive=True
                            )
                            time_start_minute = gr.Dropdown(
                                choices=[f"{i:02d}" for i in range(60)],
                                value="00",
                                label=loc.localize("viz-tab-start-time-label-minute"),
                                interactive=True
                            )
                    
                    with gr.Column():
                        with gr.Row():
                            time_end_hour = gr.Dropdown(
                                choices=[f"{i:02d}" for i in range(24)],
                                value="23",
                                label=loc.localize("viz-tab-end-time-label-hour"),
                                interactive=True
                            )
                            time_end_minute = gr.Dropdown(
                                choices=[f"{i:02d}" for i in range(60)],
                                value="59",
                                label=loc.localize("viz-tab-end-time-label-minute"),
                                interactive=True
                            )

        # Action button and output for smooth distribution plot
        plot_predictions_btn = gr.Button(
            loc.localize("viz-tab-plot-distributions-button-label"), 
            variant="huggingface"
        )
        smooth_distribution_output = gr.Plot(label=loc.localize("viz-tab-distribution-plot-label"), visible=False)

        # Add map button and output after the existing plot components
        plot_map_btn = gr.Button(
            loc.localize("viz-tab-plot-map-button-label"), 
            variant="huggingface"
        )
        map_output = gr.Plot(label=loc.localize("viz-tab-map-plot-label"), visible=False)

        # Interactions
        def get_selection_func(state_key, on_select):
            def select_directory_on_empty():
                folder = gu.select_folder(state_key=state_key)
                if folder:
                    files = get_selection_tables(folder)
                    files_to_display = files[:100] + [["..."]] if len(files) > 100 else files
                    return [files, files_to_display, gr.update(visible=True)] + on_select(files)
                return ["", [[loc.localize("eval-tab-no-files-found")]], gr.update(visible=False)] + [gr.update(visible=False)] * 6
            return select_directory_on_empty

        prediction_select_directory_btn.click(
            get_selection_func("eval-predictions-dir", update_prediction_columns),
            outputs=[prediction_files_state, prediction_directory_input, prediction_group]
            + [prediction_columns[label] for label in ["Start Time", "End Time", "Class", "Confidence", "Recording"]],
            show_progress=True,
        )

        metadata_select_directory_btn.click(
            get_selection_func("viz-metadata-dir", update_metadata_columns),
            outputs=[
                metadata_files_state,
                metadata_directory_input,
                metadata_group,
                metadata_columns["Site"],
                metadata_columns["Country"], 
                metadata_columns["X"],
                metadata_columns["Y"]
            ],
            show_progress=True,
        )

        # Add visibility toggle for metadata group
        metadata_directory_input.change(
            lambda x: gr.update(visible=bool(x)),
            inputs=[metadata_files_state],
            outputs=[metadata_group],
        )

        # Update processor and selections when columns change
        for comp in list(prediction_columns.values()) + list(metadata_columns.values()) + [coordinate_format]:
            comp.change(
                fn=update_selections,
                inputs=[
                    prediction_files_state,
                    metadata_files_state,
                    prediction_columns["Start Time"],
                    prediction_columns["End Time"],
                    prediction_columns["Class"],
                    prediction_columns["Confidence"],
                    prediction_columns["Recording"],
                    metadata_columns["Site"],
                    metadata_columns["Country"],
                    metadata_columns["X"],
                    metadata_columns["Y"],
                    coordinate_format,
                    select_classes_checkboxgroup,
                    select_recordings_checkboxgroup,
                ],
                outputs=[
                    select_classes_checkboxgroup,
                    select_recordings_checkboxgroup,
                    processor_state,
                ],
            ).success(
                fn=update_datetime_defaults,
                inputs=[processor_state],
                outputs=[
                    date_range_start,
                    date_range_end,
                    time_start_hour,
                    time_start_minute,
                    time_end_hour,
                    time_end_minute
                ]
            )

        # Plot button action
        plot_predictions_btn.click(
            fn=plot_predictions_action,
            inputs=[
                processor_state,
                select_classes_checkboxgroup,
                select_recordings_checkboxgroup,
                confidence_threshold,
                date_range_start,
                date_range_end,
                time_start_hour,
                time_start_minute,
                time_end_hour,
                time_end_minute,
            ],
            outputs=[processor_state, smooth_distribution_output]
        )

        # Add click handler for map button
        plot_map_btn.click(
            fn=plot_spatial_distribution,
            inputs=[
                processor_state,
                metadata_columns["Country"],
                metadata_columns["X"],
                metadata_columns["Y"],
                coordinate_format,
                metadata_columns["Site"],
                confidence_threshold,
                date_range_start,
                date_range_end,
                time_start_hour,
                time_start_minute,
                time_end_hour,
                time_end_minute,
            ],
            outputs=[map_output]
        )


if __name__ == "__main__":
    gu.open_window(build_visualization_tab)
