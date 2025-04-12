import json
import os
import shutil
import tempfile
import typing
import io
from pathlib import Path  # Add this import

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px  # Add this import

import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer.visualization.data_processor import DataProcessor
from birdnet_analyzer.visualization.plotting.confidences import ConfidencePlotter
from birdnet_analyzer.visualization.utils.coordinates import process_coordinates


class ProcessorState(typing.NamedTuple):
    """State of the DataProcessor."""
    processor: DataProcessor
    prediction_dir: str
    metadata_dir: str  # Add metadata directory to state
    color_map: typing.Optional[typing.Dict[str, str]] = None  # Add color map to state


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
        "Duration": "File Duration (s)",
        "Confidence": "Confidence",
    }

    # Default columns for metadata
    metadata_default_columns = {
        "Site": "Site",
        "Country": "country", #"Country",
        "X": "N", #"Longitude",  # Changed from separate Longitude/Easting
        "Y": "W", #"Latitude",   # Changed from separate Latitude/Northing
    }

    localized_column_labels = {
        "Start Time": loc.localize("eval-tab-column-start-time-label"),
        "End Time": loc.localize("eval-tab-column-end-time-label"),
        "Class": loc.localize("eval-tab-column-class-label"),
        "Recording": loc.localize("eval-tab-column-recording-label"),
        "Duration": loc.localize("eval-tab-column-duration-label"),
        "Confidence": loc.localize("eval-tab-column-confidence-label"),
        "Site": loc.localize("viz-tab-column-site-label"),
        "Country": loc.localize("viz-tab-column-country-label"),
        "X": loc.localize("viz-tab-column-x-label"),  # Will show as "Longitude/Easting"
        "Y": loc.localize("viz-tab-column-y-label"),  # Will show as "Latitude/Northing"
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
        pred_duration,
        prediction_dir=None,
    ):
        """
        Creates a simplified DataProcessor for predictions only.
        """
        if not prediction_files:
            return [], [], None, None

        # If we haven't saved these predictions yet, save them
        if prediction_dir is None:
            prediction_dir = save_uploaded_files(prediction_files)

        # Fallback for prediction columns
        pred_start_time = pred_start_time if pred_start_time else prediction_default_columns["Start Time"]
        pred_end_time = pred_end_time if pred_end_time else prediction_default_columns["End Time"]
        pred_class = pred_class if pred_class else prediction_default_columns["Class"]
        pred_confidence = pred_confidence if pred_confidence else prediction_default_columns["Confidence"]
        pred_recording = pred_recording if pred_recording else prediction_default_columns["Recording"]
        pred_duration = pred_duration if pred_duration else prediction_default_columns["Duration"]

        cols_pred = {
            "Start Time": pred_start_time,
            "End Time": pred_end_time,
            "Class": pred_class,
            "Confidence": pred_confidence,
            "Recording": pred_recording,
            "Duration": pred_duration,
        }

        try:
            proc = DataProcessor(
                prediction_directory_path=prediction_dir,
                prediction_file_name=None,
                columns_predictions=cols_pred,
            )
            # Get available classes and recordings using the correct column name
            avail_classes = list(proc.classes)
            avail_recordings = proc.get_data()["recording_filename"].unique().tolist()
            return avail_classes, avail_recordings, proc, prediction_dir
        except KeyError as e:
            print(f"Column missing in files: {e}")
            raise gr.Error(
                f"{loc.localize('eval-tab-error-missing-col')}: "
                + str(e)
                + f". {loc.localize('eval-tab-error-missing-col-info')}"
            ) from e
        except Exception as e:
            print(f"Error initializing processor: {e}")
            raise gr.Error(f"{loc.localize('eval-tab-error-init-processor')}:" + str(e)) from e

    def update_prediction_columns(uploaded_files):
        """
        Called when user selects prediction files. Reads headers and updates dropdowns.
        """
        cols = get_columns_from_uploaded_files(uploaded_files)
        cols = [""] + cols
        updates = []
        for label in ["Start Time", "End Time", "Class", "Confidence", "Recording", "Duration"]:
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
        metadata_files,  # New parameter
        pred_start_time,
        pred_end_time,
        pred_class,
        pred_confidence,
        pred_recording,
        pred_duration,
        meta_site,  # New parameter
        meta_country,  # New parameter
        meta_x,  # Changed from separate easting/longitude
        meta_y,  # Changed from separate northing/latitude
        coordinate_format,  # New parameter
        current_classes,
        current_recordings,
    ):
        """
        Called to create or update the processor based on current file uploads and column selections.
        Preserves any selected classes/recordings that remain valid.
        """
        prediction_dir = save_uploaded_files(prediction_files)
        metadata_dir = save_uploaded_files(metadata_files)  # Save metadata files
        
        avail_classes, avail_recordings, proc, prediction_dir = initialize_processor(
            prediction_files,
            pred_start_time,
            pred_end_time,
            pred_class,
            pred_confidence,
            pred_recording,
            pred_duration,
            prediction_dir,
        )
        
        state = ProcessorState(proc, prediction_dir, metadata_dir) if proc else None

        # Preserve valid selections or default to all available
        new_classes = (
            avail_classes
            if not current_classes
            else [c for c in current_classes if c in avail_classes] or avail_classes
        )
        new_recordings = (
            avail_recordings
            if not current_recordings
            else [r for r in current_recordings if r in avail_recordings] or avail_recordings
        )

        return (
            gr.update(choices=avail_classes, value=new_classes),
            gr.update(choices=avail_recordings, value=new_recordings),
            state,
        )

    def plot_predictions_action(
        proc_state: ProcessorState,
        selected_classes_list,
        selected_recordings_list,
    ):
        """Uses ConfidencePlotter to plot confidence distributions for the selected classes."""
        if not proc_state or not proc_state.processor:
            raise gr.Error(loc.localize("eval-tab-error-calc-metrics-first"), print_exception=False)

        df = proc_state.processor.get_data()
        if df.empty:
            raise gr.Error("No predictions to show.")

        # Filter data first
        col_class = proc_state.processor.get_column_name("Class")
        conf_col = proc_state.processor.get_column_name("Confidence")
        
        if selected_classes_list:
            df = df[df[col_class].isin(selected_classes_list)]
        if selected_recordings_list:
            df = df[df["recording_filename"].isin(selected_recordings_list)]
            
        if df.empty:
            raise gr.Error("No predictions left after filtering.")

        plotter = ConfidencePlotter(
            data=df,
            class_col=col_class,
            conf_col=conf_col
        )

        try:
            # Get all classes in sorted order for consistent colors
            all_classes = sorted(proc_state.processor.get_data()[col_class].unique())
            
            # Define base colors for both plots
            base_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
            # Extend colors if needed by cycling through them
            colors = base_colors * (1 + len(all_classes) // len(base_colors))
            color_map = {cls: colors[i] for i, cls in enumerate(all_classes)}
            
            # Create the density plot using the color map
            fig_smooth = plotter.plot_smooth_distribution_plotly(
                bandwidth=0.2,
                title="Smooth Distribution of Confidence Scores",
                classes=selected_classes_list if selected_classes_list else all_classes,
                color_map=color_map
            )
            
            # Store color map in state
            new_state = ProcessorState(
                processor=proc_state.processor,
                prediction_dir=proc_state.prediction_dir,
                metadata_dir=proc_state.metadata_dir,
                color_map=color_map
            )
            return [new_state, gr.update(visible=True, value=fig_smooth)]
        except Exception as e:
            raise gr.Error(f"Error creating plots: {str(e)}")

    def plot_spatial_distribution(
        proc_state: ProcessorState,
        meta_country,
        meta_x,
        meta_y,
        coordinate_format,
        meta_site
    ):
        """Plot spatial distribution of predictions by class."""
        if not proc_state or not proc_state.processor:
            raise gr.Error("Please load predictions first")
            
        try:
            # Read metadata
            meta_files = list(Path(proc_state.metadata_dir).glob("*.csv"))
            if not meta_files:
                raise gr.Error("No metadata files found")
                
            metadata_df = pd.read_csv(meta_files[0])
            print(f"Loaded metadata with columns: {metadata_df.columns.tolist()}")
            
            # Convert coordinates if needed
            if coordinate_format == "UTM":
                print("Converting UTM coordinates...")
                metadata_df = process_coordinates(
                    metadata_df,
                    country_col=meta_country,
                    easting_col=meta_x,
                    northing_col=meta_y
                )
                print(f"Converted coordinates: {len(metadata_df)} rows processed")
            else:
                print("Using Lat/Long coordinates directly")
                metadata_df['latitude'] = pd.to_numeric(metadata_df[meta_y], errors='coerce')
                metadata_df['longitude'] = pd.to_numeric(metadata_df[meta_x], errors='coerce')
            
            # Set metadata in processor
            try:
                proc_state.processor.set_metadata(
                    metadata_df,
                    site_col=meta_site,
                    lat_col='latitude',
                    lon_col='longitude'
                )
            except Exception as e:
                print(f"Error setting metadata: {e}")
                raise gr.Error(f"Error setting metadata: {str(e)}")

            try:
                # Get aggregated prediction data
                agg_df = proc_state.processor.get_aggregated_locations()
                
                if agg_df.empty:
                    raise gr.Error("No predictions with valid locations found")

                class_col = proc_state.processor.get_column_name("Class")
                
                # Ensure we have a color map
                if not proc_state.color_map:
                    # Create the same color mapping as in plot_predictions_action
                    all_classes = sorted(proc_state.processor.get_data()[class_col].unique())
                    base_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
                    colors = base_colors * (1 + len(all_classes) // len(base_colors))
                    color_map = {cls: colors[i] for i, cls in enumerate(all_classes)}
                else:
                    color_map = proc_state.color_map

                # Sort classes for consistent order
                sorted_classes = sorted(agg_df[class_col].unique())

                # Create map using Plotly
                fig = px.scatter_mapbox(
                    agg_df,
                    lat='latitude',
                    lon='longitude',
                    size='count',
                    color=class_col,
                    category_orders={class_col: sorted_classes},
                    color_discrete_map=color_map,
                    hover_data=['site_name', 'count'],
                    size_max=50,  # Increased from 50 to 100
                    zoom=10,
                    height=600,
                    title="Spatial Distribution of Predictions by Class"
                )

                # Update traces for more distinctive size differences
                max_count = max(agg_df['count'])
                min_count = min(agg_df['count'])
                size_scale = 50  # Maximum marker size
                
                # Calculate size reference for better scaling
                sizeref = 2.0 * max_count / (size_scale**2)
                
                fig.update_traces(
                    marker=dict(
                        sizemin=3,  # Minimum marker size
                        sizemode='area',
                        sizeref=sizeref,  # Adjusted size reference
                        opacity=0.8
                    )
                )
                
                # Update layout
                fig.update_layout(
                    mapbox_style='open-street-map',
                    margin={"r":0,"t":30,"l":0,"b":0},
                    legend_title="Species",
                    showlegend=True,
                    mapbox=dict(
                        center=dict(
                            lat=agg_df['latitude'].mean(),
                            lon=agg_df['longitude'].mean()
                        ),
                        zoom=10  # Set a default zoom level that shows more context
                    )
                )
                
                return gr.update(value=fig, visible=True)
                
            except ValueError as ve:
                print(f"Error in data processing: {ve}")
                raise gr.Error(f"Error processing data: {str(ve)}")
                
        except Exception as e:
            print(f"Error in plot_spatial_distribution: {str(e)}")
            raise gr.Error(f"Error creating map: {str(e)}")

    def get_selection_tables(directory):
        """Reads prediction txt files and metadata csv files from directory."""
        from pathlib import Path
        directory = Path(directory)
        # Add support for both txt and csv files
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
                    for col in ["Start Time", "End Time", "Class", "Confidence", "Recording", "Duration"]:
                        prediction_columns[col] = gr.Dropdown(choices=[], label=localized_column_labels[col])

        # Metadata columns box
        with gr.Group(visible=False) as metadata_group:
            with gr.Accordion(loc.localize("viz-tab-metadata-col-accordion-label"), open=True):
                with gr.Row():
                    metadata_columns: dict[str, gr.Dropdown] = {}
                    # Updated column list with merged coordinate columns
                    for col in ["Site", "Country", "X", "Y"]:
                        label = localized_column_labels[col]
                        if col == "X":
                            label += " (Longitude/Easting)"
                        elif col == "Y":
                            label += " (Latitude/Northing)"
                        metadata_columns[col] = gr.Dropdown(choices=[], label=label)
                
                with gr.Row():
                    coordinate_format = gr.Radio(
                        choices=["Lat/Long", "UTM"],
                        value="Lat/Long",
                        label=loc.localize("viz-tab-coordinate-format-label"),
                        info=loc.localize("viz-tab-coordinate-format-info")
                    )

        # Class and Recording Selection Box (changed back to gr.Group as in evaluation tab)
        with gr.Group(visible=True) as class_recording_group:
            with gr.Accordion("Select Classes and Recordings", open=False):
                with gr.Row():
                    with gr.Column():
                        select_classes_checkboxgroup = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label="Classes",
                            info="Select classes to include in visualization",
                            interactive=True,
                            elem_classes="custom-checkbox-group",
                        )
                    with gr.Column():
                        select_recordings_checkboxgroup = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label="Recordings",
                            info="Select recordings to include in visualization",
                            interactive=True,
                            elem_classes="custom-checkbox-group",
                        )

        # Action button and output for smooth distribution plot
        plot_predictions_btn = gr.Button("Plot Confidence Distributions")
        smooth_distribution_output = gr.Plot(label="Smooth Confidence Distribution", visible=False)

        # Add map button and output after the existing plot components
        plot_map_btn = gr.Button("Plot Spatial Distribution")
        map_output = gr.Plot(label="Recording Locations", visible=False)

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
            + [prediction_columns[label] for label in ["Start Time", "End Time", "Class", "Confidence", "Recording", "Duration"]],
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
                    prediction_columns["Duration"],
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
            )

        # Plot button action
        plot_predictions_btn.click(
            fn=plot_predictions_action,
            inputs=[processor_state, select_classes_checkboxgroup, select_recordings_checkboxgroup],
            outputs=[processor_state, smooth_distribution_output],  # Add processor_state to outputs
        )

        # Add click handler for map button
        plot_map_btn.click(
            fn=plot_spatial_distribution,
            inputs=[
                processor_state,
                metadata_columns["Country"],
                metadata_columns["X"],  # Changed from separate longitude/easting
                metadata_columns["Y"],  # Changed from separate latitude/northing
                coordinate_format,
                metadata_columns["Site"]
            ],
            outputs=[map_output]
        )


if __name__ == "__main__":
    gu.open_window(build_visualization_tab)
