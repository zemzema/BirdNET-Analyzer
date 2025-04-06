import json
import os
import shutil
import tempfile
import typing
import io

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer.visualization.data_processor import DataProcessor
from birdnet_analyzer.visualization.plotting.confidences import ConfidencePlotter


class ProcessorState(typing.NamedTuple):
    """State of the DataProcessor."""
    processor: DataProcessor
    prediction_dir: str


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

    localized_column_labels = {
        "Start Time": loc.localize("eval-tab-column-start-time-label"),
        "End Time": loc.localize("eval-tab-column-end-time-label"),
        "Class": loc.localize("eval-tab-column-class-label"),
        "Recording": loc.localize("eval-tab-column-recording-label"),
        "Duration": loc.localize("eval-tab-column-duration-label"),
        "Confidence": loc.localize("eval-tab-column-confidence-label"),
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

    def update_selections(
        prediction_files,
        pred_start_time,
        pred_end_time,
        pred_class,
        pred_confidence,
        pred_recording,
        pred_duration,
        current_classes,
        current_recordings,
    ):
        """
        Called to create or update the processor based on current file uploads and column selections.
        Preserves any selected classes/recordings that remain valid.
        """
        prediction_dir = save_uploaded_files(prediction_files)
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
        state = ProcessorState(proc, prediction_dir) if proc else None

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
        """
        Uses ConfidencePlotter to plot confidence distributions for the selected classes.
        """
        if not proc_state or not proc_state.processor:
            raise gr.Error(loc.localize("eval-tab-error-calc-metrics-first"), print_exception=False)

        df = proc_state.processor.get_data()
        if df.empty:
            raise gr.Error("No predictions to show.")

        # Use the mapped column names
        col_class = proc_state.processor.get_column_name("Class")
        conf_col = proc_state.processor.get_column_name("Confidence")
        
        # Filter by classes and recordings if selected
        if selected_classes_list:
            df = df[df[col_class].isin(selected_classes_list)]
        if selected_recordings_list:
            df = df[df["recording_filename"].isin(selected_recordings_list)]
            
        if df.empty:
            raise gr.Error("No predictions left after filtering.")

        # Create ConfidencePlotter
        plotter = ConfidencePlotter(
            data=df,
            class_col=col_class,
            conf_col=conf_col
        )

        try:
            fig_smooth = plotter.plot_smooth_distribution_plotly(
                bandwidth=0.2,
                title="Smooth Distribution of Confidence Scores",
                classes=selected_classes_list
            )
            return gr.update(visible=True, value=fig_smooth)
        except Exception as e:
            raise gr.Error(f"Error creating plots: {str(e)}")

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

        # File Selection UI
        with gr.Row():
            with gr.Column():
                prediction_select_directory_btn = gr.Button(loc.localize("eval-tab-prediction-selection-button-label"))
                prediction_directory_input = gr.Matrix(
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

        # Interactions
        def get_selection_tables(directory):
            from pathlib import Path
            directory = Path(directory)
            return list(directory.glob("*.txt"))

        # Folder selection
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

        # Update processor and selections when columns change
        for comp in prediction_columns.values():
            comp.change(
                fn=update_selections,
                inputs=[
                    prediction_files_state,
                    prediction_columns["Start Time"],
                    prediction_columns["End Time"],
                    prediction_columns["Class"],
                    prediction_columns["Confidence"],
                    prediction_columns["Recording"],
                    prediction_columns["Duration"],
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
            outputs=[smooth_distribution_output],
        )


if __name__ == "__main__":
    gu.open_window(build_visualization_tab)
