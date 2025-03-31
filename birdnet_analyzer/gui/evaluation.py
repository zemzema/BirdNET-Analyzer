import json
import os
import shutil
import tempfile
import typing

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

import birdnet_analyzer.gui.localization as loc
import birdnet_analyzer.gui.utils as gu
from birdnet_analyzer.evaluation.assessment.performance_assessor import PerformanceAssessor
from birdnet_analyzer.evaluation.core import process_data
from birdnet_analyzer.evaluation.preprocessing.data_processor import DataProcessor


class ProcessorState(typing.NamedTuple):
    processor: DataProcessor
    annotation_dir: str
    prediction_dir: str


def build_evaluation_tab():
    # Default columns for annotations
    annotation_default_columns = {
        "Start Time": "Begin Time (s)",
        "End Time": "End Time (s)",
        "Class": "Class",
        "Recording": "Begin File",
        "Duration": "File Duration (s)",
    }

    # Default columns for predictions
    prediction_default_columns = {
        "Start Time": "Begin Time (s)",
        "End Time": "End Time (s)",
        "Class": "Common Name",
        "Recording": "Begin File",
        "Duration": "File Duration (s)",
        "Confidence": "Confidence",
    }

    def download_class_mapping_template():
        template_mapping = {
            "Predicted Class Name 1": "Annotation Class Name 1",
            "Predicted Class Name 2": "Annotation Class Name 2",
            "Predicted Class Name 3": "Annotation Class Name 3",
            "Predicted Class Name 4": "Annotation Class Name 4",
            "Predicted Class Name 5": "Annotation Class Name 5",
        }
        fd, temp_path = tempfile.mkstemp(suffix=".json")

        with os.fdopen(fd, "w") as f:
            json.dump(template_mapping, f, indent=4)

        desired_path = os.path.join(os.path.dirname(temp_path), "class_mapping_template.json")

        if os.path.exists(desired_path):
            os.remove(desired_path)

        os.rename(temp_path, desired_path)

        return gr.update(value=desired_path, visible=True)

    def download_results_table(pa: PerformanceAssessor, predictions, labels, class_wise_value):
        if pa is None or predictions is None or labels is None:
            return None
        try:
            metrics_df = pa.calculate_metrics(predictions, labels, per_class_metrics=class_wise_value)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            metrics_df.to_csv(temp_file.name, index=True)
            temp_file.close()
            desired_path = os.path.join(os.path.dirname(temp_file.name), "results_table.csv")

            if os.path.exists(desired_path):
                os.remove(desired_path)

            os.rename(temp_file.name, desired_path)

            return gr.update(value=desired_path)
        except Exception as e:
            print(f"Error saving results table: {e}")
            raise gr.Error("Error saving results table") from e

    def download_data_table(processor_state: ProcessorState):
        if processor_state is None:
            return None
        try:
            data_df = processor_state.processor.get_sample_data()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            data_df.to_csv(temp_file.name, index=False)
            temp_file.close()
            desired_path = os.path.join(os.path.dirname(temp_file.name), "data_table.csv")

            if os.path.exists(desired_path):
                os.remove(desired_path)

            os.rename(temp_file.name, desired_path)

            return gr.update(value=desired_path)
        except Exception as e:
            raise gr.Error("Error saving data table") from e

    def get_columns_from_uploaded_files(files):
        columns = set()

        if files:
            for file_obj in files:
                try:
                    df = pd.read_csv(file_obj, sep=None, engine="python", nrows=0)
                    columns.update(df.columns)
                except Exception as e:
                    print(f"Error reading file {file_obj}: {e}")
                    raise gr.Error(f"Error reading file {file_obj}") from e

        return sorted(list(columns))

    def save_uploaded_files(files):
        if not files:
            return None

        temp_dir = tempfile.mkdtemp()

        for file_obj in files:
            dest_path = os.path.join(temp_dir, os.path.basename(file_obj))
            shutil.copy(file_obj, dest_path)

        return temp_dir

    # Single initialize_processor that can reuse given directories.
    def initialize_processor(
        annotation_files,
        prediction_files,
        mapping_file_obj,
        sample_duration_value,
        min_overlap_value,
        recording_duration,
        ann_start_time,
        ann_end_time,
        ann_class,
        ann_recording,
        ann_duration,
        pred_start_time,
        pred_end_time,
        pred_class,
        pred_confidence,
        pred_recording,
        pred_duration,
        annotation_dir=None,
        prediction_dir=None,
    ):
        if not annotation_files or not prediction_files:
            return [], [], None, None, None

        if annotation_dir is None:
            annotation_dir = save_uploaded_files(annotation_files)

        if prediction_dir is None:
            prediction_dir = save_uploaded_files(prediction_files)

        # Fallback for annotation columns.
        ann_start_time = ann_start_time if ann_start_time else annotation_default_columns["Start Time"]
        ann_end_time = ann_end_time if ann_end_time else annotation_default_columns["End Time"]
        ann_class = ann_class if ann_class else annotation_default_columns["Class"]
        ann_recording = ann_recording if ann_recording else annotation_default_columns["Recording"]
        ann_duration = ann_duration if ann_duration else annotation_default_columns["Duration"]

        # Fallback for prediction columns.
        pred_start_time = pred_start_time if pred_start_time else prediction_default_columns["Start Time"]
        pred_end_time = pred_end_time if pred_end_time else prediction_default_columns["End Time"]
        pred_class = pred_class if pred_class else prediction_default_columns["Class"]
        pred_confidence = pred_confidence if pred_confidence else prediction_default_columns["Confidence"]
        pred_recording = pred_recording if pred_recording else prediction_default_columns["Recording"]
        pred_duration = pred_duration if pred_duration else prediction_default_columns["Duration"]

        cols_ann = {
            "Start Time": ann_start_time,
            "End Time": ann_end_time,
            "Class": ann_class,
            "Recording": ann_recording,
            "Duration": ann_duration,
        }
        cols_pred = {
            "Start Time": pred_start_time,
            "End Time": pred_end_time,
            "Class": pred_class,
            "Confidence": pred_confidence,
            "Recording": pred_recording,
            "Duration": pred_duration,
        }

        # Handle mapping file: if it has a temp_files attribute use that, otherwise assume it's a filepath.
        if mapping_file_obj and hasattr(mapping_file_obj, "temp_files"):
            mapping_path = list(mapping_file_obj.temp_files)[0]
        else:
            mapping_path = mapping_file_obj if mapping_file_obj else None

        if mapping_path:
            with open(mapping_path, "r") as f:
                class_mapping = json.load(f)
        else:
            class_mapping = None

        try:
            proc = DataProcessor(
                prediction_directory_path=prediction_dir,
                prediction_file_name=None,
                annotation_directory_path=annotation_dir,
                annotation_file_name=None,
                class_mapping=class_mapping,
                sample_duration=sample_duration_value,
                min_overlap=min_overlap_value,
                columns_predictions=cols_pred,
                columns_annotations=cols_ann,
                recording_duration=recording_duration,
            )
            avail_classes = list(proc.classes)  # Ensure it's a list
            avail_recordings = proc.samples_df["filename"].unique().tolist()

            return avail_classes, avail_recordings, proc, annotation_dir, prediction_dir
        except KeyError as e:
            print(f"Column missing in files: {e}")
            raise gr.Error("Column missing in files: " + str(e) + ". Please check the column names.") from e
        except Exception as e:
            print(f"Error initializing processor: {e}")

            raise gr.Error(loc.localize("eval-tab-selections-dataframe-error-message") + str(e)) from e

    # update_selections is triggered when files or mapping file change.
    # It creates the temporary directories once and stores them along with the processor.
    # It now also receives the current selection values so that user selections are preserved.
    def update_selections(
        annotation_files,
        prediction_files,
        mapping_file_obj,
        sample_duration_value,
        min_overlap_value,
        recording_duration_value: str,
        ann_start_time,
        ann_end_time,
        ann_class,
        ann_recording,
        ann_duration,
        pred_start_time,
        pred_end_time,
        pred_class,
        pred_confidence,
        pred_recording,
        pred_duration,
        current_classes,
        current_recordings,
    ):
        if recording_duration_value.strip() == "":
            rec_dur = None
        else:
            try:
                rec_dur = float(recording_duration_value)
            except ValueError:
                rec_dur = None

        # Create temporary directories once.
        annotation_dir = save_uploaded_files(annotation_files)
        prediction_dir = save_uploaded_files(prediction_files)
        avail_classes, avail_recordings, proc, annotation_dir, prediction_dir = initialize_processor(
            annotation_files,
            prediction_files,
            mapping_file_obj,
            sample_duration_value,
            min_overlap_value,
            rec_dur,
            ann_start_time,
            ann_end_time,
            ann_class,
            ann_recording,
            ann_duration,
            pred_start_time,
            pred_end_time,
            pred_class,
            pred_confidence,
            pred_recording,
            pred_duration,
            annotation_dir,
            prediction_dir,
        )
        # Build a state dictionary to store the processor and the directories.
        state = ProcessorState(proc, annotation_dir, prediction_dir)
        # If no current selection exists, default to all available classes/recordings;
        # otherwise, preserve any selections that are still valid.
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

    with gr.Tab("Evaluation"):
        # Custom CSS to match the layout style of other files and remove gray backgrounds.
        gr.Markdown(
            """
            <style>
            //body { background-color: #fff; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
            // .gradio-container {
            //     border: 1px solid #ccc;
            //     border-radius: 8px;
            //     padding: 16px;
            //     background-color: transparent;
            // }
            /* Override any group styles */
            //.gradio-group {
            //    background-color: transparent !important;
            //    border: none !important;
            //    box-shadow: none !important;
            //}
            /* Grid layout for checkbox groups */
            .custom-checkbox-group {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                grid-gap: 8px;
            }
            </style>
            """
        )

        processor_state = gr.State()
        pa_state = gr.State()
        predictions_state = gr.State()
        labels_state = gr.State()
        annotation_files_state = gr.State()
        prediction_files_state = gr.State()

        def get_selection_tables(directory):
            from pathlib import Path

            directory = Path(directory)

            return list(directory.glob("*.txt"))

        # Update column dropdowns when files are uploaded.
        def update_annotation_columns(uploaded_files):
            cols = get_columns_from_uploaded_files(uploaded_files)
            cols = [""] + cols
            updates = []

            for label in ["Start Time", "End Time", "Class", "Recording", "Duration"]:
                default_val = annotation_default_columns.get(label)
                val = default_val if default_val in cols else None
                updates.append(gr.update(choices=cols, value=val))

            return updates

        def update_prediction_columns(uploaded_files):
            cols = get_columns_from_uploaded_files(uploaded_files)
            cols = [""] + cols
            updates = []

            for label in ["Start Time", "End Time", "Class", "Confidence", "Recording", "Duration"]:
                default_val = prediction_default_columns.get(label)
                val = default_val if default_val in cols else None
                updates.append(gr.update(choices=cols, value=val))

            return updates

        def get_selection_func(state_key, on_select):
            def select_directory_on_empty():  # Nishant - Function modified for For Folder selection
                folder = gu.select_folder(state_key=state_key)

                if folder:
                    files = get_selection_tables(folder)
                    files_to_display = files[:100] + [["..."]] if len(files) > 100 else files
                    return [files, files_to_display, gr.update(visible=True)] + on_select(files)

                return ["", [[loc.localize("eval-tab-selections-dataframe-no-files-found")]]]

            return select_directory_on_empty

        with gr.Row():
            with gr.Column():
                annotation_select_directory_btn = gr.Button(loc.localize("eval-tab-annotation-selection-button-label"))
                annotation_directory_input = gr.Matrix(
                    interactive=False,
                    headers=[
                        loc.localize("eval-tab-selections-dataframe-column-subpath-header"),
                    ],
                )

            with gr.Column():
                prediction_select_directory_btn = gr.Button(loc.localize("eval-tab-prediction-selection-button-label"))
                prediction_directory_input = gr.Matrix(
                    interactive=False,
                    headers=[
                        loc.localize("eval-tab-selections-dataframe-column-subpath-header"),
                    ],
                )

        # ----------------------- Annotations Columns Box -----------------------
        with gr.Group(visible=False) as annotation_group:
            with gr.Accordion("Annotations Columns", open=True):
                with gr.Row():
                    annotation_columns: dict[str, gr.Dropdown] = {}

                    for label_text in ["Start Time", "End Time", "Class", "Recording", "Duration"]:
                        annotation_columns[label_text] = gr.Dropdown(choices=[], label=label_text)

        # ----------------------- Predictions Columns Box -----------------------
        with gr.Group(visible=False) as prediction_group:
            with gr.Accordion("Predictions Columns", open=True):
                with gr.Row():
                    prediction_columns: dict[str, gr.Dropdown] = {}

                    for label_text in ["Start Time", "End Time", "Class", "Confidence", "Recording", "Duration"]:
                        prediction_columns[label_text] = gr.Dropdown(choices=[], label=label_text)

        # ----------------------- Class Mapping Box -----------------------
        with gr.Group(visible=False) as mapping_group:
            with gr.Accordion("Class Mapping (Optional)", open=False):
                with gr.Row():
                    mapping_file = gr.File(label="Upload Mapping File", file_count="single", file_types=[".json"])
                    download_mapping_button = gr.DownloadButton(label="Download Template")

            download_mapping_button.click(
                fn=download_class_mapping_template, inputs=[], outputs=download_mapping_button
            )

        # ----------------------- Classes and Recordings Selection Box -----------------------
        with gr.Group(visible=False) as class_recording_group:
            with gr.Accordion("Select Classes and Recordings", open=False):
                with gr.Row():
                    with gr.Column():
                        select_classes_checkboxgroup = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label="Select Classes",
                            info="Select the classes to calculate the metrics.",
                            interactive=True,
                            elem_classes="custom-checkbox-group",
                        )

                    with gr.Column():
                        select_recordings_checkboxgroup = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label="Select Recordings",
                            info="Select the recordings to calculate the metrics.",
                            interactive=True,
                            elem_classes="custom-checkbox-group",
                        )

        # ----------------------- Parameters Box -----------------------
        with gr.Group():
            with gr.Accordion("Parameters", open=False):
                with gr.Row():
                    sample_duration = gr.Number(
                        value=3, label="Sample Duration (s)", precision=0, info="Audio sample length (in seconds)."
                    )
                    recording_duration = gr.Textbox(
                        label="Recording Duration (s)",
                        placeholder="Determined from files",
                        info="Inferred from the data if not specified.",
                    )
                    min_overlap = gr.Number(
                        value=0.5,
                        label="Minimum Overlap (s)",
                        info="Overlap needed to assign an annotation to a sample.",
                    )
                    threshold = gr.Slider(
                        minimum=0.01,
                        maximum=0.99,
                        value=0.1,
                        label="Threshold",
                        info="Threshold for classifying a prediction as positive.",
                    )
                    class_wise = gr.Checkbox(
                        label="Class-wise Metrics", value=False, info="Calculate metrics separately for each class."
                    )

        # ----------------------- Metrics Box -----------------------
        with gr.Group():
            with gr.Accordion("Select Metrics", open=False):
                with gr.Row():
                    metric_info = {
                        "AUROC": "AUROC measures the likelihood that the model ranks a random positive case higher than a random negative case.",
                        "Precision": "Precision measures how often the model's positive predictions are actually correct.",
                        "Recall": "Recall measures the percentage of actual positive cases correctly identified by the model for each class.",
                        "F1 Score": "The F1 score is the harmonic mean of precision and recall, balancing both metrics.",
                        "Average Precision (AP)": "Average Precision summarizes the precision-recall curve by averaging precision across all recall levels.",
                        "Accuracy": "Accuracy measures the percentage of correct predictions made by the model.",
                    }
                    metrics_checkboxes = {}

                    for metric_name, description in metric_info.items():
                        metrics_checkboxes[metric_name.lower()] = gr.Checkbox(
                            label=metric_name, value=True, info=description
                        )

        # ----------------------- Actions Box -----------------------
        with gr.Row():
            calculate_button = gr.Button("Calculate Metrics", variant="huggingface")
            plot_metrics_button = gr.Button("Plot Metrics", variant="huggingface")
            plot_confusion_button = gr.Button("Plot Confusion Matrix", variant="huggingface")
            plot_metrics_all_thresholds_button = gr.Button("Plot Metrics All Thresholds", variant="huggingface")

        with gr.Row():
            download_results_button = gr.DownloadButton(
                label="Download Results Table", visible=True, variant="huggingface"
            )
            download_data_button = gr.DownloadButton(label="Download Data Table", visible=True, variant="huggingface")

        download_results_button.click(
            fn=download_results_table,
            inputs=[pa_state, predictions_state, labels_state, class_wise],
            outputs=download_results_button,
        )
        download_data_button.click(fn=download_data_table, inputs=[processor_state], outputs=download_data_button)
        # results_text = gr.Textbox(label="Results", lines=10, visible=False)
        metric_table = gr.Dataframe(
            show_label=False,
            type="pandas",
            visible=False,
        )
        plot_output = gr.Plot(visible=False, show_label=False)

        # Update available selections (classes and recordings) and the processor state when files or mapping file change.
        # Also pass the current selection values so that user selections are preserved.
        for comp in list(annotation_columns.values()) + list(prediction_columns.values()) + [mapping_file]:
            comp.change(
                fn=update_selections,
                inputs=[
                    annotation_files_state,
                    prediction_files_state,
                    mapping_file,
                    sample_duration,
                    min_overlap,
                    recording_duration,
                    annotation_columns["Start Time"],
                    annotation_columns["End Time"],
                    annotation_columns["Class"],
                    annotation_columns["Recording"],
                    annotation_columns["Duration"],
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

        # calculate_metrics now uses the stored temporary directories from processor_state.
        # The function now accepts selected_classes and selected_recordings as inputs.
        def calculate_metrics(
            mapping_file_obj,
            sample_duration_value,
            min_overlap_value,
            recording_duration_value: str,
            ann_start_time,
            ann_end_time,
            ann_class,
            ann_recording,
            ann_duration,
            pred_start_time,
            pred_end_time,
            pred_class,
            pred_confidence,
            pred_recording,
            pred_duration,
            threshold_value,
            class_wise_value,
            selected_classes_list,
            selected_recordings_list,
            proc_state: ProcessorState,
            *metrics_checkbox_values,
        ):
            selected_metrics = []

            for value, (m_lower, _) in zip(metrics_checkbox_values, metrics_checkboxes.items()):
                if value:
                    selected_metrics.append(m_lower)

            valid_metrics = {
                "accuracy": "accuracy",
                "recall": "recall",
                "precision": "precision",
                "f1 score": "f1",
                "average precision (ap)": "ap",
                "auroc": "auroc",
            }
            metrics = tuple([valid_metrics[m] for m in selected_metrics if m in valid_metrics])

            # Fall back to available classes from processor state if none selected.
            if not selected_classes_list and proc_state and proc_state.processor:
                selected_classes_list = list(proc_state.processor.classes)

            if not selected_classes_list:
                raise gr.Error("Error: At least one class must be selected.")

            if recording_duration_value.strip() == "":
                rec_dur = None
            else:
                try:
                    rec_dur = float(recording_duration_value)
                except ValueError as e:
                    raise gr.Error("Error: Please enter a valid number for Recording Duration.") from e

            if mapping_file_obj and hasattr(mapping_file_obj, "temp_files"):
                mapping_path = list(mapping_file_obj.temp_files)[0]
            else:
                mapping_path = mapping_file_obj if mapping_file_obj else None

            try:
                metrics_df, pa, preds, labs = process_data(
                    annotation_path=proc_state.annotation_dir,
                    prediction_path=proc_state.prediction_dir,
                    mapping_path=mapping_path,
                    sample_duration=sample_duration_value,
                    min_overlap=min_overlap_value,
                    recording_duration=rec_dur,
                    columns_annotations={
                        "Start Time": ann_start_time,
                        "End Time": ann_end_time,
                        "Class": ann_class,
                        "Recording": ann_recording,
                        "Duration": ann_duration,
                    },
                    columns_predictions={
                        "Start Time": pred_start_time,
                        "End Time": pred_end_time,
                        "Class": pred_class,
                        "Confidence": pred_confidence,
                        "Recording": pred_recording,
                        "Duration": pred_duration,
                    },
                    selected_classes=selected_classes_list,
                    selected_recordings=selected_recordings_list,
                    metrics_list=metrics,
                    threshold=threshold_value,
                    class_wise=class_wise_value,
                )

                table = metrics_df.T.reset_index(names=[""])

                return (
                    gr.update(value=table, visible=True),
                    pa,
                    preds,
                    labs,
                    gr.update(),
                    gr.update(),
                    proc_state,
                )
            except Exception as e:
                print("Error processing data:", e)
                raise gr.Error("Error processing data.") from e

        # Updated calculate_button click now passes the selected classes and recordings.
        calculate_button.click(
            calculate_metrics,
            inputs=[
                mapping_file,
                sample_duration,
                min_overlap,
                recording_duration,
                annotation_columns["Start Time"],
                annotation_columns["End Time"],
                annotation_columns["Class"],
                annotation_columns["Recording"],
                annotation_columns["Duration"],
                prediction_columns["Start Time"],
                prediction_columns["End Time"],
                prediction_columns["Class"],
                prediction_columns["Confidence"],
                prediction_columns["Recording"],
                prediction_columns["Duration"],
                threshold,
                class_wise,
                select_classes_checkboxgroup,
                select_recordings_checkboxgroup,
                processor_state,
            ]
            + [checkbox for checkbox in metrics_checkboxes.values()],
            outputs=[
                metric_table,
                pa_state,
                predictions_state,
                labels_state,
                select_classes_checkboxgroup,
                select_recordings_checkboxgroup,
                processor_state,
            ],
        )

        def plot_metrics(pa: PerformanceAssessor, predictions, labels, class_wise_value):
            if pa is None or predictions is None or labels is None:
                raise gr.Error("Please calculate metrics first.", print_exception=False)
            try:
                fig = pa.plot_metrics(predictions, labels, per_class_metrics=class_wise_value)
                plt.close(fig)

                return fig, gr.update(visible=True)
            except Exception as e:
                raise gr.Error(f"Error plotting metrics: {e}") from e

        plot_metrics_button.click(
            plot_metrics,
            inputs=[pa_state, predictions_state, labels_state, class_wise],
            outputs=[plot_output, plot_output],
        )

        def plot_confusion_matrix(pa: PerformanceAssessor, predictions, labels):
            if pa is None or predictions is None or labels is None:
                raise gr.Error("Please calculate metrics first.", print_exception=False)
            try:
                fig = pa.plot_confusion_matrix(predictions, labels)
                plt.close(fig)

                return fig, gr.update(visible=True)
            except Exception as e:
                raise gr.Error(f"Error plotting confusion matrix: {e}") from e

        plot_confusion_button.click(
            plot_confusion_matrix,
            inputs=[pa_state, predictions_state, labels_state],
            outputs=[plot_output, plot_output],
        )

        annotation_select_directory_btn.click(
            get_selection_func("eval-annotations-dir", update_annotation_columns),
            outputs=[annotation_files_state, annotation_directory_input, annotation_group]
            + [annotation_columns[label] for label in ["Start Time", "End Time", "Class", "Recording", "Duration"]],
            show_progress=True,
        )

        prediction_select_directory_btn.click(
            get_selection_func("eval-predictions-dir", update_prediction_columns),
            outputs=[prediction_files_state, prediction_directory_input, prediction_group]
            + [
                prediction_columns[label]
                for label in ["Start Time", "End Time", "Class", "Confidence", "Recording", "Duration"]
            ],
            show_progress=True,
        )

        def toggle_after_selection(annotation_files, prediction_files):
            return [gr.update(visible=annotation_files and prediction_files)] * 2

        annotation_directory_input.change(
            toggle_after_selection,
            inputs=[annotation_files_state, prediction_files_state],
            outputs=[mapping_group, class_recording_group],
        )

        prediction_directory_input.change(
            toggle_after_selection,
            inputs=[annotation_files_state, prediction_files_state],
            outputs=[mapping_group, class_recording_group],
        )

        def plot_metrics_all_thresholds(pa: PerformanceAssessor, predictions, labels, class_wise_value):
            if pa is None or predictions is None or labels is None:
                return None, "Please calculate metrics first.", gr.update(visible=False)
            try:
                fig = pa.plot_metrics_all_thresholds(predictions, labels, per_class_metrics=class_wise_value)
                plt.close(fig)

                return fig, gr.update(visible=True)
            except Exception as e:
                raise gr.Error(f"Error plotting metrics for all thresholds: {e}") from e

        plot_metrics_all_thresholds_button.click(
            plot_metrics_all_thresholds,
            inputs=[pa_state, predictions_state, labels_state, class_wise],
            outputs=[plot_output, plot_output],
        )


if __name__ == "__main__":
    gu.open_window(build_evaluation_tab)
