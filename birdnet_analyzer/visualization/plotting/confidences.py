import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from typing import List, Optional, Dict


class ConfidencePlotter:
    """
    A helper class to plot distribution (ridgeline or stacked) plots of confidence scores
    for each class using only matplotlib and plotly.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        class_col: str = "class",
        conf_col: str = "confidence"
    ):
        """
        Initialize the ConfidencePlotter with a DataFrame and column names.

        Args:
            data (pd.DataFrame): Data containing confidence scores and class labels.
            class_col (str): Name of the column that indicates class/species/label.
            conf_col (str): Name of the column that indicates confidence score.
        """
        self.data = data.copy()
        self.class_col = class_col
        self.conf_col = conf_col

        # Ensure these columns exist
        if self.class_col not in self.data.columns:
            raise ValueError(f"Column '{self.class_col}' not found in data.")
        if self.conf_col not in self.data.columns:
            raise ValueError(f"Column '{self.conf_col}' not found in data.")

        # Gather unique classes
        self.classes = sorted(self.data[self.class_col].dropna().unique())

    def plot_ridgeline_matplotlib(
        self,
        figsize=(6, 8),
        bandwidth=0.1,
        overlap=0.6,
        fill=True,
        alpha=0.6,
        cmap="Spectral_r",
        title="Ridgeline Plot of Confidence Scores"
    ):
        """
        Creates a ridgeline plot using matplotlib and scipy's gaussian_kde for each class.

        Args:
            figsize (tuple): Figure size (width, height).
            bandwidth (float): Optional bandwidth scaling for KDE smoothing. 
                               This can be adjusted if the default smoothing is too coarse or too fine.
            overlap (float): Vertical overlap between consecutive distributions (0->none, 1->complete).
            fill (bool): Whether to fill under the KDE curve.
            alpha (float): Transparency of the fill/line.
            cmap (str): Colormap name for the ridgeline gradient.
            title (str): Title for the entire figure.
        """
        # Prepare figure
        fig, ax = plt.subplots(figsize=figsize)

        # Choose a colormap
        cm = plt.get_cmap(cmap)
        num_classes = len(self.classes)

        # For vertical spacing, let's define a y‑offset for each class
        # We'll plot from top to bottom
        y_offsets = np.linspace(0, -(num_classes-1)*overlap, num_classes)

        # Plot each class
        for i, cls in enumerate(self.classes):
            cls_data = self.data.loc[self.data[self.class_col] == cls, self.conf_col].dropna().values
            if len(cls_data) < 2:
                # We need at least 2 points for a KDE
                continue

            # Compute Gaussian KDE
            kde = gaussian_kde(cls_data)
            if bandwidth:
                # Adjust the covariance factor if a bandwidth is given
                kde.set_bandwidth(bw_method=kde.factor * bandwidth)

            # Evaluate KDE on a grid from min to max
            x_min, x_max = cls_data.min(), cls_data.max()
            x_grid = np.linspace(x_min, x_max, 200)
            y_grid = kde.evaluate(x_grid)

            # Vertical offset for the ridgeline
            y_offset = y_offsets[i]
            color = cm(i / num_classes)

            if fill:
                ax.fill_between(
                    x_grid,
                    y_offset,
                    y_grid + y_offset,
                    color=color,
                    alpha=alpha
                )
            ax.plot(
                x_grid,
                y_grid + y_offset,
                color=color,
                alpha=alpha,
                label=cls if i == 0 else None  # only label once if desired
            )

            # Label for each class on the left side
            ax.text(
                x_min,
                y_offset + 0.02,
                cls,
                ha="right",
                va="bottom",
                fontsize=9
            )

        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("Confidence Score")
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    def plot_histogram_plotly(
        self,
        nbins: int = 50,
        facet_col: bool = False,
        title="Histogram of Confidence Scores by Class"
    ):
        """
        Creates a stacked or faceted histogram plot using Plotly Express,
        with each class as a separate color or facet.

        Args:
            nbins (int): Number of bins for the histogram.
            facet_col (bool): If True, show each class in its own column (facet).
                              If False, stack them with different colors.
            title (str): Title of the Plotly figure.

        Returns:
            plotly.graph_objects.Figure: The resulting Plotly figure.
        """
        if facet_col:
            # Use facet_col to place each class in its own subplot
            fig = px.histogram(
                self.data,
                x=self.conf_col,
                color=self.class_col,
                facet_col=self.class_col,
                facet_col_wrap=3,  # Adjust number of columns as needed
                nbins=nbins,
                title=title
            )
            # Adjust layout to avoid overlapping labels
            fig.update_layout(height=600, width=1000)
            fig.update_yaxes(matches=None)  # so each facet has its own y-scale
        else:
            # Use color to differentiate classes in a single histogram
            fig = px.histogram(
                self.data,
                x=self.conf_col,
                color=self.class_col,
                nbins=nbins,
                barmode="overlay",  # or "stack"
                title=title
            )
            fig.update_layout(barmode="overlay", hovermode="x unified")
            fig.update_traces(opacity=0.6)

        fig.update_xaxes(title_text="Confidence Score")
        fig.update_yaxes(title_text="Count")

        fig.show()
        return fig

    def plot_smooth_distribution_plotly(
        self,
        bandwidth: float = 0.2,
        title: str = "Smooth Distribution of Confidence Scores",
        classes: List[str] = None,
        color_map: Dict[str, str] = None
    ) -> go.Figure:
        """
        Creates a smooth distribution plot using plotly.

        Args:
            bandwidth: Width of the smoothing window
            title: Plot title
            classes: List of classes to plot (defaults to all)
            color_map: Dict mapping class names to colors
        """
        if not classes:
            classes = sorted(self.data[self.class_col].unique())
            
        fig = go.Figure()
        dash_patterns = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']

        for i, class_name in enumerate(classes):
            class_data = self.data[self.data[self.class_col] == class_name][self.conf_col].dropna()
            if len(class_data) < 2:
                continue
                
            kde = gaussian_kde(class_data, bw_method=bandwidth)
            x_range = np.linspace(class_data.min(), class_data.max(), 200)
            y_values = kde(x_range)
            
            color = color_map.get(class_name) if color_map else None
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_values,
                    mode="lines",
                    fill="tozeroy",
                    name=str(class_name),
                    line=dict(
                        color=color,
                        dash=dash_patterns[i % len(dash_patterns)]
                    )
                )
            )

        if len(fig.data) == 0:
            raise ValueError("Not enough data points for any of the selected classes.")

        fig.update_layout(
            title=title,
            xaxis_title="Confidence Score",
            yaxis_title="Estimated Density",
            legend_title="Class",
            legend=dict(x=1.02, y=1),
            margin=dict(r=150)  # add right margin for legend
        )
        
        return fig
