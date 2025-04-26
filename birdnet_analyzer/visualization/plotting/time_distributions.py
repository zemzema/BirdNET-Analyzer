from typing import List, Optional, Dict, Tuple
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.colors as mcolors  # For color conversion

class TimeDistributionPlotter:
    def __init__(self, data: pd.DataFrame, class_col: str):
        self.data = data
        self.class_col = class_col
        self.base_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
        
    def _get_color_map(self, classes: List[str]) -> Dict[str, str]:
        """Create consistent color mapping for classes."""
        colors = self.base_colors * (1 + len(classes) // len(self.base_colors))
        return {cls: colors[i] for i, cls in enumerate(sorted(classes))}
    
    def _color_to_rgba(self, color: str, opacity: float = 0.5) -> str:
        """Convert any color format to rgba string."""
        # Convert named color or hex to RGB values
        rgb = mcolors.to_rgb(color)
        # Scale to 0-255 range and build rgba string
        return f'rgba({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)},{opacity})'
        
    def _aggregate_by_period(self, cls_data: pd.DataFrame, time_period: str):
        """Aggregate data by specified time period."""
        if time_period == 'hour':
            # Count detections for each hour of each day
            return (cls_data.groupby(['year', 'month', 'day', 'hour'])
                   .size().reset_index(name='count')
                   .groupby('hour'))
        elif time_period == 'day':
            # Count detections for each weekday of each week
            return (cls_data.groupby(['year', 'month', 'day', 'weekday_name'])
                   .size().reset_index(name='count')
                   .groupby('weekday_name'))
        elif time_period == 'month':
            # Count detections for each month of each year
            return (cls_data.groupby(['year', 'month_name'])
                   .size().reset_index(name='count')
                   .groupby('month_name'))
        else:  # year
            # Count detections for each day of each year
            return (cls_data.groupby(['year', 'month', 'day'])
                   .size().reset_index(name='count')
                   .groupby('year'))
        
    def plot_distribution(self, time_period: str, use_boxplot: bool = False, title: str = None) -> go.Figure:
        """Plot species counts distribution over the specified time period."""
        if self.data.empty:
            raise ValueError("No data to plot")
            
        # Set up period-specific configurations
        if time_period == 'hour':
            x_title = 'Hour of Day'
            x_values = list(range(24))
            x_text = [f"{h:02d}:00" for h in range(24)]
        elif time_period == 'day':
            x_title = 'Day of Week'
            x_values = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            x_text = x_values
        elif time_period == 'month':
            x_title = 'Month'
            x_values = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
            x_text = x_values
        elif time_period == 'year':
            x_title = 'Year'
            x_values = sorted(self.data['year'].unique())
            x_text = [str(y) for y in x_values]
        else:
            raise ValueError(f"Invalid time period: {time_period}")
            
        # Create figure
        fig = go.Figure()
        classes = sorted(self.data[self.class_col].unique())
        color_map = self._get_color_map(classes)
        
        if use_boxplot:
            # Offset boxplots for each class to avoid overlap
            num_classes = len(classes)
            offsets = np.linspace(-0.3, 0.3, num_classes)
            
            for i, cls in enumerate(classes):
                cls_data = self.data[self.data[self.class_col] == cls]
                
                # Prepare data for boxplot
                period_values = []
                y_values = []
                x_positions = []
                
                # Get data for each period
                for j, period in enumerate(x_values):
                    # Find data for current period
                    agg_data = self._aggregate_by_period(cls_data, time_period)
                    if period in agg_data.groups:
                        period_counts = agg_data.get_group(period)['count'].values
                        if len(period_counts) > 0:
                            # Store data for boxplot
                            period_values.extend([str(period)] * len(period_counts))
                            y_values.extend(period_counts)
                            # Calculate offset position
                            x_positions.extend([j + offsets[i]] * len(period_counts))
                
                if len(period_values) > 0:
                    # Convert color to rgba format
                    color = color_map.get(cls)
                    rgba_color = self._color_to_rgba(color, 0.5)
                    
                    # Create a single boxplot for this class with custom positions
                    fig.add_trace(go.Box(
                        x=x_positions,
                        y=y_values,
                        name=str(cls),
                        marker_color=color,
                        boxpoints='outliers',  # Show outliers
                        jitter=0.3,           # Add jitter to points
                        pointpos=0,           # Offset of points from box
                        line=dict(width=2),   # Box line width
                        fillcolor=rgba_color,  # Use properly converted color
                        hovertemplate=(
                            "Species: " + str(cls) + "<br>" +
                            "Count: %{y}<br>" +
                            "Median: %{median}<br>" +
                            "Q1: %{q1}<br>" +
                            "Q3: %{q3}<br>" +
                            "<extra></extra>"
                        )
                    ))
            
            # Set custom x-axis ticks and labels
            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(x_text))),
                    ticktext=x_text
                )
            )
        else:
            # Original histogram plotting logic
            for cls in classes:
                cls_data = self.data[self.data[self.class_col] == cls]
                grouped = self._aggregate_by_period(cls_data, time_period)
                
                counts = []
                for val in x_values:
                    count = (grouped.get_group(val)['count'].sum() 
                            if val in grouped.groups else 0)
                    counts.append(count)
                    
                fig.add_trace(go.Bar(
                    name=str(cls),
                    x=x_text,
                    y=counts,
                    marker_color=color_map.get(cls),
                    opacity=0.6,
                    hovertemplate=(
                        f"{x_title}: %{{x}}<br>"
                        "Species: " + str(cls) + "<br>"
                        "Count: %{y}<br>"
                        "<extra></extra>"
                    )
                ))
        
        # Update layout
        plot_type = "Boxplots" if use_boxplot else "Distribution"
        fig.update_layout(
            barmode='group' if not use_boxplot else None,  # Changed from 'overlay' to 'group'
            boxmode='group' if use_boxplot else None,
            title=title or f'Species {plot_type} by {x_title}', # if year, it is visualized by all days of that year.
            xaxis_title=x_title,
            yaxis_title='Count',
            legend_title='Species',
            legend=dict(x=1.02, y=1),
            margin=dict(r=150),
            showlegend=True
        )
        
        return fig
