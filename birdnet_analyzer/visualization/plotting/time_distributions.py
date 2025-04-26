from typing import List, Optional, Dict
import pandas as pd
import plotly.graph_objects as go

class TimeDistributionPlotter:
    """Plotter for time-based distribution of species counts."""
    
    def __init__(self, data: pd.DataFrame, class_col: str):
        self.data = data
        self.class_col = class_col
        self.base_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
        
    def _get_color_map(self, classes: List[str]) -> Dict[str, str]:
        """Create consistent color mapping for classes."""
        colors = self.base_colors * (1 + len(classes) // len(self.base_colors))
        return {cls: colors[i] for i, cls in enumerate(sorted(classes))}
        
    def plot_distribution(self, time_period: str, title: str = None) -> go.Figure:
        """Plot species counts distribution over the specified time period."""
        if self.data.empty:
            raise ValueError("No data to plot")
            
        # Determine grouping based on time period
        if time_period == 'hour':
            group_col = 'hour'
            x_title = 'Hour of Day'
            x_values = list(range(24))
            x_text = [f"{h:02d}:00" for h in range(24)]
        elif time_period == 'day':
            group_col = 'weekday_name'
            x_title = 'Day of Week'
            x_values = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            x_text = x_values
        elif time_period == 'month':
            group_col = 'month_name'
            x_title = 'Month'
            x_values = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
            x_text = x_values
        elif time_period == 'year':
            group_col = 'year'
            x_title = 'Year'
            x_values = sorted(self.data['year'].unique())
            x_text = [str(y) for y in x_values]
        else:
            raise ValueError(f"Invalid time period: {time_period}")
            
        # Group data and get counts
        grouped = self.data.groupby([group_col, self.class_col]).size().reset_index(name='count')
        classes = sorted(grouped[self.class_col].unique())
        color_map = self._get_color_map(classes)
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each class
        for cls in classes:
            cls_data = grouped[grouped[self.class_col] == cls]
            counts = []
            
            # Ensure all time periods are represented
            for val in x_values:
                count = cls_data[cls_data[group_col] == val]['count'].iloc[0] if val in cls_data[group_col].values else 0
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
        fig.update_layout(
            barmode='overlay',
            title=title or f'Species Counts by {x_title}',
            xaxis_title=x_title,
            yaxis_title='Count',
            legend_title='Species',
            legend=dict(x=1.02, y=1),
            margin=dict(r=150),
            showlegend=True
        )
        
        return fig
