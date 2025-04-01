import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Optional

def create_sunburst_chart(contributions: Dict[str, float]) -> go.Figure:
    """
    Create a sunburst chart visualization of channel contributions.
    
    Parameters:
    -----------
    contributions : Dict[str, float]
        Dictionary mapping channel names to their contribution values
        
    Returns:
    --------
    go.Figure
        Plotly figure object with the sunburst chart
    """
    # Filter out the base effect for separate handling
    channel_contribs = {k: v for k, v in contributions.items() if k != "base_effect"}
    
    # Channel labels and values
    channel_labels = list(channel_contribs.keys())
    channel_values = list(channel_contribs.values())
    
    # For sunburst chart, we need to create parent-child relationships
    labels = ["Total"] + channel_labels
    parents = [""] + ["Total" for _ in channel_labels]
    values = [sum(channel_values)] + channel_values
    
    # Add base effect if it exists
    if "base_effect" in contributions:
        labels.append("Base Effect")
        parents.append("Total")
        values.append(contributions["base_effect"])
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(
            colors=["#4FC3F7"] + px.colors.sequential.Plasma[:len(labels)-1],
            line=dict(color="#121212", width=1)
        ),
        textinfo="label+percent",
        insidetextorientation="radial"
    ))
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(t=10, b=10, r=10, l=10),
        height=500
    )
    
    return fig

def create_contribution_bar_chart(contributions: Dict[str, float]) -> go.Figure:
    """
    Create a bar chart of channel contributions.
    
    Parameters:
    -----------
    contributions : Dict[str, float]
        Dictionary mapping channel names to their contribution values
        
    Returns:
    --------
    go.Figure
        Plotly figure object with the bar chart
    """
    # Filter out the base effect
    channel_contribs = {k: v for k, v in contributions.items() if k != "base_effect"}
    
    # Sort by contribution value
    sorted_channels = sorted(
        [(channel, value) for channel, value in channel_contribs.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    if not sorted_channels:
        # Return empty figure if no channels
        return go.Figure()
    
    channels, values = zip(*sorted_channels)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(channels),
        y=list(values),
        marker_color=px.colors.sequential.Plasma[:len(channels)],
        text=[f"${v:.2f}" for v in values],
        textposition="auto"
    ))
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(30, 30, 30, 0.3)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        xaxis_title="Marketing Channel",
        yaxis_title="Contribution Value",
        height=400
    )
    
    return fig

def create_stacked_area_chart(
    dates: List,
    channel_contributions: Dict[str, np.ndarray],
    base_effect: np.ndarray,
    actual_outcomes: Optional[List] = None
) -> go.Figure:
    """
    Create a stacked area chart of time series attribution.
    
    Parameters:
    -----------
    dates : List
        List of dates for the x-axis
    channel_contributions : Dict[str, np.ndarray]
        Dictionary mapping channel names to arrays of contribution values over time
    base_effect : np.ndarray
        Array of base effect values over time
    actual_outcomes : Optional[List]
        List of actual outcome values to overlay on the chart
        
    Returns:
    --------
    go.Figure
        Plotly figure object with the stacked area chart
    """
    fig = go.Figure()
    
    # Add base effect first
    fig.add_trace(go.Scatter(
        x=dates,
        y=base_effect,
        mode='lines',
        line=dict(width=0),
        stackgroup='one',
        name='Base Effect',
        fillcolor='rgba(128, 128, 128, 0.7)'
    ))
    
    # Add each channel
    for i, (channel, values) in enumerate(channel_contributions.items()):
        color_idx = i % len(px.colors.sequential.Plasma)
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            line=dict(width=0),
            stackgroup='one',
            name=channel,
            fillcolor=px.colors.sequential.Plasma[color_idx]
        ))
    
    # Add actual outcome data if provided
    if actual_outcomes is not None:
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual_outcomes,
            mode='lines',
            name='Actual Outcome',
            line=dict(color='white', width=2, dash='dot')
        ))
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(30, 30, 30, 0.3)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        xaxis_title="Date",
        yaxis_title="Contribution",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def create_budget_comparison_chart(
    channels: List[str],
    current_values: List[float],
    optimized_values: List[float]
) -> go.Figure:
    """
    Create a bar chart comparing current and optimized budget allocations.
    
    Parameters:
    -----------
    channels : List[str]
        List of channel names
    current_values : List[float]
        List of current budget values
    optimized_values : List[float]
        List of optimized budget values
        
    Returns:
    --------
    go.Figure
        Plotly figure object with the comparison bar chart
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=channels,
        y=current_values,
        name="Current Spend",
        marker_color='rgba(79, 195, 247, 0.7)'
    ))
    
    fig.add_trace(go.Bar(
        x=channels,
        y=optimized_values,
        name="Optimized Spend",
        marker_color='rgba(126, 87, 194, 0.7)'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(30, 30, 30, 0.3)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        xaxis_title="Channel",
        yaxis_title="Budget Allocation",
        barmode='group',
        height=400
    )
    
    return fig

def create_roi_table(comparison_data: List[Dict]) -> go.Figure:
    """
    Create a table with ROI and budget comparison data.
    
    Parameters:
    -----------
    comparison_data : List[Dict]
        List of dictionaries with channel comparison data
        
    Returns:
    --------
    go.Figure
        Plotly figure object with the table
    """
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Channel", "Current", "Optimized", "Change", "ROI"],
            fill_color="#1E1E1E",
            align="left",
            font=dict(color="white", size=12)
        ),
        cells=dict(
            values=[
                [d["Channel"] for d in comparison_data],
                [f"${d['Current Spend']:,.0f}" for d in comparison_data],
                [f"${d['Optimized Spend']:,.0f}" for d in comparison_data],
                [f"{d['Change %']:+.1f}%" for d in comparison_data],
                [f"{d['ROI']:.2f}x" for d in comparison_data]
            ],
            fill_color=[[
                "#262626" if d["Change %"] == 0 else
                "rgba(76, 175, 80, 0.1)" if d["Change %"] > 0 else
                "rgba(244, 67, 54, 0.1)" for d in comparison_data
            ]],
            align="left",
            font=dict(color="white", size=11),
            height=30
        )
    )])
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(t=0, b=0, r=0, l=0),
        height=400
    )
    
    return fig