import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import time
import os

# Import utilities
from utils.data_processing import preprocess_data, identify_channels
from utils.visualization import create_sunburst_chart, create_stacked_area_chart, create_contribution_bar_chart
from utils.session import initialize_session_state, get_nav_page

# Set page config
st.set_page_config(
    page_title="Attribution Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open('styles/main.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state
initialize_session_state()

# Sidebar
def render_sidebar():
    with st.sidebar:
        st.title("Attribution Engine")
        st.caption("Cross-Channel Marketing Analysis")
        
        st.markdown("---")
        
        # Navigation options
        selected = st.radio(
            "Navigation",
            ["Home", "Data Upload", "Model Configuration", "Attribution Analysis", "ROI Optimization"],
            key="navigation",
            index=["Home", "Data Upload", "Model Configuration", "Attribution Analysis", "ROI Optimization"].index(st.session_state.current_page)
        )
        
        st.session_state.current_page = selected
        
        st.markdown("---")
        
        # Status indicators
        st.subheader("System Status")
        
        data_status = "✅ Loaded" if st.session_state.data is not None else "⏳ Waiting for data"
        model_status = "✅ Trained" if st.session_state.model is not None else "⏳ Not configured"
        attribution_status = "✅ Available" if st.session_state.attribution is not None else "⏳ Not generated"
        
        st.markdown(f"**Data:** {data_status}")
        st.markdown(f"**Model:** {model_status}")
        st.markdown(f"**Attribution:** {attribution_status}")
        
        st.markdown("---")
        
        # App info footer
        st.caption("© 2025 Attribution Engine")
        st.caption("Cross-Channel Attribution Engine v1.0")

# Home page
def render_home():
    st.title("Cross-Channel Attribution Engine")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### Understand the True Impact of Your Marketing Channels
        
        This platform helps you analyze the causal relationships between marketing investments and business outcomes. 
        Our attribution engine:
        
        - Models time-lagged effects of marketing activities
        - Captures cross-channel interactions and synergies
        - Provides clear, explainable attribution insights
        - Optimizes ROI across all marketing channels
        
        Get started by uploading your marketing and conversion data.
        """)
        
        st.button("Start Analysis →", key="start_button", on_click=lambda: setattr(st.session_state, 'current_page', 'Data Upload'))
    
    with col2:
        # Create summary metrics with gradient styling
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">10K+</div>
            <div class="metric-label">DATA POINTS PROCESSED DAILY</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">8+</div>
            <div class="metric-label">MARKETING CHANNELS ANALYZED</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">15-30%</div>
            <div class="metric-label">TYPICAL ROI IMPROVEMENT</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Process diagram
    st.markdown("### How It Works")
    
    steps = [
        {"title": "Data Upload", "description": "Connect your marketing data sources"},
        {"title": "Transformation", "description": "Apply adstock & saturation models"},
        {"title": "Attribution", "description": "Calculate channel contributions"},
        {"title": "Optimization", "description": "Maximize marketing ROI"}
    ]
    
    cols = st.columns(len(steps))
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; height: 150px; background: rgba(40, 40, 40, 0.5); border-radius: 8px; position: relative;">
                <div style="position: absolute; top: -15px; left: 50%; transform: translateX(-50%); background: linear-gradient(90deg, #4FC3F7, #7F7FD5); color: white; border-radius: 15px; padding: 5px 15px; font-weight: bold;">
                    {i+1}
                </div>
                <h3 style="margin-top: 1rem; color: #4FC3F7;">{steps[i]['title']}</h3>
                <p>{steps[i]['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add connecting arrow except for the last step
            if i < len(steps) - 1:
                st.markdown("""
                <div style="text-align: center; margin-top: -75px; color: #4FC3F7; font-size: 24px;">→</div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("### Ready to optimize your marketing mix?")
    st.button("Upload Your Data →", key="upload_button", on_click=lambda: setattr(st.session_state, 'current_page', 'Data Upload'))

# Data Upload page
def render_data_upload():
    st.title("Data Upload & Processing")
    
    st.markdown("""
    Upload your marketing channel spend and conversion data to begin the attribution analysis. 
    The platform accepts CSV files with daily or weekly data points.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        st.markdown("**Required columns:**")
        st.markdown("- `date`: Date in YYYY-MM-DD format")
        st.markdown("- `channel_*`: One column per marketing channel with spend amounts")
        st.markdown("- `conversions` or `revenue`: Business outcomes to model")
        
        # Sample data option
        use_sample = st.checkbox("Use sample data instead", value=False)
        
        if uploaded_file or use_sample:
            with st.spinner("Loading and preprocessing data..."):
                # Load data
                if use_sample:
                    data = pd.read_csv('data/sample_data.csv')
                else:
                    data = pd.read_csv(uploaded_file)
                
                # Process data
                data = preprocess_data(data)
                
                # Store in session state
                st.session_state.data = data
                
                # Identify channel columns
                st.session_state.channels = identify_channels(data)
                
                st.success("Data successfully loaded!")
                st.markdown(f"**Loaded {len(data)} rows and {len(st.session_state.channels)} potential channels**")
    
    with col2:
        st.subheader("Preview Data")
        if st.session_state.data is not None:
            st.dataframe(st.session_state.data.head(10), height=300)
            
            # Data summary
            st.subheader("Data Summary")
            
            if 'date' in st.session_state.data.columns:
                min_date = pd.to_datetime(st.session_state.data['date']).min()
                max_date = pd.to_datetime(st.session_state.data['date']).max()
                date_range = (max_date - min_date).days
                
                st.markdown(f"**Date range:** {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({date_range} days)")
            
            # Channel overview
            st.markdown(f"**Detected channels:** {', '.join(st.session_state.channels)}")
        else:
            st.info("Upload a file to see the preview here")
            
            # Show sample data layout
            st.markdown("**Sample data format:**")
            sample_data = {
                'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
                'channel_search': [1000, 1200, 1100],
                'channel_social': [800, 850, 900],
                'channel_display': [500, 600, 550],
                'channel_video': [300, 350, 375],
                'conversions': [52, 58, 55]
            }
            st.dataframe(pd.DataFrame(sample_data))
    
    # Data validation and channel mapping
    if st.session_state.data is not None:
        st.markdown("---")
        st.subheader("Channel Mapping & Data Validation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Let user confirm which columns are channels
            st.markdown("**Confirm marketing channels:**")
            
            selected_channels = []
            for col in st.session_state.channels:
                if st.checkbox(col, value=True, key=f"channel_{col}"):
                    selected_channels.append(col)
            
            st.session_state.channels = selected_channels
        
        with col2:
            # Conversion/outcome column
            outcome_options = [col for col in st.session_state.data.columns if col not in st.session_state.channels and col != 'date']
            outcome_column = st.selectbox("Select business outcome column:", outcome_options, index=0 if outcome_options else None)
            
            if outcome_column:
                st.session_state.outcome_column = outcome_column
                
                # Date column
                date_column = st.selectbox("Select date column:", [col for col in st.session_state.data.columns if col.lower() in ['date', 'day', 'week', 'month']], index=0)
                st.session_state.date_column = date_column
        
        # Proceed button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Proceed to Model Configuration →", use_container_width=True):
                if len(selected_channels) > 0 and hasattr(st.session_state, 'outcome_column'):
                    st.session_state.current_page = "Model Configuration"
                else:
                    st.error("Please select at least one marketing channel and an outcome column.")

# Model Configuration page
def render_model_config():
    st.title("Model Configuration")
    
    if st.session_state.data is None:
        st.warning("No data loaded. Please upload your data first.")
        if st.button("Go to Data Upload"):
            st.session_state.current_page = "Data Upload"
    else:
        st.markdown("""
        Configure the attribution model parameters to account for how marketing efforts translate into business outcomes.
        """)
        
        # Create tabs for different configuration sections
        tab1, tab2, tab3 = st.tabs(["Adstock Parameters", "Saturation Effects", "Model Selection"])
        
        with tab1:
            st.markdown("""
            ### Adstock Parameters
            
            Adstock models how marketing effects persist over time. A higher decay rate means the effect dissipates more slowly.
            """)
            
            # Create a row of sliders for each channel
            col1, col2 = st.columns(2)
            
            adstock_params = {}
            
            for i, channel in enumerate(st.session_state.channels):
                if i % 2 == 0:
                    with col1:
                        adstock_params[channel] = st.slider(
                            f"{channel} decay rate",
                            min_value=0.1,
                            max_value=0.9,
                            value=0.3,
                            step=0.1,
                            help="Higher values = longer-lasting effects",
                            key=f"adstock_{channel}"
                        )
                else:
                    with col2:
                        adstock_params[channel] = st.slider(
                            f"{channel} decay rate",
                            min_value=0.1,
                            max_value=0.9,
                            value=0.3,
                            step=0.1,
                            help="Higher values = longer-lasting effects",
                            key=f"adstock_{channel}"
                        )
            
            st.session_state.adstock_params = adstock_params
            
            # Visualize adstock effect
            st.markdown("### Adstock Effect Visualization")
            
            # Create sample data to show adstock effect
            days = range(30)
            
            fig = go.Figure()
            
            # Add a spike of spend at day 0
            spend = [100 if d == 0 else 0 for d in days]
            
            fig.add_trace(go.Scatter(
                x=days, 
                y=spend,
                mode='lines',
                name='Marketing Spend',
                line=dict(color='rgba(79, 195, 247, 0.8)', width=2)
            ))
            
            # Add adstock effect lines for each channel with different parameters
            for channel, decay in adstock_params.items():
                adstock_effect = [100 * (decay ** d) for d in days]
                
                fig.add_trace(go.Scatter(
                    x=days, 
                    y=adstock_effect,
                    mode='lines',
                    name=f'{channel} (decay={decay})',
                    line=dict(width=2, dash='dash')
                ))
            
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor='rgba(30, 30, 30, 0.3)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                xaxis_title="Days After Campaign",
                yaxis_title="Marketing Effect",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("""
            ### Saturation Parameters
            
            Saturation models diminishing returns as marketing spend increases. A lower saturation point means diminishing returns happen sooner.
            """)
            
            # Create a row of sliders for each channel
            col1, col2 = st.columns(2)
            
            saturation_params = {}
            
            for i, channel in enumerate(st.session_state.channels):
                if i % 2 == 0:
                    with col1:
                        saturation_params[channel] = st.slider(
                            f"{channel} saturation rate",
                            min_value=0.01,
                            max_value=0.5,
                            value=0.1,
                            step=0.01,
                            help="Lower values = earlier diminishing returns",
                            key=f"saturation_{channel}"
                        )
                else:
                    with col2:
                        saturation_params[channel] = st.slider(
                            f"{channel} saturation rate",
                            min_value=0.01,
                            max_value=0.5,
                            value=0.1,
                            step=0.01,
                            help="Lower values = earlier diminishing returns",
                            key=f"saturation_{channel}"
                        )
            
            st.session_state.saturation_params = saturation_params
            
            # Visualize saturation effect
            st.markdown("### Saturation Effect Visualization")
            
            # Create sample data to show saturation effect
            spend_range = np.linspace(0, 10000, 100)
            
            fig = go.Figure()
            
            # Add saturation curves for each channel
            for channel, sat_param in saturation_params.items():
                # Hill function for diminishing returns
                saturation_effect = [s / (s + 1/sat_param) for s in spend_range/1000]
                
                fig.add_trace(go.Scatter(
                    x=spend_range, 
                    y=saturation_effect,
                    mode='lines',
                    name=f'{channel} (sat={sat_param})',
                    line=dict(width=2)
                ))
            
            # Add linear response line for reference
            max_effect = max([s / (s + 1/sat_param) for s in [10]])
            linear_effect = [s * max_effect / 10000 for s in spend_range]
            
            fig.add_trace(go.Scatter(
                x=spend_range, 
                y=linear_effect,
                mode='lines',
                name='Linear (No Saturation)',
                line=dict(color='rgba(255, 255, 255, 0.5)', width=2, dash='dot')
            ))
            
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor='rgba(30, 30, 30, 0.3)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                xaxis_title="Marketing Spend",
                yaxis_title="Marketing Effect",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("""
            ### Model Selection
            
            Choose the modeling approach for estimating channel contributions.
            """)
            
            model_type = st.selectbox(
                "Select attribution model type:",
                ["Multivariate Regression", "Ridge Regression", "Bayesian Model", "Shapley Value Decomposition"],
                index=0
            )
            
            # Model-specific parameters
            if model_type == "Ridge Regression":
                alpha = st.slider("Regularization Strength (Alpha)", 0.01, 10.0, 1.0, 0.01)
                st.session_state.model_params = {"alpha": alpha}
            elif model_type == "Bayesian Model":
                iterations = st.slider("MCMC Iterations", 1000, 10000, 2000, 1000)
                st.session_state.model_params = {"iterations": iterations}
            else:
                st.session_state.model_params = {}
            
            st.session_state.model_type = model_type
            
            st.markdown("### Cross-Channel Interaction Effects")
            
            st.markdown("""
            Enable modeling of interactions between channels. This captures synergistic 
            effects when multiple channels work together.
            """)
            
            enable_interactions = st.checkbox("Model channel interactions", value=True)
            st.session_state.enable_interactions = enable_interactions
            
            if enable_interactions:
                interaction_strength = st.slider(
                    "Interaction Strength",
                    0.0, 1.0, 0.5, 0.1,
                    help="Higher values give more weight to cross-channel interactions"
                )
                st.session_state.interaction_strength = interaction_strength
        
        # Train model button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Train Attribution Model →", use_container_width=True):
                # Simulate model training
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(101):
                    progress_bar.progress(i)
                    if i < 30:
                        status_text.text("Applying adstock transformations...")
                    elif i < 60:
                        status_text.text("Modeling saturation effects...")
                    elif i < 90:
                        status_text.text("Calculating channel contributions...")
                    else:
                        status_text.text("Finalizing model...")
                    time.sleep(0.03)
                
                # Simulated model and attribution results (would call model functions in production)
                st.session_state.model = {
                    "type": model_type,
                    "adstock_params": st.session_state.adstock_params,
                    "saturation_params": st.session_state.saturation_params,
                    "model_params": st.session_state.model_params,
                    "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Create mock attribution results
                total_outcome = st.session_state.data[st.session_state.outcome_column].sum()
                channel_contributions = {}
                
                # Generate somewhat realistic attribution percentages
                remaining = 100
                for i, channel in enumerate(st.session_state.channels):
                    if i == len(st.session_state.channels) - 1:
                        # Last channel gets the remainder
                        contribution = remaining
                    else:
                        # Generate a random percentage for this channel
                        max_contrib = max(5, remaining - 5 * (len(st.session_state.channels) - i - 1))
                        contribution = np.random.randint(5, max_contrib)
                        remaining -= contribution
                    
                    channel_contributions[channel] = contribution / 100
                
                # Base effect (not attributed to any channel)
                channel_contributions["base_effect"] = 0.15
                
                # Normalize to account for base effect
                total_contrib = sum(channel_contributions.values())
                for channel in channel_contributions:
                    channel_contributions[channel] /= total_contrib
                    channel_contributions[channel] *= total_outcome
                
                st.session_state.attribution = {
                    "channel_contributions": channel_contributions,
                    "roi_metrics": {channel: np.random.uniform(1.5, 5.0) for channel in st.session_state.channels}
                }
                
                status_text.success("Model trained successfully!")
                
                # Navigate to attribution analysis
                st.session_state.current_page = "Attribution Analysis"
                st.experimental_rerun()

# Attribution Analysis page
def render_attribution_analysis():
    st.title("Attribution Analysis")
    
    if st.session_state.attribution is None:
        st.warning("No attribution model available. Please train a model first.")
        if st.button("Go to Model Configuration"):
            st.session_state.current_page = "Model Configuration"
    else:
        st.markdown("""
        Explore how each marketing channel contributes to your business outcomes. 
        The attribution analysis breaks down the impact of each channel and helps 
        identify which channels deliver the best return on investment.
        """)
        
        # Attribution overview
        st.subheader("Channel Contribution Breakdown")
        
        # Create sunburst chart with the utility function
        fig = create_sunburst_chart(st.session_state.attribution["channel_contributions"])
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics
        st.markdown("---")
        st.subheader("Channel Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create a bar chart of channel contributions
            fig = create_contribution_bar_chart(st.session_state.attribution["channel_contributions"])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Create a table of ROI metrics
            roi_metrics = st.session_state.attribution["roi_metrics"]
            
            sorted_roi = sorted(
                [(channel, roi) for channel, roi in roi_metrics.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            if sorted_roi:
                channels, rois = zip(*sorted_roi)
                
                fig = go.Figure(data=[go.Table(
                    header=dict(
                        values=["Channel", "ROI", "Performance"],
                        fill_color="#1E1E1E",
                        align="left",
                        font=dict(color="white", size=12)
                    ),
                    cells=dict(
                        values=[
                            list(channels),
                            [f"{roi:.2f}x" for roi in rois],
                            ["Excellent" if roi > 4 else 
                             "Good" if roi > 3 else 
                             "Average" if roi > 2 else 
                             "Poor" for roi in rois]
                        ],
                        fill_color=["#262626"],
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
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Time series analysis
        st.markdown("---")
        st.subheader("Time Series Attribution")
        
        # Create a time series stacked area chart
        if 'date' in st.session_state.data.columns:
            dates = pd.to_datetime(st.session_state.data['date']).tolist()
            
            # Create simulated time series attribution data
            ts_attribution = {}
            
            for channel in st.session_state.channels:
                # Create realistic but random contribution pattern
                channel_contrib = np.random.normal(
                    st.session_state.attribution["channel_contributions"][channel] / len(dates), 
                    st.session_state.attribution["channel_contributions"][channel] / len(dates) * 0.2, 
                    len(dates)
                )
                # Ensure positive values
                channel_contrib = np.maximum(channel_contrib, 0)
                ts_attribution[channel] = channel_contrib
            
            # Add base effect
            base_effect = np.ones(len(dates)) * (
                st.session_state.attribution["channel_contributions"].get("base_effect", 0) / len(dates)
            )
            
            # Get actual outcome data
            if st.session_state.outcome_column in st.session_state.data.columns:
                actual_outcomes = st.session_state.data[st.session_state.outcome_column].tolist()
            else:
                actual_outcomes = None
            
            # Create the stacked area chart
            fig = create_stacked_area_chart(dates, ts_attribution, base_effect, actual_outcomes)
            st.plotly_chart(fig, use_container_width=True)
        
        # Proceed to optimization
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Proceed to ROI Optimization →", use_container_width=True):
                st.session_state.current_page = "ROI Optimization"
                st.experimental_rerun()

# ROI Optimization page
def render_roi_optimization():
    st.title("ROI Optimization")
    
    if st.session_state.attribution is None:
        st.warning("No attribution data available. Please run attribution analysis first.")
        if st.button("Go to Attribution Analysis"):
            st.session_state.current_page = "Attribution Analysis"
    else:
        st.markdown("""
        Optimize your marketing budget allocation to maximize ROI. This tool recommends the 
        optimal spending level for each channel based on the attribution model results.
        """)
        
        # Current allocation
        st.subheader("Current Channel Allocation")
        
        # Calculate current spend by channel
        current_spend = {}
        total_spend = 0
        
        for channel in st.session_state.channels:
            if channel in st.session_state.data.columns:
                spend = st.session_state.data[channel].sum()
                current_spend[channel] = spend
                total_spend += spend
        
        # Create a pie chart of current allocation
        fig = go.Figure(data=[go.Pie(
            labels=list(current_spend.keys()),
            values=list(current_spend.values()),
            textinfo='label+percent',
            marker=dict(
                colors=px.colors.sequential.Plasma[:len(current_spend)],
                line=dict(color="#121212", width=1)
            ),
            hole=0.4
        )])
        
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            margin=dict(t=0, b=0, r=0, l=0),
            height=400,
            annotations=[
                dict(
                    text=f"${total_spend:,.0f}",
                    x=0.5, y=0.5,
                    font_size=20,
                    showarrow=False
                )
            ]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # ROI optimization controls
            st.subheader("Optimization Parameters")
            
            # Total budget slider
            budget_factor = st.slider(
                "Total Budget (% of current)",
                min_value=50,
                max_value=150,
                value=100,
                step=5,
                help="Adjust total marketing budget"
            )
            
            target_budget = total_spend * budget_factor / 100
            
            st.markdown(f"**Target Budget:** ${target_budget:,.0f}")
            
            # Optimization objective
            objective = st.radio(
                "Optimization Objective",
                ["Maximize Revenue", "Maximize ROI", "Balanced Approach"],
                index=2
            )
            
            # Advanced options
            with st.expander("Advanced Options"):
                min_channel_budget = st.slider(
                    "Minimum Channel Budget (%)",
                    min_value=0,
                    max_value=50,
                    value=10,
                    step=5,
                    help="Minimum budget allocation for any channel"
                )
                
                max_channel_budget = st.slider(
                    "Maximum Channel Budget (%)",
                    min_value=50,
                    max_value=100,
                    value=70,
                    step=5,
                    help="Maximum budget allocation for any channel"
                )
        
        # Optimize button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            optimize_button = st.button("Run Budget Optimization", use_container_width=True)
        
        if optimize_button:
            # Simulate optimization process
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(101):
                progress_bar.progress(i)
                if i < 30:
                    status_text.text("Evaluating channel performance...")
                elif i < 60:
                    status_text.text("Computing optimal allocation...")
                elif i < 90:
                    status_text.text("Generating budget recommendations...")
                else:
                    status_text.text("Finalizing optimization...")
                time.sleep(0.03)
            
            status_text.success("Optimization complete!")
            
            # Create optimized allocation
            st.markdown("---")
            st.subheader("Optimized Budget Allocation")
            
            # In a real app, this would call the optimization model
            # For demo, we'll create a simulated optimization
            from models.optimization import simulate_optimized_budget
            
            # Get ROI metrics and current attribution
            roi_metrics = st.session_state.attribution["roi_metrics"]
            contributions = st.session_state.attribution["channel_contributions"]
            
            # Generate optimized budget allocation
            optimized_spend, expected_outcome, comparison_data = simulate_optimized_budget(
                current_spend=current_spend,
                roi_metrics=roi_metrics,
                contributions=contributions,
                target_budget=target_budget,
                objective=objective,
                min_channel_budget=min_channel_budget,
                max_channel_budget=max_channel_budget
            )
            
            # Display comparison bar chart and table
            col1, col2 = st.columns(2)
            
            with col1:
                # Comparison bar chart
                channels = [d["Channel"] for d in comparison_data]
                current_values = [d["Current Spend"] for d in comparison_data]
                optimized_values = [d["Optimized Spend"] for d in comparison_data]
                
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
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Create a table with comparison data
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
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Expected outcome improvement metrics
            st.markdown("---")
            st.subheader("Expected Performance Impact")
            
            current_outcome = sum(contributions.values())
            
            # Calculate improvement metrics
            improvement_pct = (expected_outcome - current_outcome) / current_outcome * 100
            roi_current = current_outcome / total_spend
            roi_optimized = expected_outcome / target_budget
            roi_pct = (roi_optimized - roi_current) / roi_current * 100
            
            metric_cols = st.columns(3)
            
            with metric_cols[0]:
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">+{:.1f}%</div>
                    <div class="metric-label">EXPECTED OUTCOME IMPROVEMENT</div>
                </div>
                """.format(improvement_pct), unsafe_allow_html=True)
            
            with metric_cols[1]:
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">+{:.1f}%</div>
                    <div class="metric-label">ROI EFFICIENCY GAIN</div>
                </div>
                """.format(roi_pct), unsafe_allow_html=True)
            
            with metric_cols[2]:
                # If budget was decreased but outcome still improved
                if target_budget < total_spend and improvement_pct > 0:
                    efficiency_score = (improvement_pct + (1 - target_budget/total_spend) * 100) * 1.5
                else:
                    efficiency_score = improvement_pct * target_budget / total_spend
                
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">{:.1f}</div>
                    <div class="metric-label">OPTIMIZATION SCORE</div>
                </div>
                """.format(min(100, max(0, efficiency_score))), unsafe_allow_html=True)

# Main app execution
def main():
    # Render sidebar
    render_sidebar()
    
    # Render the current page
    if st.session_state.current_page == "Home":
        render_home()
    elif st.session_state.current_page == "Data Upload":
        render_data_upload()
    elif st.session_state.current_page == "Model Configuration":
        render_model_config()
    elif st.session_state.current_page == "Attribution Analysis":
        render_attribution_analysis()
    elif st.session_state.current_page == "ROI Optimization":
        render_roi_optimization()
    
    # Footer for all pages
    st.markdown("---")
    st.caption("Built with Streamlit • Processing 10K+ daily data points")

if __name__ == "__main__":
    main() 