import pandas as pd
import numpy as np
from datetime import datetime

def preprocess_data(data):
    """
    Preprocess the uploaded marketing data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw marketing data
        
    Returns:
    --------
    pandas.DataFrame
        Processed data ready for analysis
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Convert date column to datetime if it exists
    date_cols = [col for col in df.columns if col.lower() in ['date', 'day', 'week', 'month']]
    if date_cols:
        date_col = date_cols[0]
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            # If conversion fails, keep as is
            pass
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        # Fill missing values with appropriate methods
        # For numeric columns, use median
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # For categorical columns, use most frequent value
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    # Sort by date if date column exists
    if date_cols:
        df = df.sort_values(by=date_col)
    
    return df

def identify_channels(data):
    """
    Automatically identify marketing channel columns in the data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Marketing data
        
    Returns:
    --------
    list
        List of identified channel column names
    """
    # Get all column names
    all_columns = data.columns
    
    # Define common identifiers for marketing channels
    channel_identifiers = ['channel', 'spend', 'cost', 'budget', 'campaign', 'advertising', 'marketing', 'media']
    exclude_columns = ['date', 'day', 'month', 'year', 'week', 'conversions', 'revenue', 'sales', 'target', 'outcome', 'goal']
    
    # First look for columns that explicitly have 'channel' in their name
    explicit_channels = [col for col in all_columns if any(ci in col.lower() for ci in channel_identifiers)]
    
    # If none found, use heuristic: look for numeric columns that aren't dates or outcomes
    if not explicit_channels:
        numeric_cols = data.select_dtypes(include=np.number).columns
        potential_channels = [col for col in numeric_cols if not any(ex in col.lower() for ex in exclude_columns)]
        
        # If there are still too many columns, use columns that start with uppercase or have CamelCase
        if len(potential_channels) > 10:
            # More sophisticated heuristic - channels often have proper names
            # like Facebook, Google, etc. which start with uppercase
            channels = [col for col in potential_channels if col[0].isupper() or '_' in col or any(c.isupper() for c in col[1:])]
            
            # If still none found, use all potential channels
            if not channels:
                channels = potential_channels
        else:
            channels = potential_channels
    else:
        channels = explicit_channels
    
    return channels

def apply_adstock(data, column, decay_rate):
    """
    Apply adstock transformation to a marketing channel.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the marketing data
    column : str
        Name of the column to transform
    decay_rate : float
        Adstock decay rate (between 0 and 1)
        
    Returns:
    --------
    numpy.ndarray
        Transformed values with adstock effect
    """
    # Get the original values
    values = data[column].values
    
    # Initialize result array
    result = np.zeros_like(values, dtype=float)
    
    # Apply adstock transformation
    for i in range(len(values)):
        if i == 0:
            result[i] = values[i]
        else:
            result[i] = values[i] + result[i-1] * decay_rate
    
    return result

def apply_saturation(values, k):
    """
    Apply saturation transformation to a marketing channel.
    
    Parameters:
    -----------
    values : numpy.ndarray
        Array of values to transform
    k : float
        Saturation parameter (lower values mean earlier saturation)
        
    Returns:
    --------
    numpy.ndarray
        Transformed values with saturation effect
    """
    # Avoid division by zero
    eps = 1e-10
    
    # Apply Hill function for diminishing returns
    # s / (s + 1/k) is a common Hill function for saturation
    transformed = values / (values + (1/k) + eps)
    
    return transformed

def prepare_data_for_modeling(data, channels, outcome_column, adstock_params=None, saturation_params=None):
    """
    Prepare data for modeling by applying transformations.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Marketing data
    channels : list
        List of channel column names
    outcome_column : str
        Name of the outcome column
    adstock_params : dict, optional
        Dictionary of adstock parameters for each channel
    saturation_params : dict, optional
        Dictionary of saturation parameters for each channel
        
    Returns:
    --------
    tuple
        (X, y) where X is the feature matrix and y is the target vector
    """
    df = data.copy()
    
    # Apply transformations if parameters are provided
    if adstock_params:
        for channel, decay_rate in adstock_params.items():
            if channel in df.columns:
                df[f"{channel}_adstock"] = apply_adstock(df, channel, decay_rate)
    
    # Determine which columns to apply saturation to
    if saturation_params:
        cols_to_transform = [f"{c}_adstock" if adstock_params and c in adstock_params else c 
                           for c in channels]
        
        for i, channel in enumerate(channels):
            if channel in saturation_params:
                col_to_transform = cols_to_transform[i]
                if col_to_transform in df.columns:
                    k = saturation_params[channel]
                    df[f"{channel}_sat"] = apply_saturation(df[col_to_transform].values, k)
    
    # Determine final feature columns
    if adstock_params and saturation_params:
        X_cols = [f"{c}_sat" for c in channels]
    elif adstock_params:
        X_cols = [f"{c}_adstock" for c in channels]
    elif saturation_params:
        X_cols = [f"{c}_sat" for c in channels]
    else:
        X_cols = channels
    
    # Filter to only include columns that exist
    X_cols = [col for col in X_cols if col in df.columns]
    
    # Prepare X and y
    X = df[X_cols].values
    y = df[outcome_column].values
    
    return X, y, X_cols