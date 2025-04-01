import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from typing import Dict, List, Tuple, Any

def train_attribution_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "Multivariate Regression",
    model_params: Dict[str, Any] = None
) -> Any:
    """
    Train an attribution model on the prepared data.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    model_type : str, optional
        Type of model to train
    model_params : Dict[str, Any], optional
        Model-specific parameters
        
    Returns:
    --------
    Any
        Trained model object
    """
    if model_params is None:
        model_params = {}
    
    if model_type == "Ridge Regression":
        alpha = model_params.get("alpha", 1.0)
        model = Ridge(alpha=alpha)
    elif model_type == "Bayesian Model":
        # In a real implementation, this would use a Bayesian model
        # For demo, we'll use a simple Ridge model
        model = Ridge(alpha=0.5)
    elif model_type == "Shapley Value Decomposition":
        # In a real implementation, this would use Shapley value calculations
        # For demo, we'll use a simple Linear model
        model = LinearRegression()
    else:  # Default to Multivariate Regression
        model = LinearRegression()
    
    # Fit the model
    model.fit(X, y)
    
    return model

def calculate_channel_contributions(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    y: np.ndarray
) -> Dict[str, float]:
    """
    Calculate channel contributions based on the trained model.
    
    Parameters:
    -----------
    model : Any
        Trained attribution model
    X : np.ndarray
        Feature matrix
    feature_names : List[str]
        Names of the features (channels)
    y : np.ndarray
        Target vector (actual outcomes)
        
    Returns:
    --------
    Dict[str, float]
        Dictionary mapping channels to their contribution values
    """
    # Get model coefficients
    coefficients = model.coef_
    
    # Calculate the contribution of each channel
    contributions = {}
    
    # Calculate individual channel contributions
    for i, feature in enumerate(feature_names):
        # For each channel, its contribution is coefficient * feature values
        channel_contrib = coefficients[i] * X[:, i]
        contributions[feature] = np.sum(channel_contrib)
    
    # Calculate base effect (intercept)
    if hasattr(model, 'intercept_'):
        base_effect = model.intercept_ * len(y)
        contributions['base_effect'] = base_effect
    
    # Ensure contributions are positive
    # In a real model, negative contributions would be handled more carefully
    for channel in contributions:
        if contributions[channel] < 0:
            contributions[channel] = 0
    
    # Normalize to match total actual outcomes
    total_contribution = sum(contributions.values())
    total_actual = np.sum(y)
    
    normalization_factor = total_actual / total_contribution if total_contribution > 0 else 1
    
    for channel in contributions:
        contributions[channel] *= normalization_factor
    
    return contributions

def calculate_roi_metrics(
    contributions: Dict[str, float],
    spend_by_channel: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate ROI metrics for each channel.
    
    Parameters:
    -----------
    contributions : Dict[str, float]
        Dictionary of channel contributions
    spend_by_channel : Dict[str, float]
        Dictionary of spend by channel
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of ROI metrics by channel
    """
    roi_metrics = {}
    
    for channel in spend_by_channel:
        spend = spend_by_channel.get(channel, 0)
        contribution = contributions.get(channel, 0)
        
        # Avoid division by zero
        if spend > 0:
            roi = contribution / spend
        else:
            roi = 0
        
        roi_metrics[channel] = roi
    
    return roi_metrics

def run_attribution_analysis(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    spend_by_channel: Dict[str, float],
    model_type: str = "Multivariate Regression",
    model_params: Dict[str, Any] = None
) -> Tuple[Dict[str, float], Dict[str, float], Any]:
    """
    Run a complete attribution analysis.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    feature_names : List[str]
        Names of the features (channels)
    spend_by_channel : Dict[str, float]
        Dictionary of spend by channel
    model_type : str, optional
        Type of model to train
    model_params : Dict[str, Any], optional
        Model-specific parameters
        
    Returns:
    --------
    Tuple[Dict[str, float], Dict[str, float], Any]
        Tuple containing:
        - Dictionary of channel contributions
        - Dictionary of ROI metrics
        - Trained model
    """
    # Train the model
    model = train_attribution_model(X, y, model_type, model_params)
    
    # Calculate channel contributions
    contributions = calculate_channel_contributions(model, X, feature_names, y)
    
    # Calculate ROI metrics
    roi_metrics = calculate_roi_metrics(contributions, spend_by_channel)
    
    return contributions, roi_metrics, model