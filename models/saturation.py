import numpy as np
from typing import List, Union, Dict, Optional

def apply_saturation_transformation(
    values: Union[List[float], np.ndarray],
    k: float = 0.1,
    method: str = 'hill'
) -> np.ndarray:
    """
    Apply saturation transformation to model diminishing returns.
    
    Saturation transformations model how the effectiveness of marketing spend
    diminishes as spend increases.
    
    Parameters:
    -----------
    values : Union[List[float], np.ndarray]
        Original marketing spend or activity values
    k : float, optional
        Saturation parameter (controls how quickly diminishing returns set in)
        Lower values mean earlier saturation/stronger diminishing returns
    method : str, optional
        Saturation function to use:
        - 'hill' (default): Hill function s/(s+k) - common in marketing science
        - 'log': Logarithmic transformation
        - 'power': Power transformation
        
    Returns:
    --------
    np.ndarray
        Transformed values after applying saturation effect
    
    Examples:
    ---------
    >>> apply_saturation_transformation([100, 200, 300, 400, 500], k=0.1)
    array([0.90909091, 0.95238095, 0.96774194, 0.97560976, 0.98039216])
    
    >>> apply_saturation_transformation([100, 200, 300, 400, 500], method='log', k=0.1)
    array([0.4605, 0.5298, 0.5704, 0.5991, 0.6215])
    """
    # Convert to numpy array and ensure positive values
    values_array = np.maximum(np.array(values, dtype=float), 1e-10)
    
    # Apply appropriate saturation transformation
    if method == 'hill':
        # Hill function (s / (s + 1/k)) - classic diminishing returns formula
        # Avoid division by zero with small epsilon
        eps = 1e-10
        transformed = values_array / (values_array + (1/k) + eps)
    
    elif method == 'log':
        # Logarithmic transformation
        # log(1 + k*s) / (1 + log(1 + k*s))
        log_term = np.log1p(k * values_array)
        transformed = log_term / (1 + log_term)
    
    elif method == 'power':
        # Power transformation
        # s^(1-k)
        transformed = np.power(values_array, 1 - k)
        # Normalize to [0, 1] range
        if np.max(transformed) > 0:
            transformed = transformed / np.max(transformed)
    
    else:
        raise ValueError(f"Unknown saturation method: {method}")
    
    return transformed

def apply_s_curve_transformation(
    values: Union[List[float], np.ndarray],
    k: float = 0.1,
    L: float = 1.0,
    x0: Optional[float] = None
) -> np.ndarray:
    """
    Apply S-curve (logistic) transformation to model complex response curves.
    
    S-curves model scenarios where marketing effect starts slowly, then accelerates,
    and finally plateaus (diminishing returns).
    
    Parameters:
    -----------
    values : Union[List[float], np.ndarray]
        Original marketing spend or activity values
    k : float, optional
        Steepness parameter (how sharp the S-curve is)
    L : float, optional
        Maximum value (upper asymptote)
    x0 : float, optional
        Midpoint of the curve (where growth is fastest)
        If None, automatically calculated as the median of non-zero values
        
    Returns:
    --------
    np.ndarray
        Transformed values after applying S-curve
    """
    # Convert to numpy array
    values_array = np.array(values, dtype=float)
    
    # If x0 not provided, use median of non-zero values
    if x0 is None:
        non_zero = values_array[values_array > 0]
        x0 = np.median(non_zero) if len(non_zero) > 0 else np.median(values_array)
    
    # Apply logistic function
    # L / (1 + exp(-k * (x - x0)))
    transformed = L / (1 + np.exp(-k * (values_array - x0)))
    
    return transformed

def scale_to_original_magnitude(
    original: Union[List[float], np.ndarray],
    transformed: np.ndarray
) -> np.ndarray:
    """
    Scale the transformed values to maintain the original magnitude.
    
    This is useful when you want to preserve the overall scale of the data
    while applying the saturation effects.
    
    Parameters:
    -----------
    original : Union[List[float], np.ndarray]
        Original values before transformation
    transformed : np.ndarray
        Values after saturation transformation (typically in [0,1] range)
        
    Returns:
    --------
    np.ndarray
        Scaled transformed values with similar magnitude to original
    """
    original_array = np.array(original, dtype=float)
    
    # Calculate scaling factor to match the sum of original values
    sum_original = np.sum(original_array)
    sum_transformed = np.sum(transformed)
    
    if sum_transformed > 0:
        scaling_factor = sum_original / sum_transformed
        scaled = transformed * scaling_factor
    else:
        scaled = np.zeros_like(transformed)
    
    return scaled

def apply_saturation_batch(
    data: Dict[str, Union[List[float], np.ndarray]],
    params: Dict[str, Dict[str, float]],
    maintain_magnitude: bool = True
) -> Dict[str, np.ndarray]:
    """
    Apply saturation transformations to multiple channels with different parameters.
    
    Parameters:
    -----------
    data : Dict[str, Union[List[float], np.ndarray]]
        Dictionary mapping channel names to their original values
    params : Dict[str, Dict[str, float]]
        Dictionary mapping channel names to their saturation parameters
        Each channel's parameters should include 'k' and optionally 'method'
    maintain_magnitude : bool, optional
        Whether to scale the transformed values to maintain the original magnitude
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary mapping channel names to their transformed values
    """
    result = {}
    
    for channel, values in data.items():
        if channel in params:
            channel_params = params[channel]
            k = channel_params.get('k', 0.1)
            
            # Determine the saturation method
            sat_method = channel_params.get('method', 'hill')
            
            if sat_method == 's_curve':
                L = channel_params.get('L', 1.0)
                x0 = channel_params.get('x0', None)
                transformed = apply_s_curve_transformation(values, k, L, x0)
            else:
                transformed = apply_saturation_transformation(values, k, sat_method)
            
            # Scale to maintain magnitude if requested
            if maintain_magnitude:
                result[channel] = scale_to_original_magnitude(values, transformed)
            else:
                result[channel] = transformed
        else:
            # If no parameters specified, use default transformation
            transformed = apply_saturation_transformation(values)
            
            if maintain_magnitude:
                result[channel] = scale_to_original_magnitude(values, transformed)
            else:
                result[channel] = transformed
    
    return result