import numpy as np
from typing import List, Union, Dict

def apply_adstock_transformation(
    values: Union[List[float], np.ndarray],
    decay_rate: float = 0.5,
    lag: int = 0
) -> np.ndarray:
    """
    Apply adstock transformation to model time-lagged effects of marketing.
    
    The adstock transformation models how marketing effects persist over time, with
    diminishing impact as time passes from the initial marketing activity.
    
    Parameters:
    -----------
    values : Union[List[float], np.ndarray]
        Original marketing spend or activity values
    decay_rate : float, optional
        Rate at which the effect decays over time (0 to 1)
        Higher values mean longer-lasting effects
    lag : int, optional
        Number of periods to lag the effect
        
    Returns:
    --------
    np.ndarray
        Transformed values after applying adstock effect
    
    Examples:
    ---------
    >>> apply_adstock_transformation([100, 0, 0, 0, 0], decay_rate=0.5)
    array([100., 50., 25., 12.5, 6.25])
    
    >>> apply_adstock_transformation([100, 200, 100, 0, 0], decay_rate=0.3)
    array([100., 230., 169., 50.7, 15.21])
    """
    # Convert to numpy array if needed
    values_array = np.array(values, dtype=float)
    
    # Apply lag if specified
    if lag > 0:
        values_array = np.pad(values_array, (lag, 0), 'constant')[:-lag]
    
    # Initialize result array
    result = np.zeros_like(values_array)
    
    # Apply adstock transformation
    for i in range(len(values_array)):
        if i == 0:
            result[i] = values_array[i]
        else:
            result[i] = values_array[i] + result[i-1] * decay_rate
    
    return result

def apply_geometric_adstock(
    values: Union[List[float], np.ndarray],
    decay_rate: float = 0.5,
    lag: int = 0
) -> np.ndarray:
    """
    Apply geometric adstock transformation.
    
    This is an alternative method where the effect of a marketing activity
    is distributed over future periods according to a geometric series.
    
    Parameters:
    -----------
    values : Union[List[float], np.ndarray]
        Original marketing spend or activity values
    decay_rate : float, optional
        Rate at which the effect decays over time (0 to 1)
    lag : int, optional
        Number of periods to lag the effect
        
    Returns:
    --------
    np.ndarray
        Transformed values after applying geometric adstock
    """
    # Convert to numpy array if needed
    values_array = np.array(values, dtype=float)
    n = len(values_array)
    
    # Initialize result array
    result = np.zeros(n)
    
    # Generate weight vector for the geometric decay
    max_periods = n
    weights = np.array([decay_rate ** i for i in range(max_periods)])
    
    # Apply convolution with lag
    for i in range(n):
        for j in range(min(i + 1, max_periods)):
            if i - j >= lag:
                result[i] += values_array[i - j] * weights[j]
    
    return result

def apply_delayed_adstock(
    values: Union[List[float], np.ndarray],
    decay_rate: float = 0.5,
    peak_delay: int = 2
) -> np.ndarray:
    """
    Apply adstock with peak delay transformation.
    
    Some marketing channels (like TV) have a delayed peak effect. This transformation
    models the effect building up to a peak after some delay and then decaying.
    
    Parameters:
    -----------
    values : Union[List[float], np.ndarray]
        Original marketing spend or activity values
    decay_rate : float, optional
        Rate at which the effect decays over time (0 to 1)
    peak_delay : int, optional
        Number of periods until the effect reaches its peak
        
    Returns:
    --------
    np.ndarray
        Transformed values after applying delayed adstock
    """
    # Convert to numpy array if needed
    values_array = np.array(values, dtype=float)
    n = len(values_array)
    
    # Initialize result array
    result = np.zeros(n)
    
    # Generate weight vector with delayed peak
    max_periods = min(n, peak_delay + 20)  # Add buffer beyond peak
    
    # Build-up phase
    buildup = np.array([i/peak_delay for i in range(1, peak_delay + 1)])
    
    # Decay phase
    decay = np.array([decay_rate ** i for i in range(max_periods - peak_delay)])
    
    # Combine to create weight vector with peak at peak_delay
    weights = np.concatenate([buildup, decay])
    weights = weights[:max_periods]
    
    # Normalize weights to ensure the total effect is preserved
    weights = weights / np.sum(weights)
    
    # Apply convolution
    for i in range(n):
        for j in range(min(i + 1, max_periods)):
            result[i] += values_array[i - j] * weights[j] if i - j >= 0 else 0
    
    return result

def apply_adstock_batch(
    data: Dict[str, Union[List[float], np.ndarray]],
    params: Dict[str, Dict[str, float]]
) -> Dict[str, np.ndarray]:
    """
    Apply adstock transformations to multiple channels with different parameters.
    
    Parameters:
    -----------
    data : Dict[str, Union[List[float], np.ndarray]]
        Dictionary mapping channel names to their original values
    params : Dict[str, Dict[str, float]]
        Dictionary mapping channel names to their adstock parameters
        Each channel's parameters should include 'decay_rate' and optionally 'lag'
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary mapping channel names to their transformed values
    """
    result = {}
    
    for channel, values in data.items():
        if channel in params:
            channel_params = params[channel]
            decay_rate = channel_params.get('decay_rate', 0.5)
            lag = channel_params.get('lag', 0)
            
            # Determine the adstock type
            adstock_type = channel_params.get('type', 'standard')
            
            if adstock_type == 'geometric':
                result[channel] = apply_geometric_adstock(values, decay_rate, lag)
            elif adstock_type == 'delayed':
                peak_delay = channel_params.get('peak_delay', 2)
                result[channel] = apply_delayed_adstock(values, decay_rate, peak_delay)
            else:  # standard
                result[channel] = apply_adstock_transformation(values, decay_rate, lag)
        else:
            # If no parameters specified, use default adstock
            result[channel] = apply_adstock_transformation(values)
    
    return result