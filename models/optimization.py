import numpy as np
from typing import Dict, List, Tuple, Any

def simulate_optimized_budget(
    current_spend: Dict[str, float],
    roi_metrics: Dict[str, float],
    contributions: Dict[str, float],
    target_budget: float,
    objective: str = "Balanced Approach",
    min_channel_budget: float = 10,
    max_channel_budget: float = 70
) -> Tuple[Dict[str, float], float, List[Dict[str, Any]]]:
    """
    Simulate an optimized budget allocation based on ROI metrics and channel contributions.
    
    Parameters:
    -----------
    current_spend : Dict[str, float]
        Dictionary of current spend by channel
    roi_metrics : Dict[str, float]
        Dictionary of ROI metrics by channel
    contributions : Dict[str, float]
        Dictionary of attribution contributions by channel
    target_budget : float
        Target total budget
    objective : str, optional
        Optimization objective: "Maximize Revenue", "Maximize ROI", or "Balanced Approach"
    min_channel_budget : float, optional
        Minimum channel budget as percentage of current spend
    max_channel_budget : float, optional
        Maximum channel budget as percentage of current spend
        
    Returns:
    --------
    Tuple[Dict[str, float], float, List[Dict[str, Any]]]
        Tuple containing:
        - Dictionary of optimized spend by channel
        - Expected outcome after optimization
        - List of comparison data for each channel
    """
    # Simple optimization algorithm biased toward higher ROI channels
    total_spend = sum(current_spend.values())
    optimized_spend = {}
    
    # Simplified optimization that favors higher ROI channels
    if objective == "Maximize ROI":
        # More weight to ROI
        weights = {channel: roi**2 for channel, roi in roi_metrics.items()}
    elif objective == "Maximize Revenue":
        # More weight to contribution
        weights = {channel: contributions.get(channel, 0)**2 * roi 
                  for channel, roi in roi_metrics.items()}
    else:  # Balanced
        # Balance between ROI and contribution
        weights = {channel: contributions.get(channel, 0) * roi**1.5 
                  for channel, roi in roi_metrics.items()}
    
    weight_sum = sum(weights.values())
    
    for channel, weight in weights.items():
        # Initial allocation based on weight
        alloc = target_budget * weight / weight_sum
        
        # Apply min/max constraints
        min_alloc = current_spend.get(channel, 0) * min_channel_budget / 100
        max_alloc = current_spend.get(channel, 0) * max_channel_budget / 100
        
        # Ensure we're not dividing by zero
        if current_spend.get(channel, 0) == 0:
            min_alloc = target_budget * 0.05  # 5% minimum
            max_alloc = target_budget * 0.2   # 20% maximum
        
        optimized_spend[channel] = max(min_alloc, min(alloc, max_alloc))
    
    # Adjust to meet target budget exactly
    total_optimized = sum(optimized_spend.values())
    adjustment_factor = target_budget / total_optimized
    
    for channel in optimized_spend:
        optimized_spend[channel] *= adjustment_factor
    
    # Create comparison data
    comparison_data = []
    for channel in current_spend.keys():
        current = current_spend.get(channel, 0)
        optimized = optimized_spend.get(channel, 0)
        change_pct = (optimized - current) / current * 100 if current > 0 else 100
        
        comparison_data.append({
            "Channel": channel,
            "Current Spend": current,
            "Optimized Spend": optimized,
            "Change %": change_pct,
            "ROI": roi_metrics.get(channel, 0)
        })
    
    # Sort by change percentage
    comparison_data.sort(key=lambda x: x["Change %"], reverse=True)
    
    # Calculate expected outcome improvement
    current_outcome = sum(contributions.values())
    expected_improvement = 0
    
    for channel in current_spend.keys():
        current = current_spend.get(channel, 0)
        optimized = optimized_spend.get(channel, 0)
        roi = roi_metrics.get(channel, 0)
        
        # Simple model: if we spend more, we get more, but with diminishing returns
        if optimized > current:
            # Saturation formula for increased spend
            improvement = (optimized - current) * roi * 0.8
        else:
            # If we reduce spend, we lose proportionally to contribution
            reduction_factor = optimized / current if current > 0 else 0
            improvement = contributions.get(channel, 0) * (reduction_factor - 1)
        
        expected_improvement += improvement
    
    expected_outcome = current_outcome + expected_improvement
    
    return optimized_spend, expected_outcome, comparison_data

def optimize_budget_with_diminishing_returns(
    contributions: Dict[str, float],
    roi_metrics: Dict[str, float],
    current_spend: Dict[str, float],
    target_budget: float,
    saturation_params: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Optimize budget allocation considering diminishing returns.
    
    Parameters:
    -----------
    contributions : Dict[str, float]
        Dictionary of attribution contributions by channel
    roi_metrics : Dict[str, float]
        Dictionary of ROI metrics by channel
    current_spend : Dict[str, float]
        Dictionary of current spend by channel
    target_budget : float
        Target total budget
    saturation_params : Dict[str, float], optional
        Dictionary of saturation parameters by channel
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of optimized spend by channel
    """
    # Default saturation parameters if not provided
    if saturation_params is None:
        saturation_params = {channel: 0.1 for channel in contributions.keys()}
    
    # Simple hill-climbing algorithm
    channels = list(contributions.keys())
    optimized_spend = {channel: current_spend.get(channel, 0) for channel in channels}
    
    # Calculate initial total spend
    total_spend = sum(optimized_spend.values())
    
    # Adjust to match target budget
    if total_spend > 0:
        adjustment_factor = target_budget / total_spend
        for channel in optimized_spend:
            optimized_spend[channel] *= adjustment_factor
    
    # Initialize with equal distribution if no current spend
    else:
        for channel in optimized_spend:
            optimized_spend[channel] = target_budget / len(optimized_spend)
    
    # Run optimization iterations
    for _ in range(100):  # Limited iterations for simulation
        improved = False
        
        # Try to improve by moving budget between channels
        for i, channel1 in enumerate(channels):
            for channel2 in channels[i+1:]:
                # Try moving budget from channel1 to channel2
                delta = optimized_spend[channel1] * 0.05  # Move 5% at a time
                
                # Calculate current expected returns
                current_return1 = calc_channel_return(
                    optimized_spend[channel1], 
                    roi_metrics.get(channel1, 1), 
                    saturation_params.get(channel1, 0.1)
                )
                current_return2 = calc_channel_return(
                    optimized_spend[channel2], 
                    roi_metrics.get(channel2, 1), 
                    saturation_params.get(channel2, 0.1)
                )
                current_total = current_return1 + current_return2
                
                # Calculate returns after moving budget
                new_return1 = calc_channel_return(
                    optimized_spend[channel1] - delta, 
                    roi_metrics.get(channel1, 1), 
                    saturation_params.get(channel1, 0.1)
                )
                new_return2 = calc_channel_return(
                    optimized_spend[channel2] + delta, 
                    roi_metrics.get(channel2, 1), 
                    saturation_params.get(channel2, 0.1)
                )
                new_total = new_return1 + new_return2
                
                # If improvement, apply the change
                if new_total > current_total:
                    optimized_spend[channel1] -= delta
                    optimized_spend[channel2] += delta
                    improved = True
                
                # Also try the reverse direction
                delta = optimized_spend[channel2] * 0.05
                
                new_return1 = calc_channel_return(
                    optimized_spend[channel1] + delta, 
                    roi_metrics.get(channel1, 1), 
                    saturation_params.get(channel1, 0.1)
                )
                new_return2 = calc_channel_return(
                    optimized_spend[channel2] - delta, 
                    roi_metrics.get(channel2, 1), 
                    saturation_params.get(channel2, 0.1)
                )
                new_total = new_return1 + new_return2
                
                if new_total > current_total:
                    optimized_spend[channel1] += delta
                    optimized_spend[channel2] -= delta
                    improved = True
        
        # Stop if no improvement in this iteration
        if not improved:
            break
    
    # Final adjustment to exactly match target budget
    total_spend = sum(optimized_spend.values())
    adjustment_factor = target_budget / total_spend
    
    for channel in optimized_spend:
        optimized_spend[channel] *= adjustment_factor
    
    return optimized_spend

def calc_channel_return(spend, roi, saturation):
    """
    Calculate expected return with diminishing returns.
    
    Parameters:
    -----------
    spend : float
        Amount spent on the channel
    roi : float
        ROI metric for the channel
    saturation : float
        Saturation parameter for the channel
        
    Returns:
    --------
    float
        Expected return
    """
    # Avoid division by zero
    eps = 1e-10
    
    # Use Hill function for diminishing returns
    # (spend / (spend + 1/saturation)) * roi * spend
    return (spend / (spend + (1/saturation) + eps)) * roi * spend