import pandas as pd

def calculate_frost_label(future_temp_window: pd.Series, threshold: float = 3.0) -> int:
    """
    Determines if a frost/cold-injury event occurs in the given future time window.
    
    Based on Omazić et al. (2024), a Tmin threshold of 3.0 °C (or 2.5 °C) best 
    describes frost formation in meteorological shelters. 
    
    Note: Dew point (Td) is intentionally excluded from this label to capture 
    'black frost' events, but must be included as a predictive feature.
    
    Args:
        future_temp_window (pd.Series): Continuous temperature values over the prediction horizon.
        threshold (float): Set to 3.0 °C based on literature for shelter-measured frost risk.
                           
    Returns:
        int: 1 if the minimum temperature in the window drops to or below the threshold, else 0.
    """
    if future_temp_window.min() <= threshold:
        return 1
    return 0