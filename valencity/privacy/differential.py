"""
Differential Privacy mechanisms.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd


class DifferentialPrivacy:
    """
    Utilities for adding differentially private noise to data.
    """
    
    @staticmethod
    def laplace_mechanism(
        value: Union[float, int, pd.Series, np.ndarray], 
        sensitivity: float, 
        epsilon: float,
        random_state: Optional[int] = None
    ) -> Union[float, pd.Series, np.ndarray]:
        """
        Add Laplace noise to value(s) to achieve epsilon-differential privacy.
        
        The Laplace mechanism adds noise scale = sensitivity / epsilon.
        
        Args:
            value: The true value(s) (numeric).
            sensitivity: The maximum amount the value can change if one individual 
                         is removed from the dataset.
            epsilon: The privacy budget. Smaller epsilon = more privacy (more noise).
            random_state: Seed for reproducibility.
            
        Returns:
            Noisy value(s).
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        scale = sensitivity / epsilon
        
        if isinstance(value, (pd.Series, np.ndarray, list)):
            noise = np.random.laplace(0, scale, size=len(value))
        else:
            noise = np.random.laplace(0, scale)
            
        return value + noise

    @staticmethod
    def gaussian_mechanism(
        value: Union[float, int, pd.Series, np.ndarray],
        sensitivity: float,
        epsilon: float,
        delta: float,
        random_state: Optional[int] = None
    ) -> Union[float, pd.Series, np.ndarray]:
        """
        Add Gaussian noise to achieve (epsilon, delta)-differential privacy.
        
        Sigma = sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
        
        Args:
            value: The true value(s).
            sensitivity: The L2 sensitivity.
            epsilon: Privacy budget.
            delta: Probability of privacy breach (usually << 1/N).
            random_state: Seed.
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        # Calculate sigma for Gaussian noise
        sigma = (np.sqrt(2 * np.log(1.25 / delta)) * sensitivity) / epsilon
        
        if isinstance(value, (pd.Series, np.ndarray, list)):
            noise = np.random.normal(0, sigma, size=len(value))
        else:
            noise = np.random.normal(0, sigma)
            
        return value + noise
