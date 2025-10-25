import numpy as np

def weighted_mean_std(values, weights):
    """가중 평균과 표준편차 계산"""
    weights = np.array(weights)
    weights /= np.sum(weights)
    mean = np.sum(values * weights)
    std = np.sqrt(np.sum(weights * (values - mean) ** 2))
    return mean, std