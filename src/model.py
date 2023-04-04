from dataclasses import dataclass


@dataclass
class yoModel:
    """Abstract class for yofication model"""
    
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, data):
        pass