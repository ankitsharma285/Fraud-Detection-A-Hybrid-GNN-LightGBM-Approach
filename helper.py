import pandas as pd 

def create_mapping(series):
    unique_values = series.unique()
    return {val: i for i, val in enumerate(unique_values)}, len(unique_values)