"""Shared feature engineering for WellbeingIQ (used by the notebooks and the dashboard)."""
import numpy as np, pandas as pd

def add_features(X):
    X = X.copy()
    u, s, l = X["Avg_Daily_Usage_Hours"], X["Sleep_Hours_Per_Night"], X["Late_Night_Usage_Hours"]
    X["Sleep_Deficit"] = (7.5 - s).clip(lower=0)
    X["Late_Night_Share"] = l / u.clip(lower=0.5)
    X["Usage_Sleep_Ratio"] = u / s
    flag = ((u >= 6) & (s < 6.5)).astype(float)
    X["HighUsage_LowSleep"] = flag.where(u.notna() & s.notna(), np.nan)
    return X
