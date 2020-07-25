import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
sns.set_context(context="poster",font_scale=1.2)
import matplotlib.pyplot as plt

def main():
    df = pd.DataFrame({"A": ["a0", "a1", "a1", "a2", "a3", "a4"], "B": ["b0", "b1", "b2", "b2", "b3", None],
                       "C": [1, 2, None, 3, 4, 5], "D": [0.1, 10.2, 11.4, 8.9, 9.1, 12], "E": [10, 19, 32, 25, 8, None],
                       "F": ["f0", "f1", "g2", "f3", "f4", "f5"]})
    df.isnull()
    df.dropna(subset=["B","C"])
    df.duplicated(["A"],keep="first")
    df.drop_duplicates(["A","B"],keep="first",inplace=False)
    df["B"].fillna("b*")
    df["E"].fillna(df["E"].mean())
    df["E"].interpolate(method="spline",order=3)
    pd.Series([1, None, 4, 10, 8]).interpolate()
    df[df['D'] < df["D"].quantile(0.75) + 1.5 * (df["D"].quantile(0.75) - df["D"].quantile(0.25))][df["D"] > df["D"].quantile(0.25) - 1.5 * (df["D"].quantile(0.75) - df["D"].quantile(0.25))]
    df[[True if item.startswith("f") else False for item in list(df["F"].values)]]
if __name__=="__main__":
    main()