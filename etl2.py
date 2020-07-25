import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
sns.set_context(context="poster",font_scale=1.2)
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest,RFE,SelectFromModel
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
def main():
    #特征选择
    df=pd.DataFrame({"A":ss.norm.rvs(size=10),"B":ss.norm.rvs(size=10),\
                     "C":ss.norm.rvs(size=10),"D":np.random.randint(low=0,high=2,size=10)})
    X=df.loc[:,["A","B","C"]]
    Y=df.loc[:,"D"]
    print("X",X)
    print("Y",Y)
    skb=SelectKBest(k=2)
    skb.fit(X.values,Y.values)
    print(skb.transform(X.values))

    rfe=RFE(estimator=SVR(kernel="linear"),n_features_to_select=2,step=1)
    print(rfe.fit_transform(X,Y))

    sfm=SelectFromModel(estimator=DecisionTreeRegressor(),threshold=0.01)
    print(sfm.fit_transform(X,Y))
if __name__=="__main__":
    main()