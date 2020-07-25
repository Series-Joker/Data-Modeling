import pandas as pd
import numpy as np

def main():
    import scipy.stats as ss
    print(ss.normaltest(ss.norm.rvs(size=10)))#正态检验
    print(ss.chi2_contingency([[15, 95], [85, 5]], False))#卡方四格表
    print(ss.ttest_ind(ss.norm.rvs(size=10), ss.norm.rvs(size=20)))#t独立分布检验
    print(ss.f_oneway([49, 50, 39,40,43], [28, 32, 30,26,34], [38,40,45,42,48]))#F分布检验
    from statsmodels.graphics.api import qqplot
    from matplotlib import pyplot as plt
    qqplot(ss.norm.rvs(size=100))#QQ图
    plt.show()

    s = pd.Series([0.1, 0.2, 1.1, 2.4, 1.3, 0.3, 0.5])
    df = pd.DataFrame([[0.1, 0.2, 1.1, 2.4, 1.3, 0.3, 0.5], [0.5, 0.4, 1.2, 2.5, 1.1, 0.7, 0.1]])
    #相关分析
    print(s.corr(pd.Series([0.5, 0.4, 1.2, 2.5, 1.1, 0.7, 0.1])))
    print(df.corr())

    import numpy as np
    #回归分析
    x = np.arange(10).astype(np.float).reshape((10, 1))
    y = x * 3 + 4 + np.random.random((10, 1))
    print(x)
    print(y)
    from sklearn.linear_model import LinearRegression
    linear_reg = LinearRegression()
    reg = linear_reg.fit(x, y)
    y_pred = reg.predict(x)
    print(reg.coef_)
    print(reg.intercept_)
    print(y.reshape(1, 10))
    print(y_pred.reshape(1, 10))
    plt.figure()
    plt.plot(x.reshape(1, 10)[0], y.reshape(1, 10)[0], "r*")
    plt.plot(x.reshape(1, 10)[0], y_pred.reshape(1, 10)[0])
    plt.show()

    #PCA降维
    df = pd.DataFrame(np.array([np.array([2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1]),
                                np.array([2.4, 0.7, 2.9, 2.2, 3, 2.7, 1.6, 1.1, 1.6, 0.9])]).T)
    from sklearn.decomposition import PCA
    lower_dim = PCA(n_components=1)
    lower_dim.fit(df.values)
    print("PCA")
    print(lower_dim.explained_variance_ratio_)
    print(lower_dim.explained_variance_)

from scipy import linalg
#一般线性PCA函数
def pca(data_mat, topNfeat=1000000):
    mean_vals = np.mean(data_mat, axis=0)
    mid_mat = data_mat - mean_vals
    cov_mat = np.cov(mid_mat, rowvar=False)
    eig_vals, eig_vects = linalg.eig(np.mat(cov_mat))
    eig_val_index = np.argsort(eig_vals)
    eig_val_index = eig_val_index[:-(topNfeat + 1):-1]
    eig_vects = eig_vects[:, eig_val_index]
    low_dim_mat = np.dot(mid_mat, eig_vects)
    # ret_mat = np.dot(low_dim_mat,eig_vects.T)
    return low_dim_mat, eig_vals


if __name__=="__main__":
    main()