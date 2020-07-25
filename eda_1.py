import pandas as pd
def main():
    df=pd.read_csv('./data/HR.csv')
    #查看数据梗概
    print(df)
    print(type(df))#类型
    print(type(df["satisfaction_level"]))
    print(df.mean())#平均值
    print(df["satisfaction_level"].mean())
    print(df.median())#中位数
    print(df["satisfaction_level"].median())
    print(df.quantile(q=0.25))#四分位数
    print(df["satisfaction_level"].quantile(q=0.25))
    print(df.mode())#众数
    print(df["department"].mode())
    print(type(df["department"].mode()))
    print(df.sum())#求和
    print(df["satisfaction_level"].sum())
    print(df.std())#标准
    print(df["satisfaction_level"].std())
    print(df.var())#方差
    print(df["satisfaction_level"].var())
    print(df.skew())#偏态
    print(df["satisfaction_level"].skew())
    print(df.kurt())#峰态
    print(df["satisfaction_level"].kurt())
    import scipy.stats as ss
    mean,var,skew,kurt = ss.norm.stats(moments="mvsk")
    print(mean, var, skew, kurt)#均值、方差、偏态、峰态
    print(ss.norm.pdf(0))#概率密度函数
    print(ss.norm.ppf(0.9))#分位值函数
    print(ss.norm.cdf(2))#累积分布函数
    print(ss.norm.cdf(2)-ss.norm.cdf(-2))
    print(ss.norm.rvs(size=100))#分布随机值生成
    print(ss.chi2)#卡方
    print(ss.t)#t分布
    print(ss.f)#F分布

    print(df.sample(100))#抽样
    print(df["satisfaction_level"].sample(frac=0.01))
    #df.mean()
if __name__=="__main__":
    main()