import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
sns.set_context(context="poster",font_scale=1.2)
import matplotlib.pyplot as plt
def main():
    df=pd.read_csv("./data/HR.csv")
    #Left与部分属性的t独立分布检验
    dp_indices=df.groupby(by="department").indices
    sales_values=df["left"].iloc[dp_indices["sales"]].values
    print(sales_values)
    technical_values=df["left"].iloc[dp_indices["technical"]].values
    print(technical_values)
    print(ss.ttest_ind(sales_values,technical_values))
    dp_keys=dp_indices.keys()
    dp_t_mat=np.zeros((len(dp_keys),len(dp_keys)))
    for i in range(len(dp_keys)):
        for j in range(len(dp_keys)):
            p_value=ss.ttest_ind(df["left"].iloc[dp_indices[dp_keys[i]]].values,\
                                        df["left"].iloc[dp_indices[dp_keys[j]]].values)[1]
            if p_value<0.05:
                dp_t_mat[i][j]=-1
            else:
                dp_t_mat[i][j]=p_value
    sns.heatmap(dp_t_mat,xticklabels=dp_keys,yticklabels=dp_keys)
    plt.show()
    #交叉分析left
    piv_tb=pd.pivot_table(df, values="left", index=["department", "salary"], columns=["time_spend_company"],aggfunc=np.mean)
    piv_tb=pd.pivot_table(df, values="left",index=["department","salary"],columns=["number_project"],aggfunc=np.mean)
    piv_tb = pd.pivot_table(df, values="left", index=["promotion_last_5years", "salary"], columns=["Work_accident"],aggfunc=np.mean)
    print(piv_tb.index)
    sns.heatmap(piv_tb,vmax=1,vmin=0)
    plt.show()
    sns.barplot(x="salary",y="left",hue="department",data=df)
    plt.show()
    sl_s=df["satisfaction_level"]
    sns.barplot(range(len(sl_s)),sl_s.sort_values())
    #sns.barplot(range(len(sl_s)),sl_s.sort_values().diff().diff())
    plt.show()
if __name__=="__main__":
    main()