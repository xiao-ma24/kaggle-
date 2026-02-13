# kaggle-初次我用原始数据三个模型测试下来，决策树交叉验证最高得0.771740014753702，测试得0.800298899506362，随机森林交叉验证最高可得0.8454419961512567，测试可得0.862786268665411，xgboost交叉验证最高可得0.8481398522853851，测试可得0.76优化后xgboost可得0.87795531153678与0.8558，第二次我将房价对数化，使房价分布更加服从正太话，让机器学习不受极大房价影响，而后决策树交叉验证最高得0.7663779128686193,测试0.7607229009814064，随机森林0.8612665961844307和0.8793912487550684，xgboost0.8945463311226318，0.8949¶
# 房价预测模型分析
本次代码耗费我很长时间，我一开始学习了泰坦尼克号数据后，只会简单的填补缺失值，也不会处理这么多特征，这次建模我学会了：
数据分析的通用步骤：
1.环境准备，导入所有需要的库，pandas、numpy、sklearn等
2.基础数据探索，加载数据并简单浏览数据大概，看一看数据有什么类型等，read_csv
3.检查并处理问题数据，isna（）检查处理，重复值检查处理
一般缺失值很大的就删除了，其他的若是数字一般用median中位数填，若是object用mode众数填，机器学习常用的就是SimpleImputer，实例化num_inp=SimpleImpter(strategy='median'),num_inp=num_inp.fit(train_df),test_df=num_inp.transform(test_df),注意：为了保证数据不泄露，用训练集的中位数，众数来填补测试集的缺失值，因为我们假设测试集在生活中是完全接触不到的，就像闭卷考试一样。还有就是我们要提取出训练集里数字的列名和object的列名，方便后续使用
4.特征分析
画sns.hsitplot看是否是正态分布，不是则建议对数化，一般用np.log1p()表示log（x+1）最后预测后再np.expm1这样做可以减少极端分布对模型的影响，减小误差，还有一个重要的是可以看看价格相关的相关热力分析，用代码train_df.select_dtype(include=[np.number]).corr(),top_corr=corr['saleprice'].sort_values().head(10）,再画个图：sns.heatmap(data=corr.loc(top_corr.index,top_corr.index),annot=True,cmap='coolwarm'),其他的可以花些箱线图如sns.boxplot(data=train_df,x='Neighborhood',y='SalePrice')看一看特征
5.特征工程（!!!）
现在已经有数字列和object列了，对与object列我们采用独立热编码，也就是将一个特征分为多个列，列名为特征的可能值，为1则是特征，为0则不是，为了减少冗余，一般加参数drop_first,让其减少一列，代码为
train_df=pd.get_dummies(train_df,columns=object_columns,drop_first=True,dtype=int)
test_df=pd.get_dummies(test_df,columns=object_columns,drop_first=True,dtype=int)
如果可以的话，自己加几个特征或减少几个无关的特征
6.数据集的划分
首先处理训练集，y=np.log1(train_df['salePrice'])对数化那一步，x=train_df.drop(columns=['saleprice'])
让后就建模了，下面就你比较简单，建模，调参，预测了，老步骤：X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=24)
7.模型搭建与训练
实例化，按顺序先基础决策树再随机森林，xgboost
8.模型评估
test只用来score=clf.score(x_test,y_test),实际还是要看交叉验证的分数才准
9.模型优化
可以画图看，用gridsearchCV找，只是比较耗时间
决策树重要的参数是min-sample——leaf/split等，随机森林是n_estimators，xgboost是挺多
n_estimators=210,    # 增加树的数量（从100→200）
    max_depth=5,         # 降低树深度（从6/8→4）
    learning_rate=0.05,  # 降低学习率（从0.1→0.05）
    subsample=0.7,       # 每棵树随机用80%样本，减少过拟合
    colsample_bytree=0.8,# 每棵树随机用80%特征，减少过拟合
    random_state=42
10.预测&生成提交文件
rfr_prediction=rfr.predict（test_df）等，选好的模型预测
注意：最好的结果最好是融合一下，看分数0.4的随机森林，0.6的xgboost
还可以用
score=r2_score(y_test,real_predictions)
score看看分数
最后提交将
final_predictions=np.expm1(test_predictions_rf*0.4+test_predictions_xgb*0.6)
submission=pd.DataFrame({
    'Id':test_df['Id'],
    'SalePrice':final_predictions
})
submission.to_csv('submission.csv',index=False,encoding='utf-8')
结束，于2026年2月12日23：46在杨桥

