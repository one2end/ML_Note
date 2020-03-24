## 载入相关库
数据科学库:pandas、numpy、scipy
可视化库: matplotlib\seabon\missingno
分析库:pandas_profiling
```python
df = pd.read_csv(r'\OneDrive\桌面\used_car_train_20200313.csv',sep = ' ')
pandas_profiling.ProfileReport(df) 
```

##  观察数据
```
df.info()
df.shape
df.desctibe()
df.head().append(df.tail()) # get观察首尾
```
## 检查数据错误
1. 缺失值
```
df.isnull().sum()
import missingno as msno
msno.matrix(df, labels=True) #无效数据密度显示
msno.bar(df) # 条形图显示
msno.heatmap(df) #热图相关性显示
msno.dendrogram(df) #树状图显示
```
2. 异常值
主要通过info(), object类型查看
describe() 查看数值分布
```
df['notRepairedDamage'].value_counts()
df['notRepairedDamage'].replace('-', np.nan, inplace=True)
```
3. 一致数据(类型\内容)
多张表数据一致
数据类型一致
区分数字\类别特征
```
numeric_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14' ]

categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode',]
```
## 类别特征分布

特征nunique分布
```
for cat_fea in categorical_features:
 print(cat_fea + "的特征分布如下：")
 print("{}特征有个{}不同的值".format(cat_fea, df[cat_fea].nunique()))
 # print(df[cat_fea].value_counts())
```

## 数字特征分析

1. 总体分布概况（无界约翰逊分布等）
```
sns.图名(x='X轴 列名', y='Y轴 列名', data=原始数据df对象)
sns.图名(x='X轴 列名', y='Y轴 列名', hue='分组绘图参数', data=原始数据df对象)
**sns.图名(x=np.array, y=np.array[, ...])
```


1. 查看预测值的具体频数
```
plt.hist(df['price'], bins= 100, color ='red') # bins 分箱统计
plt.hist(np.log(df['price']),300,color ='red') # log变化统计
```
3. 相关性分析
```
price_numeric = df[numeric_features]
correlation = price_numeric.corr()
print(correlation['price'].sort_values(ascending = False),'\n')```
4. 数据的偏度和峰度——df.skew()、df.kurt()
5. 数字特征相互之间的关系可视化
```
sns.set()
columns = ['price', 'v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
sns.pairplot(df[columns],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()
```
1. 每个数字特征得分布可视化
```
f = pd.melt(df, value_vars=numeric_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
```
7. 多变量关系可视化


## 类别特征分析
