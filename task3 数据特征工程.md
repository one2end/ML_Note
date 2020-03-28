
规范化
最小-最大规范化
$$x ^ { \prime } = \frac { x - min } { max - min }$$
```python
(df-df.min())/(df.max()-df.min())
OR
from sklearn import preprocessing
preprocessing.minmax_scale(df)
```
问题： 
• 若将来的数字超过min和max需重新定义 
• 若某个数很大则规范化后值相近且均接近0
z-score规范化 
$$x ^ { \prime } = \frac { x - \bar { x } } { \sigma }$$
```python
(df-df.mean())/df.std() 
OR
scaler = preprocessing.scale(df) 
```
特征： 
• 使用最多 
• 处理后数据的均值为0，标准差为1
小数定标规范化
$$x ^ { \prime } = \frac { x } { 10 ^ { \mathrm { j } } }$$
```python
df/10**np.ceil(np.log10(df.abs().max())) 
```
特征： 
• 移动小数点位置，移动位数取决于属性绝对值的最大值 
• 常见落在[-1, 1]之间
连续属性离散化
方法
分箱（binning）
等宽法
按照实际值的大小分箱,  每箱数可能不同.
```python
pd.cut(df.AGE, 5, labels = range(5))  # 返回区间, 加标签返回标签
```python
等频法 
按照实际数量分频,每箱数 = 总数/箱数
```python
pd.qcut(df.AGE, 5, labels = range(5))
```
聚类
特征二值化binarization 
多元问题转换为二元问题
案例
电影评分转换为推荐or不推荐
代码
```python
>>> from sklearn.preprocessing import Binarizer 
>>> X = boston.target.reshape(-1,1)  #转换为506行
>>> Binarizer(threshold = 20.0).fit_transform(X)   # 以20 为区分
```

数据规约 Data reduction 
对属性和数值进行规约获 得一个比原数据集的小的多的规约表示，但仍接近原数据的完整性，在规约后数据集上挖掘可产生近乎相同的分析结果
属性规约
向前选择
向后删除
决策树 
PCA
```python
>>> fromsklearn.decomposition import PCA 
>>> X = preprocessing.scale(boston.data) 
>>> pca = PCA(n_components=5) 
>>> pca.fit(X) 
>>> pca.explained_variance_ratio_ 
array([0.47129606, 0.11025193, 0.0955859 , 0.06596732, 0.06421661])
```
数值规约
有参方法（回归法，对数线性模型）
无参法（直方图，聚类，抽样）
直方图
表现： 
• 用分箱表示数据分布 
• 每个箱子代表一个属性-频率对
```python
data = np.random.randint(1,10,50)
plt.hist(data, bins=…)
```
抽样
特征列举
• 不放回随机抽样：从原始数据集D的N个样本中抽取n个样本，每次抽到不同的数据 
• 放回随机抽样：从原始数据集D的N个样本中抽取n个样本，抽取后记录它后放回，有可能抽到同样的数据 
• 分层抽样：数据集D为划分成互不相交的部分即层，对每一层进行简单随机抽样获得最终结果
随机抽样
```python
不放回： 
iris_df.sample(n = 10) 
iris_df.sample(frac = 0.3) 
有放回： 
iris_df.sample(n = 10, replace = True) 
iris_df.sample(frac = 0.3, replace = True)
```
分层抽样 
```python
>>> A = iris_df[iris_df.target == 0].sample(frac = 0.3) 
>>> B = iris_df[iris_df.target == 1].sample(frac = 0.2) 
>>> A.append(B) 
```

