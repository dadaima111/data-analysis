# Python数据分析

## 1.jupyter notebook

```python
# 安装jupyter notebook
pip install notebook		# 终端输入

# 打开jupyter notebook
jupyter notebook			# 终端输入

# 关闭jupyter notebook
ctrl + c					# 终端输入
```

## 2.Markdown

~~~markdown
标题：
# 标题
## 二级标题
### 三级标题

粗体：**粗体**

斜体：*斜体*

删除：~~删除~~

无序列表：- 我是列表第一行
		- 我是列表第二行
		- 我是列表第三行
		
有序列表：1. 我是列表第一项
		2. 我是列表第二项
		3. 我是列表第三项
		
链接：[我的主页地址](https://space.bilibili.com/439094393) 

插入图片：![城市景观](https://cn.bing.com/images/search?view=detailV2&ccid=AGMK9AaZ&id=880442E8923C6511DA48FA38772BD65947F46B73&thid=OIP.AGMK9AaZ62jHL_EVHKIowgHaF5&mediaurl=https%3a%2f%2fimg.zcool.cn%2fcommunity%2f01b6e45e181144a801216518f0797b.jpg%403000w_1l_2o_100sh.jpg&exph=2387&expw=3000&q=%e5%9b%be%e7%89%87&simid=608055640607257765&FORM=IRPRST&ck=2EF6EFA881BE0801466DC343ED81166F&selectedIndex=63&itb=0&ajaxhist=0&ajaxserp=0)

插入代码：我们通过`import math`来引入math库

插入代码段落：
```python
import math
print("hello world")
print(math.pi)
```

数学公式：
$y = x + 3$
$$y = x + 3$$
~~~

## 3.LaTeX

```latex
$$x+y$$
$$x-y$$
$$x\times y$$
$$x\div y$$
$$x^3$$
$$H_2O$$
$$S_{input}$$
$$\sum(x^2+y^2)$$
$$\sqrt[3]x$$
$$\sqrt[3]{a^2m^2}$$
$$\frac{x+y}{x-y}$$
```

![image-20240623092433651](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240623092433651.png)

## 4.numpy

### 4.1 数组

```python
import numpy as np
# 创建一维数组

arr1 = np.array([1,2,3])
arr1
>>>array([1, 2, 3])

# 创建二维数组
arr2 = np.array([[1,2,3],[4,5,6]])	# 最左边几个方括号就是几维数组
arr2
>>>array([[1, 2, 3],
       [4, 5, 6]])

# 返回数组的维度
print(arr1.ndim)
print(arr2.ndim)
>>>1
   2

# 返回一个元组，表示各个维度元素的个数
print(arr1.shape)
print(arr2.shape)
>>>(3,)
   (2, 3)

# 返回数组元素的总个数
print(arr1.size)
print(arr2.size)
>>>3
   6

# 返回数组元素的类型
print(arr1.dtype)
print(arr2.dtype)
>>>int32
   int32

# 创建全部都为0的数组
np.zeros(5)
>>>array([0., 0., 0., 0., 0.])

# 创建全部都为1的数组
np.ones(3)
>>>array([1., 1., 1.])

# 创建元素为数字序列的数组
np.arange(5,10,2)
>>>array([5, 7, 9])
```

### 4.2 数组进阶

```python
# 连接数组
arr1 = np.array([5,17,3,26,31])
arr2 = np.zeros(2)
np.concatenate([arr1, arr2])
>>>array([ 5., 17.,  3., 26., 31.,  0.,  0.])

# 最大值
arr4.max()

# 最小值
arr4.min()

# 和
arr4.sum()

# 平均值
arr4.mean()

# 根据条件筛选数组元素
arr = np.array([-22,3,65.,9,11,7])
arr[arr > 6]
>>>array([65.,  9., 11.,  7.])
print(arr > 6)
>>>[False False  True  True  True  True]
```

## 5.Pandas

### 5.1 Pandas基础操作

```python
# 创建Series
s1 = pd.Series([5,17,3,26,31],index=['0','1','2','3','4'])
s1
>>>0     5
   1    17
   2     3
   3    26
   4    31
   dtype: int64
    
# 获得Series的元素和索引
print(s1.values)
print(s1.index)
>>>[ 5 17  3 26 31]
   Index(['a', 'b', 'c', 'd', 'e'], dtype='object')

# 索引和切片操作
s1[2]
>>>3
s1[1:3]
>>>1    17
   2     3
   dtype: int64

# 创建自定义索引Series的另一种方式
s2 = pd.Series({"青菜": 4.1, "白萝卜": 2.2, "西红柿":5.3, "土豆":3.7, "黄瓜": 6.8})
s2
>>>青菜     4.1
   白萝卜    2.2
   西红柿    5.3
   土豆     3.7
   黄瓜     6.8
   dtype: float64
    
# 查看标签是否存在
"青菜" in s2
>>>True

# 结合逻辑运算
s2[(s2 > 5)|(s2 < 3)]
>>>白萝卜    2.2
   西红柿    5.3
   黄瓜     6.8
   dtype: float64
    
s3 = pd.Series([5,17,3,26,31],index=[1,3,5,7,9])
s3
>>>1     5
   3    17
   5     3
   7    26
   9    31
   dtype: int64
    
# loc用标签索引
s3.loc[1:3]
>>>1     5
   3    17
   dtype: int64

# iloc用位置索引
s3.iloc[1:3]
>>>3    17
   5     3
   dtype: int64

# 修改Series里的值
s3.loc[3] = 4.5
s3.iloc[3] = 5
s3
>>>1     5.0
   3     4.5
   5     3.0
   7     5.0
   9    31.0
   dtype: float64
    
# Series加减乘除
s1 = pd.Series([1,4,2,3,5],index=[1,3,5,7,9])
s2 = pd.Series([8,1,7,3,9],index=[1,2,3,5,10])
print(s1.add(s2,fill_value=0))
print(s1.sub(s2,fill_value=0))
print(s1.mul(s2,fill_value=0))
print(s1.div(s2,fill_value=0))
>>>1      9.0     
   2      1.0
   3     11.0
   5      5.0
   7      3.0
   9      5.0
   10     9.0
   dtype: float64
   1    -7.0
   2    -1.0
   3    -3.0
   5    -1.0
   7     3.0
   9     5.0
   10   -9.0
   dtype: float64
   1      8.0
   2      0.0
   3     28.0
   5      6.0
   7      0.0
   9      0.0
   10     0.0
   dtype: float64
   1     0.125000
   2     0.000000
   3     0.571429
   5     0.666667
   7          inf
   9          inf
   10    0.000000
   dtype: float64
    
# Series统计信息
s1.describe()
>>>count    5.000000
   mean     3.000000
   std      1.581139
   min      1.000000
   25%      2.000000
   50%      3.000000
   75%      4.000000
   max      5.000000
   dtype: float64

# 对元素分别执行相同操作
scores = pd.Series({"小明": 92, "小红": 67, "小杰": 70, "小丽": 88, "小华": 76})
def get_grade_from_score(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    else:
        return "D"
grades = scores.apply(get_grade_from_score)
grades
>>>小明    A
   小红    D
   小杰    C
   小丽    B
   小华    C
   dtype: object
    
# 匿名函数
half_scores = scores.apply(lambda x: 0.5*x)
half_scores
>>>小明    46.0
   小红    33.5
   小杰    35.0
   小丽    44.0
   小华    38.0
   dtype: float64
    
# 转换数据类型
scores = scores.astype(str)
scores
>>>小明    92
   小红    67
   小杰    70
   小丽    88
   小华    76
   dtype: object
    
# 针对字符串Series，保留Series每个元素的某一部分
scores.str.slice(0, 1)
>>>小明    9
   小红    6
   小杰    7
   小丽    8
   小华    7
   dtype: object
```

### 5.2 DateFrame

```python
# 创建有标签索引的DateFrame
df = pd.DataFrame({"学号":{"xiaoming":"01","xiaohong":"02","xiaojie":"03",
                        "xiaoli":"04","xiaohua":"05"},
                   "班级":{"xiaoming":"二班","xiaohong":"一班","xiaojie":"二班",
                         "xiaoli":"三班","xiaohua":"一班"},
                   "成绩":{"xiaoming": 92,"xiaohong": 67,"xiaojie": 70,
                        "xiaoli": 88,"xiaohua": 76}})
# 获得DateFrame的索引
df.index
>>>Index(['xiaoming', 'xiaohong', 'xiaojie', 'xiaoli', 'xiaohua'], dtype='object')

# 获得DateFrame的列名
df.columns
>>>Index(['学号', '班级', '成绩'], dtype='object')

# 获得DateFrame的所有值
df.values
>>>array([['01', '二班', 92],
       ['02', '一班', 67],
       ['03', '二班', 70],
       ['04', '三班', 88],
       ['05', '一班', 76]], dtype=object)

# 对DateFrame进行转置
df.T

# 提取DateFrame的列
df["班级"]
>>>xiaoming    二班
   xiaohong    一班
   xiaojie     二班
   xiaoli      三班
   xiaohua     一班
   Name: 班级, dtype: object
df[["学号","成绩"]]
>>>       学号	成绩
xiaoming   01	  92
xiaohong   02	  67
xiaojie	   03	  70
xiaoli	   04	  88
xiaohua	   05	  76

# 根据条件进行筛选
df[df["成绩"] > 80]
>>>	      学号	班级	  成绩
xiaoming   01	  二班	92
xiaoli	   04	  三班	88

# 获取DateFrame的前N行
df.head(N)

# 更新DateFrame的一列值
df["性别"] = pd.Series(["男","男","女","女","男"],index=["xiaoming","xiaohong","xiaohua","xiaojie","xiaoli"])

# 给DateFrame增加一列值
df["身高"] = ["172","173","174","175","176"]

# 删除DateFrame的行
df.drop("xiaojie")

# 删除DateFrame的列
df.drop("身高",axis=1)

# DateFrame加减乘除
s1.add(s2,fill_value=0))
print(s1.sub(s2,fill_value=0))
print(s1.mul(s2,fill_value=0))
print(s1.div(s2,fill_value=0)
      
# DateFrame行或列操作
average = students.loc[:,"考试1":"考试3"].mean(axis=1)
students.loc[:,"考试1":"考试3"].applymap(get_grade)
students.describe()
students["考试1"] = students["考试1"].astype('int')
```

## 6.读取文件

```python
# 读取json文件
survey_df = pd.read_json("D:\cell_phones_survey.json")
survey_df

# 读取csv文件
df1 = pd.read_csv('fifa_players.csv')

# 设置'header=None',表示不要把第一行当做列名
df = pd.read_csv('fifa_players (no header).csv', header=None)

# 设置'index_col=列名/列的位置索引'，返回的DataFrame就会把列名的值作为标签索引
df1 = pd.read_csv('_fifa_players.csv', index_col='player_id')

# 更改展示列数上限,默认是20列
pd.set_option('display.max_columns', 150)

# 更改展示值的字符上限，默认的字符上限是50
pd.set_option('display.max_colwidth', 500)

# 获取开头/结尾/随机N行
df1.head(10)
df1.tail(10)
df1.sample(10)

# 获得DataFrame的概况信息
df1.info()

# 检查数据是否有空缺值
df1.isnull()

# 提取丢失数据的行
scores[scores["考试3"].isnull()]

# 评估重复数据,查看多个变量
df1.duplicated(subset=["学号"，"性别"])

# 评估不一致数据
df1.value_counts()

# 评估无效/错误数据
df1.sort_values()
```

## 7.清洗数据

```python
# 重命名索引和列名
df1.rename(index={"2_":"2","_5":"5","6*":6})
df2.rename(index=函数/方法)
df2.rename(columns=函数/方法)

# inplace=True则rename方法并不返回新的DateFrame
df1.rename(columns={...},inplace=True)

# 把某列设为索引
df2.set_index("str")

# 重设索引
df3.reset_index()

# 对索引和列名重名排序
df4.sort_index()

# 对列进行拆分expand=True，表示把分隔后的结果分别用单独的Series表示
df2['人口密度'].str.split('/', expand=True)

# 把不同列合并成一列,可以传入可选参数sep，来指定拼接时的分隔符
df3["姓"].str.cat(df3["名"],sep="-")

# 把宽数据转换成长数据
df4 = pd.melt(df4, 
	id_vars=['国家代码', '年份'], 	# 里面放入想保持原样的列
    var_name='年龄组', 	 # 要被赋值为在转换后的DataFrame中，包含原本列名值的新列列名
    value_name='肺结核病例数')	# 要被赋值为在转换后的DataFrame中，包含原本变量值的新列列名

# 对行进行拆分
df5.explode("str")

# 对行或列进行删除
df6.drop([2, 4])
df6.drop(["考试2","考试3"], axis=1)

# 对整列缺失值进行填充
df1["国家"] = "中国"

# 对某个缺失值进行填充
df2.loc['003', '销售额'] = 800

# 对部分缺失值进行填充
df3.loc['003': '004', '日期'] = '2005-01-03'

# 自动找到缺失值进行填充
df4['B'].fillna(df4['B'].mean())

# 删除存在缺失值的行
# 删除所有有缺失值的行
df5.dropna()  
# 删除关键值缺失的行
df5.dropna(subset=['工资'])
# 删除有缺失值的列
df5.dropna(axis=1)

# 删除重复数据
# 删除所有变量的值都重复的行
df6['姓名'].drop_duplicates()
# 删除特定变量的值重复的行
df6.drop_duplicates(subset=['姓名', '性别'])
# 可选参数keep
df6.drop_duplicates(subset=['姓名', '性别'], keep='last')

# 对值进行替换
# replace方法参数是两个数字或字符串
df7['学校'].replace('清华', '清华大学')
# replace方法第一个参数是一个列表
df7['学校'].replace(['清华', '五道口职业技术学院', 'Tsinghua University'],'清华大学')
# replace方法参数是单个字典
replace_dict = {'华南理工': '华南理工大学','清华': '清华大学','北大': '北京大学','中大': '中山大学'}
df7.replace(replace_dict)

# 对值的类型进行替换
# Series的astype方法
s1 = pd.Series([1, 2, 3])
s1.astype(float)
# Pandas的数据类型category
s2 = pd.Series(["红色", "红色", "橙色", "蓝色"])
s2.astype('category')

# 转换成datetime日期时间类型
df3['日期'] = pd.to_datetime(df3['日期'])
```

## 8.保存数据

```python
# 写入CSV文件
df1.to_csv('cleaned_sales_data.csv')
cleaned_df = pd.read_csv('cleaned_sales_data.csv')

# 默认to_csv和read_csv后先调用的rename方法，给列换一个有意义的名字
cleaned_df.rename(columns={'Unnamed: 0': '销售ID'}, inplace=True)
# 调用set_index方法，把那一列设置为DataFrame的索引
cleaned_df.set_index('销售ID', inplace=True)

# 写入CSV文件时，不保存DataFrame的索引
df1.to_csv('cleaned_sales_data2.csv', index=False)
cleaned_df_without_index = pd.read_csv('cleaned_sales_data2.csv')
```

## 9.合并数据

```python
# 对DataFrame进行纵向拼接
pd.concat([df1,df2],ignore_index=True)

# 对DataFrame进行横向拼接
pd.concat([df5, df6],axis=1)

# 根据某列的值合并DataFrame
pd.merge(customer_df,order_df,on='客户ID')

# 根据多列的值匹配来进行合并
pd.merge(order_df2,customer_df2,on=['客户ID','订单日期'])

# 当某个变量，虽然在两个DataFrame里面出现，但是列名并不统一
pd.merge(order_df3,customer_df3,left_on=['订单日期','客户编号'],
         right_on=['交易日期','客户ID'])

# 除了用于匹配的列，两张表还有其它的重名列,merge函数会自动为列名的结尾加上后缀，'_x'表示来自第一个表，'_y'表示来自第二个表
pd.merge(df7,df8,on=['日期','店铺'])
pd.merge(df7, df8, on=['日期', '店铺'], suffixes=['_df7', '_df8'])

# 可以给merge函数传入可选参数how='合并类型'，用来指定合并类型，默认合并类型是inner
pd.merge(customer_df4, order_df4, on='客户ID', how='inner')
pd.merge(customer_df4, order_df4, on='客户ID', how='outer')
pd.merge(customer_df4, order_df4, on='客户ID', how='left')
pd.merge(customer_df4, order_df4, on='客户ID', how='right')

# 根据索引合并DataFrame
customer_df4.join(order_df4, lsuffix='_customer', rsuffix='_order')
```

## 10.分组聚合

```python
# 根据变量进行分组
df.groupby('分店编号')[['销售额', '销售数量']].mean()

# 根据多个变量进行分组聚合运算
df.groupby(['分店编号', '时间段'])[['销售额', '销售数量']].mean()

# 自定义聚合函数
def max_plus_10(nums):
    return nums.max() + 10
df.groupby(['分店编号', '时间段'])[['销售额', '销售数量']].apply(max_plus_10)

# 透视表
pd.pivot_table(df,index=['列名1','列名2'],
               columns='列名3',values='列名4',aggfunc='函数名')

# 进行数据分箱
age_bins = [0, 10, 20, 30, 40, 50, 60, 120]
pd.cut(df1.年龄, age_bins)

# 分组标签
age_labels = ['儿童', '青少年', '青年', '壮年', '中年', '中老年', '老年']
pd.cut(df1.年龄, age_bins, labels=age_labels)

# 层次化索引
# 用外层索引，会一次性提取出多行
grouped_df2.loc['001']
# 要提取一行，可以在外层索引后，继续用内层索引继续去提取
grouped_df2.loc['001'].loc['2022Q1']

# 根据条件筛选数据
# 把条件是否符合对应的布尔值的Series，作为DataFrame的索引，来筛选出所有布尔值为True的行
df1[(df1['性别'] == '男') & (df1['年龄'] <= 20)]
df1.query('(性别 == "男") & (年龄 <= 20)')
```

## 11.统计

```python
# 集中趋势
# 平均数
df.mean()
# 中位数
df.median()
# 众数
df.mode()

# 离散趋势
# 极差
df.max() - df.min()
df.ptp()
# 方差
df.var()
# 标准差
df.std()
# 四分位距
df.quantile(0.75) - df.quantile(0.25)

# 分布形状
df.plot()
```

## 12.可视化数据

```python
# 散点图
sns.scatterplot(df1, x="total_bill", y="tip")
plt.show()

# 折线图
sns.lineplot(data=df2, x="year", y="passengers")
plt.show()

# 条形图
sns.barplot(data=df3, x="species", y="body_mass_g")
plt.show()

# 计数图
sns.countplot(data=df3, x="species")
plt.show()

# 饼图
plt.pie(df4['vote'], labels=df4['fruit'], autopct='%.1f%%')
plt.show()
```

