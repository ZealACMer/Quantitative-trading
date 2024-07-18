###vn.py是基于Python语言的量化交易系统
#完整的量化业务链包括数据获取、数据分析、策略回测和实盘交易。

#操作系统Windows、Linux、OS X三选一
#在安装完anaconda之后，继续安装以下库

#安装PostgreSQL数据库接口psycopg2,搜索相应的源
conda search psycopg2

conda install psycopg2

#安装数据接口库TuShare，conda中没有相应的源，需要使用pip安装到当前环境中
pip install tushare

```python
    #SciPy
    from scipy import io as spio
    import numpy as np 
    a=np.arange(10)
    spio.savemat('a.mat',{'a':a})
    data=spio.loadmat('a.mat',struct_as_record=True)
    data['a']
    >>> array([[0,1,2,3,4,5,6,7,8,9]])

    #计算矩阵的行列式
    from scipy import linalg
    m=np.array([[1,2],[3,4]])
    linalg.det(m)
    >>> -2.0

    #计算最小值
    import numpy as np 
    from scipy import optimize
    import matplotlib.pyplot as plt 

    def f(x):
    	return x**2+20*np.sin(x)

    x=np.arange(-10,10,0.1)

    plt.plot(x,f(x))

    #使用穷举法计算最小值，可以替换为模拟退火等更高效的算法
    grid=(-10,10,0.1)
    x_min=optimize.brute(f,(grid,))
    x_min
    >>> array([-1.42754883])


    import pandas as pd 
    data={'A':['x','y','z'],'B':[1000,2000,3000],'C':[10,20,30]}
	df=pd.DataFrame(data,index=['a','b','c'])
	df

	  A  B   C 
	a x 1000 10
	b y 2000 20
	c z 3000 30

	#左边的a、b、c是索引，代表每一行数据的标识。这里的索引是显式指定的。如果没有指定，会自动生成从0开始的数字索引。
	#列标签，表头的A、B、C就是标签部分，代表了每一列的名称。
	#位于表格正中间的9个数据就是DataFrame的数据部分。

	df.values
	>>>
	array([['x', 1000, 10],
       ['y', 2000, 20],
       ['z', 3000, 30]], dtype=object)

	df.index #行索引
	>>>
	Index(['a', 'b', 'c'], dtype='object')

	df.columns #列索引
	>>>
	Index(['A', 'B', 'C'], dtype='object')

	#指定列索引和行索引
	df=pd.DataFrame(data,columns=['C','B','A'],index=['a','b','c'])

	#如果某列不存在，为其赋值，会创建一个新列，其他数据不变。
	df['D']=10
		A B C D 
	   a      10
	   b      10
	   c      10

    #删除列
    del df['D']

    #添加一行
    new_df=pd.DataFrame({'A':'new','B':4000,'C':40},index=['d'])
	df=df.append(new_df)
	df

	#使用loc添加行
	df.loc['e']=['new2',5000,50]
	df 

	df2=pd.DataFrame([1,2,3,4,5],index=['a','b','c','d','z'],columns=['E'])
	df2
	  E 
	a 1 
	b 2
	c 3
	d 4
	z 5

	df
	  A    B    C 
	a x    100  10 
	b y    200  20  
	c z    300  30  
	d new  400  40  
	e new2 5000 50

	df.join(df2)
	  A    B    C  E
	a x    100  10 1
	b y    200  20 2 
	c z    300  30 3 
	d new  400  40 4 
	e new2 5000 50 NaN #E没有行索引e

	df.join(df2,how='outer') # inner交集 outer并集 left默认值，调用方法的对象的索引值，right被连接对象的索引值

	  A    B    C  E
	a x    100  10 1
	b y    200  20 2 
	c z    300  30 3 
	d new  400  40 4 
	e new2 5000 50 NaN
	z NaN  NaN  NaN 5

	#生成一个DatetimeIndex对象的日期序列,参数：start 开始日期字符串 end 结束日期字符串 periods整数,如果
	#start或者end空缺，则必须指定。从start开始，生成periods日期数据。
	dates=pd.date_range('20160101',periods=8)
	dates
	DatetimeIndex(['2016-01-01', '2016-01-02', '2016-01-03', '2016-01-04',
                '2016-01-05', '2016-01-06', '2016-01-07', '2016-01-08'],
                dtype='datetime64[ns]', freq='D') #周期默认为D，即一天，也可以指定为5H(5小时)
	频率参数值：
	B 交易日
	C 自定义交易日
	D 日历日
	W 每周
	M 每月底
	SM 半个月频率(15号和月底)
	BM 每个月份最后一个交易日
	CBM 自定义每个交易月
	MS 日历月初
	SMS 月初开始的半月频率(1号，15号)
	BMS 交易月初
	CBMS 自定义交易月初
	Q 季度末
	BQ 交易季度末
	QS 季度初
	BQS 交易季度初
	A 年末
	BA 交易年度末
	AS 年初
	BAS 交易年度初
	BH 交易小时
	H 小时 
	T,min 分钟
	S 秒
	L,ms 毫秒
	U,us 微妙
	N 纳秒

	df=pd.DataFrame(np.random.randn(8,4),index=dates,columns=list('ABCD')) #8行4列的数据，列标签ABCD，行标签日期
	df

	#按列求和
	df.sum()
	>>>
	A    0.241727
	B   -0.785350
	C   -0.547433
	D   -1.449231
	dtype: float64

	#按列求均值
	df.mean() 
	A    0.030216
	B   -0.098169
	C   -0.068429
	D   -0.181154
	dtype: float64

	#按列求累计总和
	df.cumsum() #第一行不变，第二行的数变为第一行加上第二行，以此类推

	#使用describe一键生成多种统计数据
	df.describe()

	      A  B  C  D
   count
   mean  按列进行统计的数据
   std 
   min 
   25%
   50%
   75%
   max

   #根据某一列的值进行排序(从小到大)
   df.sort_values('A')

   #根据行索引排序，倒序
   df.sort_index(ascending=False)

   #选取某一列，返回的是Series对象
   df['A']
   >>>
    2016-01-01   -1.142350
	2016-01-02   -0.816178
	2016-01-03    0.030206
	2016-01-04    1.930175
	2016-01-05    0.571512
	2016-01-06    0.220445
	2016-01-07    0.292176
	2016-01-08   -0.844260
	Freq: D, Name: A, dtype: float64

   #选取某几行(0到4行)
   df[0:5]
   				A  B  C  D
    2016-01-01   
	2016-01-02   
	2016-01-03   ······
	2016-01-04    
	2016-01-05
   
   #根据标签Label选取数据，使用的是loc方法
   df.loc[dates[0]]
   >>>
	A   -1.142350
	B   -1.999351
	C    0.772343
	D   -0.851840
	Name: 2016-01-01 00:00:00, dtype: float64

	df.loc[:,['A','C']] #所有行，A和C两列
	df.loc['20160102':'20160106',['A','C']] #2~6五行，A和C两列
	df.loc['20160102',['A','C']] #如果只有一个时间点，返回的值是Series对象
	df.loc['20160102':'20160102',['A','C']] #如果想要获取DataFrame对象

	#loc是按照标签进行索引，iloc方法是按照绝对位置获取数据
	df.iloc[2] #获取第二行数据

	df.iloc[3:6,1:3]

	#对于工业代码，推荐使用loc、iloc等方法。因为这些方法是经过优化的，拥有更好的性能。

	#选取A列中值大于0的行
	df.A>0
	>>>
	2016-01-01    False
	2016-01-02    False
	2016-01-03     True
	2016-01-04     True
	2016-01-05     True
	2016-01-06     True
	2016-01-07     True
	2016-01-08    False
	Freq: D, Name: A, dtype: bool #Series类型的布尔数组

	#通过布尔数组选取相应的行
	df[df.A>0]

	#寻找df中所有大于0的数据，不能与0比较的数据，显示为NaN
	df[df>0]

	#为df添加新的一列E
	df['E']=0
	df

	#使用loc改变一列的值
	df.loc[:,'E']=1
	df

	#使用loc改变单个值
	df.loc['2016-01-01','E']=2
	df

	#将D列全部变为2
	#np.array([2]*6) 时，你实际上是在创建一个包含6个元素的数组，每个元素的值都是2。
	#[2 2 2 2 2 2]
	df.loc[:,'D']=np.array([2]*len(df))
	df

	#DataFrame使用ix来进行混合索引，行索引使用绝对位置，列索引使用标签
	df.ix[1,'E'] = 3
	df 

	#假如如索引本身就是整数类型，那么ix只会使用标签索引，而不会使用位置索引，即使没能在索引中找到相应的值（这个时候会报错）。
	#如果索引既有整数类型，也有其他类型（比如字符串），那么ix对于整数会直接使用位置索引，但对于其他类型（比如字符串）则会使用标签索引。

	#建议使用loc和iloc，避免问题的出现

	#Series由一组数据以及相关的数据标签(索引)组成。
	import pandas as pd 
	s=pd.Series([1,4,6,2,3])
	s 
	>>>
	0    1
	1    4
	2    6
	3    2
	4    3

	#获取值
	s.values 
	>>>
	array([1,4,6,2,3]),dtype=int64

	#获取索引
	s.index 
	>>>
	Index([0,1,2,3,4]),dtype=int64

	#定义索引
	s=pd.Series([1,2,3,4],index=['a','b','c','d'])
	s 

	#利用索引选取数据
	s['a']
	s[['b','c']]


	#对Series进行数据运算的时候也会保留索引
	s[s>1]
	>>>
	b 2
	c 3
	d 4

	s*3
	>>>
	a 3 
	b 6 
	c 9
	d 12

	#Series在不同索引中对齐数据
	In[26]: s1=pd.Series([1,2,3],index=['a','b','c'])
	In[27]: s2=pd.Series([4,5,6],index=['b','c','d'])
	In[28]: s1+s2
	Out[28]: 
	a   NaN
	b     6
	c     8
	d   NaN


	#Series的索引可以通过赋值的方式直接修改
	In[30]: s.index
	Out[30]: Index([u'a', u'b', u'c', u'd'], dtype='object')
	In[31]: s.index=['w','x','y','z']
	In[32]: s.index
	Out[32]: Index([u'w', u'x', u'y', u'z'], dtype='object')
	In[33]: s
	Out[33]: 
	w    1
	x    2
	y    3
	z    4

	#在Python中，前缀u 用来表示一个字符串是Unicode字符串。这种字符串类型在Python 2.x中是必要的，因为默认的字符串类型是ASCII，而Unicode字符串需要显式声明。


	##StatsModels是Python的统计建模和计量经济学工具包
	#Scipy的stats模块是围绕随机变量提供数值方法的，缺少回归方法，
	#StatsModels提供了回归方法

	import numpy as np
	import statsmodels.api as sm

	# Generate artificial data (2 regressors + constant)
	nobs = 100
	#生成一个 nobs 行2列的数组，数组中的每个元素都是从0到1的随机数
	X = np.random.random((nobs, 2))
	#使用 sm.add_constant 函数向 X 添加一个常数项（通常是1）。这在回归分析中是常见的做法，用于拟合截距项。结果是一个 nobs 行3列的数组，其中每行的第一个元素都是1。
	X = sm.add_constant(X)
	beta = [1, .1, .5]
	e = np.random.random(nobs)
	y = np.dot(X, beta) + e

	# Fit regression model
	results = sm.OLS(y, X).fit()

	# Inspect the results
	print(results.summary())


	#查看两个api包含的函数
	import statsmodels
	print(len(dir(statsmodels.formula.api)))
	print(len(dir(statsmodels.api)))
	print(dir(statsmodels.formula.api))
	print(dir(statsmodels.api))

	##seaborn是统计可视化库
	#Matplotlib是Python最核心、最底层的可视化库

	#散点图
	import numpy as np
	import matplotlib.pyplot as plt
	N=1000
	x=np.random.randn(N)
	y=np.random.randn(len(x))
	plt.scatter(x,y)
	plt.show()

	#直方图：主要用于研究数据的频率分布
	import numpy as np
	import matplotlib.pyplot as plt

	mu = 100  # mean of distribution
	sigma = 20  # standard deviation of distribution
	x = mu + sigma * np.random.randn(2000)

	plt.hist(x, bins=20,color='red',normed=True)

	plt.show()

	#函数图
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib.patches import Polygon

	def func(x):
	    return -(x - 1) * (x - 6)+50

	x = np.linspace(0, 10)
	y = func(x)

	fig, ax = plt.subplots()
	plt.plot(x, y, 'r', linewidth=2)
	plt.ylim(ymin=0)

	a, b = 2, 9 
	ix = np.linspace(a, b)
	iy = func(ix)
	verts = [(a, 0)] + list(zip(ix, iy)) + [(b, 0)]
	poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
	ax.add_patch(poly)

	plt.text(0.5 * (a + b), 20, r"$\int_a^b (-(x - 1) * (x - 6)+50)\mathrm{d}x$",
	         horizontalalignment='center', fontsize=20)

	plt.figtext(0.9, 0.05, '$x$')
	plt.figtext(0.1, 0.9, '$y$')

	ax.set_xticks((a, b))
	ax.set_xticklabels(('$a$', '$b$'))
	ax.set_yticks([])

	plt.show()

	#在Windows下面，使用Matplotlib画图时，中文会显示为乱码，主要原因是Matplotlib默认没有指定中文字体。

	#解决方法一，画图的时候指定字体
	import matplotlib.pyplot as plt
	from matplotlib.font_manager import FontProperties
	font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)

	plt.plot([1,2,3])
	plt.title(u"测试",fontproperties=font)

	#解决方法二：修改配置文件
	'''
	将C:\Windows\Fonts下面的字体simsun.ttf（微软雅黑字体）复制到D:\Programs\Anaconda\Lib\site-packages\matplotlib\mpl-data\fonts\ttf文件夹下（Anaconda文件夹的位置与安装位置有关）。

	用记事本打开D:\Programs\Anaconda\Lib\site-packages\matplotlib\mpl-data\matplotlibrc。找到如下两行代码：

	#font.family         : sans-serif
	#font.sans-serif     : Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, 
    Arial, Helvetica, Avant Garde, sans-serif

    去掉这两行前面的#，并且在font.sans-serif的冒号后面加上SimHei，结果如下所示：
    font.family         : sans-serif
 	font.sans-serif     : SimHei,Bitstream Vera Sans, Lucida Grande, Verdana, Geneva,  
    Lucid, Arial, Helvetica, Avant Garde, sans-serif
	'''
	#重新启动Python，Matplotlib就可以输出中文字符了。

	##seaborn中文显示乱码问题
	import seaborn as sns
	sns.set_style('whitegrid',{'font.sans-serif':['simhei','Arial']})

	#如果Mac OS X修改配置文件的方法没有成功，采用以下方法
	from pylab import mpl
	mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
	mpl.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题


	#使用seaborn生成线性回归图
	%matplotlib inline  
	import seaborn as sns
	import matplotlib.pyplot as plt

	sns.set()

	# 加载数据
	iris = sns.load_dataset("iris")

	# 绘图
	g = sns.lmplot(x="sepal_length", y="sepal_width", hue="species",
	               truncate=True, size=6, data=iris)

	# 更改x,y轴的标签
	g.set_axis_labels("Sepal length (mm)", "Sepal width (mm)")

	#观察数据
	iris.head()

	'''
	以看到，iris是一个DataFrame，按照species分类，绘制了sepal_length和sepal_width的线性回归图，不同的类别是由不同的颜色来表现的。线性回归图包含了散点图和线性回归的拟合直线。
	'''

	import numpy as np
	import seaborn as sns
	import matplotlib.pyplot as plt
	sns.set(style="white", context="talk")
	rs = np.random.RandomState(7)


	# 将图分为3*1的子图
	f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

	# 生成数据
	x = np.array(list("ABCDEFGHI"))
	y1 = np.arange(1, 10)

	# 绘制柱状图
	sns.barplot(x, y1, palette="BuGn_d", ax=ax1)

	# 更改y标签
	ax1.set_ylabel("Sequential")

	# 生成新数据，绘制第二个图
	y2 = y1 - 5
	sns.barplot(x, y2, palette="RdBu_r", ax=ax2)
	ax2.set_ylabel("Diverging")

	# 重新排列数据，绘制第三个图
	y3 = rs.choice(y1, 9, replace=False)
	sns.barplot(x, y3, palette="Set3", ax=ax3)
	ax3.set_ylabel("Qualitative")

	# 移除边框
	sns.despine(bottom=True)

	# 将y轴的tick设置为空（美化图形）
	plt.setp(f.axes, yticks=[])

	# 设置三个图的上下间隔
	plt.tight_layout(h_pad=3)

	#使用seaborn绘制热力图
	import matplotlib.pyplot as plt
	import seaborn as sns
	sns.set()

	# 加载数据
	flights_long = sns.load_dataset("flights")
	flights = flights_long.pivot("month", "year", "passengers")

	# 绘制热力图
	f, ax = plt.subplots(figsize=(9, 6))
	sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)

	#在研究金融数据的时候，最常用的功能就是K线图
	#将十年的数据绘制到一张图上进行观察。这种情况就需要利用动态图了，
	#动态图的意思就是，可以像行情软件那样放大或缩小区间，来回滚动，这样观察长时间的序列就非常容易了
	#highcharts可以绘制很多动态的可视化图形，进行数据可视化研究非常方便

	#Python如何调用highcharts
	#github项目地址：https://github.com/kyper-data/python-highcharts/blob/master/examples/highstock/candlestick-and-volume.py

	#绘制带成交量的K线图
	# -*- coding: utf-8 -*-
	"""
	Highstock Demos
	Two panes, candlestick and volume: http://www.highcharts.com/stock/demo/candlestick-
	    and-volume
	"""
	from highcharts import Highstock
	from highcharts.highstock.highstock_helper import jsonp_loader
	H = Highstock()

	data_url = 'http://www.highcharts.com/samples/data/jsonp.php?filename=aapl-ohlcv.json&callback=?'
	data = jsonp_loader(data_url, sub_d = r'(\/\*.*\*\/)')

	ohlc = []
	volume = []
	groupingUnits = [
	['week', [1]], 
	['month', [1, 2, 3, 4, 6]]
	]

	for i in range(len(data)):
	    ohlc.append(
	        [
	        data[i][0], # the date
	        data[i][1], # open
	        data[i][2], # high
	        data[i][3], # low
	        data[i][4]  # close
	        ]
	        )
	    volume.append(
	        [
	        data[i][0], # the date
	        data[i][5]  # the volume 
	        ]
	    )


	options = {
	    'rangeSelector': {
	                'selected': 1
	            },

	    'title': {
	        'text': 'AAPL Historical'
	    },

	    'yAxis': [{
	        'labels': {
	            'align': 'right',
	            'x': -3
	        },
	        'title': {
	            'text': 'OHLC'
	        },
	        'height': '60%',
	        'lineWidth': 2
	    }, {
	        'labels': {
	            'align': 'right',
	            'x': -3
	        },
	        'title': {
	            'text': 'Volume'
	        },
	        'top': '65%',
	        'height': '35%',
	        'offset': 0,
	        'lineWidth': 2
	    }],
	}

	H.add_data_set(ohlc, 'candlestick', 'AAPL', dataGrouping = {
	                    'units': groupingUnits
	                }
	)
	H.add_data_set(volume, 'column', 'Volume', yAxis = 1, dataGrouping = {
	                    'units': groupingUnits
	                }
	)

	H.set_dict_options(options)
	H.save_file('highcharts')
	H

	#最终显示要依赖JavaScript，只能在Jupyter Notebook中显示。也可以使用H.save_file(file_path)将图像保存为html文件。这个文件可以分享给其他人，使用普通浏览器就能打开。

	#highcharts还可以绘制雷达图
	# -*- coding: utf-8 -*-
	"""
	Highcharts Demos
	Spiderweb: http://www.highcharts.com/demo/polar-spider
	"""

	from highcharts import Highchart
	H = Highchart(width=550, height=400)

	options = {
	    'chart': {
	        'polar': True,
	        'type': 'line',
	        'renderTo': 'test'
	    },

	    'title': {
	        'text': 'Budget vs spending',
	        'x': -80
	    },

	    'pane': {
	        'size': '80%'
	    },

	    'xAxis': {
	        'categories': ['Sales', 'Marketing', 'Development', 'Customer Support',
	                'Information Technology', 'Administration'],
	        'tickmarkPlacement': 'on',
	        'lineWidth': 0
	    },

	    'yAxis': {
	        'gridLineInterpolation': 'polygon',
	        'lineWidth': 0,
	        'min': 0
	    },

	    'tooltip': {
	        'shared': True,
	        'pointFormat': '<span style="color:{series.color}">{series.name}: <b>${point.y:,.0f}</b><br/>'
	    },

	    'legend': {
	        'align': 'right',
	        'verticalAlign': 'top',
	        'y': 70,
	        'layout': 'vertical'
	    },
	}

	data1 = [43000, 19000, 60000, 35000, 17000, 10000]
	data2 = [50000, 39000, 42000, 31000, 26000, 14000]

	H.set_dict_options(options)
	H.add_data_set(data1, name='Allocated Budget', pointPlacement='on')
	H.add_data_set(data2, name='Actual Spending',  pointPlacement='on')
	H

	```
##在量化投资领域，除了少数的特殊统计模型，Python的统计功能几乎完全够用。


