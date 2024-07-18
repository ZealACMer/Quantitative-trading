## Anaconda

Anaconda是一个基于Python的环境管理工具，其中包含了Conda，Python，NumPy，Scipy，Jupyter Notebook在内的超过180个科学库及其依赖项。

Conda是包及其依赖项和环境的管理工具，适用于Python，R，Ruby，Lua，Scala，Java，JavaScript，C/C++和Fortran语言。Conda可以用于快速安装、运行和升级包及其依赖项，在计算机中便捷地创建、保存、加载
和切换环境。

### nb_conda
安装nb_conda用于Jupyter Notebook自动关联nb_conda的环境。
```bash
	conda install nb_conda

	#conda create -n env_name package_names
	
	#创建python2运行环境
	conda create -n py2 python=2

	#创建python3运行环境
	conda create -n py3 python=3

	#创建特定python版本的运行环境
	conda create -n py3.6 python=3.6

	#将当前环境保存为yaml文件(包括Python版本和所有包的名称)
	conda env export > environment.yaml

	#列出共享环境
	conda env list

	#删除环境
	conda env remove -n 环境名称

	#安装jupyter notebook
	conda install jupyter notebook

	#在jupyter notebook中按Ctrl+Enter执行代码块

```

```python
	#python量化编程基础

	#python数据分为字符串、数字、容器、布尔值和空值5种类型。
	#给变量起名时建议用"变量名称+数据名称"的形式，数字不能放在变量名称的开头，Python是区分大小写的。

	##字符串(String)是由数字、字母和下划线组成的一串字符，用单引号或者双引号括起来，字符串不可改变类型。
	#Python2默认的编码方式是ASCII格式，只有在文件开头加入"# encoding: UTF-8"，修改编码格式才能正确地打印汉字，
	#否则在读取中文时会报错。Python3默认使用UTF-8编码，可以正常解析中文。

	#合约类型常量
	PRODUCT_EQUITY=u'股票'
	PRODUCT_FUTURES=u'期货'
	PRODUCT_OPTION=u'期权'
	PRODUCT_INDEX=u'指数'
	PRODUCT_COMBINATION=u'组合'
	PRODUCT_FOREX=u'外汇'
	PRODUCT_UNKNOWN=u'未知'
	PRODUCT_SPOT=u'现货'

	#在实际操作中，倒入Excel表格时，表中的数据是字符串(String)，必须转换成数字数据类型才能进行下一步的数据分析。

	##数字(Number)数据类型用于存储数值，指定一个值时，Number对象就被创建，数字数据类型一旦改变就会分配一个新的对象。
	#整数(int)，在32位系统上，位数为32位，取值-2^31~2^31-1(大概21.47亿)，在64位系统上，位数为64位，取值-2^64~2^64-1
	#浮点数(float)为带有小数位的数字，占8个字节(64位)，其中52位为底，11位表示指数，剩下的一位表示符号。

	#数据类型
	EMPTY_INT=0 #整数型
	EMPTY_FLOAT=0.0 #浮点型

	#交易所推送过来的TICK行情K线数据中，高开低收为浮点型，成交量为整数型
	open=EMPTY_FLOAT #开盘价
	high=EMPTY_FLOAT #最高价
	low=EMPTY_FLOAT #最低价

	volume=EMPTY_INT #成交量
	openInterest=EMPTY_INT #持仓量

	##容器将不同的数据类型放在一起方便使用，根据用途不同分为4种，分别是列表、元组、集合和字典。
	##列表(list)可以完成大多数集合类的数据结构实现的工作。列表中的元素类型可以不同，支持数字、
	#字符串、可以包含嵌套列表。列表是写在方括号"[]"之间，用逗号分隔开的元素列表。和字符串一样，
	#列表同样可以被索引和截取，列表被截取后返回一个包含所需元素的新列表。

	#参数列表，保存了参数的名称
	paramList=['name',
				'className',
				'author',
				'vtSymbol'
				]
	#变量列表，保存了变量的名称
	varList=['inited',
			'trading',
			'pos']

	#同步列表，保存了需要保存到数据库的变量名称
	syncList=['pos']

	#定义5个元素的列表,正序索引0～4，反序索引-1~-5
	elemList = ['A'，'B', 'C', 'D','E']
	#查询第5个元素
	elemList[4]
	#查询倒数第一个元素
	elemList[-1]

	#查询列表长度
	Len=len(elemList)
	Len

	#增加元素
	elemList.append('F')
	elemList

	#删除第3个元素
	del elemList[2]
	elemList

	#修改元素
	elemList[0]='G' #修改第一个元素为'G'
	elemList

	#切片访问
	#访问前3个元素
	elemList[0:3]
	elemList[:3]
	#访问后3个元素
	elemList[-3:]

	##元组(Tuple)是一系列不可变的Python对象，与列表之间的主要区别是元组不能像列表那样改变元素的值(类似只读列表)
	#元组用小括号'()'，列表使用方括号'[]'
	elemTup=('A','B','C','D','E')

	##集合是一个无序不重复元素的序列，列表可以包含重复的元素，用花括号'{}'来表示集合
	#定义集合
	ExchangeSets={'中金所'，'上期所', '郑商所', '大商所', '上交所', '深交所'}
	print(ExchangeSets)

	#定义一个空集合必须用set()，而不是'{}'，因为'{}'是用来创建一个空字典的
	#定义一个空的集合
	BitcoinExchangeSets=set()

	#使用update增加元素
	BitcoinExchangeSets.update(['OKCOIN比特币交易所', 'LBANK比特币交易所'])
	print(BitcoinExchangeSets)

	#使用discard删除一个元素
	BitcoinExchangeSets.discard('OKCOIN比特币交易所')
	print(BitcoinExchangeSets)

	#使用in查询关键字
	txBool='LBANK比特币交易所' in BitcoinExchangeSets
	print(txBool) #true

	##字典，可存储任意类型的对象。字典包括在花括号'{}'中，d={key1: value1, key2: value2}，
	#键必须是唯一的，且是不可变的，但值则不必唯一，可取任何数据类型，如字符串、数字或元组。
	#创建新的字典
	ExchangeDict={'中金所':'CFFEX',
				'上期所':'SHFE',
				'郑商所':'CZCE',
				'大商所':'DCE',
				'国际能源交易中心':'INE'}
	#增加元素
	ExchangeDict['上金所']=['SGE']
	ExchangeDict

	#删除元素
	del ExchangeDict['上金所']
	ExchangeDict

	#查询元素，根据交易所的名称查询交易所代码
	ExchangeDict['中金所']

	#修改元素
	print('修改前，上期所代码：', ExchangeDict['上期所'])
	ExchangeDict['上期所']=['SQS']
	print('修改后，上期所代码：', ExchangeDict['上期所'])

	##布尔值是int的子类(继承int)，故判断如True==1或者False==0时，返回结果是True
	#导入金融库TA-Lib的简单均线函数SMA，定义新的函数sma，若array=True，则输出一系列均线，
	#反之，若array=False，则输出最新的一条均线数据
	import talib
	def sma(self, n, array=False)
	    """简单均线""" #多行注释 #缩进为四个空格，可以设置tab键
	    result=talib.SMA(self.close, n)
	    if array==True: #可以简化为if array:
	        return result
	    return result[-1]

	##空值(None)是Python语言中的一个特殊的值，表示的是一个空对象，但不能将其理解为0，因为0是有意义的。
	#None既可以被赋值给任何变量，也可以将任何变量赋值给None，所以None常用于初始化
	self.datetime=None #python的datetime时间对象

	##函数是实现特定功能，可以重复使用的代码块
	#自定义函数
	'''
	定义函数
	函数功能：两个数相加
	输入：x,y是两个要输入的数字
	输出：z是两个数相加的和，用return导出
	'''
	def add(x,y):
		z=x+y
		return z
	a=1
	b=2
	c=add(a,b)

	print('a和b相加的和为:',c)

	'''
	调用第三方库的函数
	'''
	import talib
	def sma(self, n, array=False):
		result=talib.SMA(self.close,n)
		if array:
			return result
		return result[-1]

	#条件判断，由于Python不支持switch语句，所以多个判断条件，只能用elif来实现
	scoreNum=9.1
	if scoreNum>=8:
		print('看')
	else:
		print('不看')

	age=int(input('请输入年龄，按Enter确认：'))
	if age < 0:
		print('不能小于0岁')
	elif age==1:
		print('1岁')
	elif age==2:
		print('2岁')
	else:
		newAge=22+(age-2)*5
		print('转换新年龄为: ', newAge)

	self.inited=False
	self.count+=1
	if not self.inited and self.count>=100:
		self.inited=True

	##循环
	#pass语句，空语句，是为了保持程序结构的完整性
	#创建列表，for循环遍历元素
	eatList=['早餐','午餐','晚餐']
	for i in eatList:
		print(i) #打印列表元素

	#遍历字典中的元素，将股票代码全部改成大写
	gafataDict={'腾讯':'HK:00700','阿里巴巴':'Alibaba','苹果':'Apple','谷歌':'Google'}
	for key,value in gafataDict.items():
		newValue=value.upper()
		gafataDict[key]=newValue
	print(gafataDict)

	for key,value in gafataDict.items():
		if key=='苹果':
			continue
		print('当前公司: ',key,'当前股票代码: ',value)

	for key,value in gafataDict.items():
		if key=='苹果':
			print('当前公司: ',key,'当前股票代码: ',value)
			break

	#类和实例：面向对象编程中最重要的概念就是类(class)和实例(instance)。
	#类名通常采用驼峰式命名方式，尽量让字面意思体现出类的作用。Python采用
	#多继承机制，一个类可以同时继承多个父类。继承的基类有先后顺序，写在类名
	#后的圆括号"()"里，它也默认继承object类，object类是所有类的基类。

	# encoding: UTF-8
	class Student(object):
		'''定义学生类'''
		#类的属性
		studCnt=0 #学生人数
		classroom='102' #教师编号
		school='木木中学'
		name=None #学生名字
		age=0 #学生年龄

		#实例方法
		def __init__(self, name, age):
			'''类的初始化'''
			self.name=name
			self.age=age 
			self.studCnt+=1
			self.__headmaster='木木' #私有属性，以双下划线开头
		def head(self):
			'''打印headmaster'''
			print 'The headmaster is: ',self.__headmaster
		def displayCount(self):
			'''打印学生人数'''
			print "Total Student %d" % self.studCnt
		def displayStud(self):
			'''打印学生信息'''
			print "Name : ", self.name, ", Age: ",self.age

		#类的方法
		@classmethod
		def displayClassroom(cls):
			'''打印教室编号'''
			print "Classroom is %s" % cls.classroom

		#静态方法
		@staticmethod
		def displayKindergarden():
			'''学校名'''
			print "School is %s" % Student.school

	#类的方法与普通的函数有一个特别的区别，其必须有一个额外的名称放在前面，按照惯例这个名称是self，
	#init是初始化方法，用于设置实例的相关属性，在创建类的实例的时候，一般会自动调用这个方法，对实例
	#属性进行初始化。

	#类的方法由类调用，采用@classmethod装饰，至少传入一个cls参数(代指类本身，类似self)。执行类方法时，
	#自动将调用该方法的类赋值给cls。

	#静态方法由类调用，无默认参数，在方法定义上方加上@staticmethod，它属于类，与实例无关。

	#self引用属性时，相同名称的类属性和实例属性均有的情况下，实例属性的优先级更高
	#类对象cls引用的是类对象的属性和方法

	a=Student('AAA', 5)
	a.classroom #102
	a.__headmaster #木木
	a.head()
	a.displayCount()
	a.displayStud()
	a.displayClassroom()
	a.displayKindergarden()

	#继承类的构造方法：1 父类名称.init(self,参数1,参数2,...) 2 super(子类,self).init(参数1,参数2)
	#在定义子类的构造函数时，只有先继承再构造，才能继承父类的属性。子类的方法如果和父类的方法重名，子类就
	#会覆盖掉父类，这就是继承“多态”。调用方只管调用，不管细节，而新增一个子类时，只要确保新方法编写正确，
	#而不用管原来的代码，这就是著名的“开闭”原则(对扩展开放：允许新增子类；对修改封闭：不需要修改依赖父类的函数)

	#定义子类
	class Group(Student):
		'''定义团体子类'''
		#新增类的属性
		Class = '向日葵'
		Teacher='一一'

		#继承类的构造
		def __init__(self,name,age):
			'''类的初始化'''
			super(Group,self).__init__(name,age)

		#新增实例方法
		def favorite(self):
			'''类的初始化'''
			if self.name == '小明'
			    print u'喜欢篮球'

		#重构旧的实例方法
		def displayCount(self):
			'''统计学生人数'''
				print "人数是%d" % self.studCnt

	#定义子类实例
	b=Group('小明',5)
	b.classroom #访问父类属性
	b.Teacher #访问子类新增属性
	b.favorite() #子类新增方法
	b.displayCount() #子类和父类同名方法，子类方法覆盖父类方法

	#面向对象三个基本特征：
	##封装：将抽象得到的数据和行为（或功能）相结合，形成一个有机的整体（即类）；封装的目的是增强安全性和简化编程，
	#使用者不必了解具体的实现细节，而只要通过外部接口、特定的访问权限来使用类的成员。
	##继承：在遵循原有设计和既有代码不变的前提下，添加新功能，或改进算法。记住其“开闭”原则是对扩展开放，对修改封闭。
	##多态：所有继承子类应该能直接引用父类，这样可以把复杂类型的公用部分剥离出来，形成稳固的抽象类。其他引发变化的
	#相似因素则被分离成多个子类，以确保单一职责原则得到遵守，并能相互替换。


	#常用的两个数据分析库:Numpy与Pandas
	#NumPy中一维数组的格式是Array，Pandas中的则是Series。
	#Array中的每一个元素都必须是同一种数据类型，列表List中的元素可以是不同的数据类型
	#Series有索引，可以用index来定义这个索引，方便索引后面的元素，Array则没有。

	# Numpy一维数组
	import numpy as np 
	import pandas as pd

	# 定义一组数据array([])
	a=np.array([2,3,4,5,6])

	#查询
	a[0]  #2

	#切片
	a[1:3] #array([3,4])

	#数据类型dtype
	a.dtype #dtype('int32')

	#相对于列表，Numpy提供统计分析功能，如用std()计算标准差，用mean()计算平均值，同时也支持向量运算
	a.mean() #平均值 4.0

	a.std() #标准差 1.4142135...

	b=np.array([1,2,3])
	c=b*4
	c #array([4,8,12])

	# pandas一维数组
	#iloc用"index=[]"根据元素的位置来获取元素的值
	#loc根据索引的值来获取元素的值

	import pandas as pd 
	#定义股票数组
	stocks=pd.Series([54.74,190.9,173.14,1050.3,181.86,1139.47], index=['腾讯','阿里巴巴','苹果','谷歌','Facebook','Amazon'])
	#查看一维数组Series的统计属性
	stock.describe()
	>> count 6.000000 #元素个数
	   mean  465.0135 #平均值
	   std   491.1382 #标准差
	   min   54.74000 #最小值
	   25%   175.0202 #四分位
	   50%   186.3500 #四分位
	   75%   835.4500 #四分位
	   max   1139.49  #最大值
	   dtype: float64

	#iloc查询
	stocks.iloc[0] #54.74

	#loc查询
	stocks.loc['Facebook'] #181.86

	#和Numpy的一维数组Array一样，Pandas的Series也支持向量运算，但只能与索引值相同的值相加
	s1=pd.Series([1,2,3,4],index=['a','b','c','d'])
	s2=pd.Series([10,20,30,40],index=['a','b','e','f']);
	s3=s1+s2
	s3
	>> a 11.0
	   b 22.0
	   c NaN
	   d NaN
	   e NaN
	   f NaN
	   dtype: float64

	#在数据分析的过程中，如果不希望出现空值NaN，有两个方法：
	#用dropna()方法删除空值NaN
	#用add将两个一维数组相加，并传入fill_value参数，其中fill_value用来填充空值，比如用0来填充
	s3.dropna()
	>> a 11.0
	   b 22.0
	   dtype: float64

	s3=s1.add(s2,fill_value=0)
	s3
	>> a 11.0
	   b 22.0
	   c 3.0
	   d 4.0
	   e 30.0
	   f 40.0
	   dtype: float64

	#Numpy是通过Array创建二维数组的，所有元素都是同样的数据类型，不利于表达像Excel这样的情况
	#Pandas是通过DataFrame来创建一个二维数组的，有利于表达像Excel中的数据
	#构建二维数组
	import numpy as np 
	import pandas as pd 

	a=np.array([
		[1,2,3,4],
		[5,6,7,8],
		[9,10,11,12]
		])

	#查询二维数组的元素
	a[0,2] #3

	a[0,:] #array([1,2,3,4])

	a[:,0] #array([1,5,9])

	#分组计算，使用数轴参数"axis"，其中axis=1是按每一行进行分组计算，axis=0是按每一列进行分组计算的
	a.mean(axis=1) #array([2.5,6.5,10.5])
	a.mean(axis=0) #array([5.,6.,7.,8.])

	#Pandas二维数组，相对于Numpy的二维数组，有两个优势：
	#1.数组中的每一列都可以是不同类型，方便表示Excel中的数据
	#2.数组中的每一行/列都有一个索引表格，类似于一维数组Series，使得常见的表格数据很容易制作
	import pandas as pd 
	#定义字典，映射别名于相应列的值
	salesDict={
		'日期':['2018-08-01','2018-08-02','2018-08-03'],
		'开盘价':['001616528','001616528','0012602828'],
		'最高价':[236701,236701,236701],
		'收盘价':[330,563,600],
		'最低价':[6,1,2],
		'成交量':[82.8,28,16.9]
	}

	#导入有序字典，将salesDict定义成有序字典
	from collections import OrderedDict
	salesOrderedDict=OrderedDict(salesDict)
	salesDf=pd.DataFrame(salesOrderedDict)
	salesDf
	>> 日期 开盘价 最高价 收盘价 最低价 成交量 #按列输出相应数据

	#DataFrame可分别用iloc和loc查询数组中的元素
	#按照每一列来计算平均值
	salesDf.mean()

	#按照位置获取数据的值
	salesDf.iloc[0,3] #330
	
	#查询某一行
	salesDf.iloc[0,:]

	#查询某一列
	salesDf.iloc[:,0]

	salesDf.loc[0,'收盘价']

	salesDf.loc[0,:]

	salesDf.loc[:,'商品名称']

	#查询特定的列
	salesDf[['商品名称','应收金额']]

	salesDf.loc[:,'商品名称':'实收金额'] #切片指定范围

	#通过条件判断来筛选数据，先构建一个查询条件，然后根据查询条件来筛选出符合查询条件的行
	#构建查询条件
	querySer=salesDf.loc[:,'销售数据']>1

	#查看数组类型
	type(querySer)
	>> pandas.core.series.Series

	#查看判断结果
	querySer

	salesDf.loc[querySer,:]

	##scikit-learn机器学习库
	conda install scikit-learn
	
	#采集数据 导入数据 数据清洗 构建模型 评估模型
	
	##线性回归
	#构建字典
    import Pandas as pd
    from collections import OrderedDict
    examDict={
        '学习时间':[0.50,0.75,1.00,1.25,1.50,1.75,1.75,2.00,2.25,
              2.50,2.75,3.00,3.25,3.50,4.00,4.25,4.50,4.75,5.00,5.50],
        '分数':[10,  22,  13,  43,  20,  22,  33,  50,  62,
              48,  55,  75,  62,  73,  81,  76,  64,  82,  90,  93]
    }
    # 构建有序字典
    examOrderDict=OrderedDict(examDict)

    # 通过有序字典构建数据框pd.DataFrame
    examDf=pd.DataFrame(examOrderDict)
    examDf.head()
    >>   学习时间  分数
    0    0.50     10
    1    0.75     22
    2    1.00     13
    3    1.25     43
    4    1.50     20

    # 提取特征x和标签y
    exam_X=examDf.loc[:,'学习时间']
    exam_y=examDf.loc[:,'分数']

    #绘制散点图，查看相关性
    import matplotlib.pyplot as plt

    # 散点图scatter,其中习惯用大写X表示特征，小写y表示标签
    plt.scatter(exam_X,exam_y,color='b',label='exam data')

    # 然后在label中展开，增加散点图标签
    plt.xlabel('Hours')
    plt.ylabel('Score')

    #显示散点图
    plt.show()

    #交叉验证(Cross-validation)指的是在给定的建模样本中，拿出大部分样本建模型，
    #留小部分样本用刚建立的模型进行评测。traintestsplit是交叉验证中常用的函数，
    #功能是从样本中按比例随机选取训练数据（train）和测试数据（test）。输入参数：
    #所要划分的样本特征X值，所要划分的样本标签y值，train_size为训练数据占比，可以
    #是整数或百分比，整数表示样本数量。输出参数：训练数据的特征X_train，测试数据的特征X_test,
    #训练数据的标签y_train，测试数据的标签y_test

    #建立训练数据和测试数据
    X_train, X_test, y_train, y_test=train_test_split(exam_X,
                                            exam_y,
                                            train_size=0.8) #80%数据为训练数据，20%数据为测试数据
    #输出数据大小
    print('原始数据特征',exam_X.shape ,
        '，训练数据特征', X_train.shape ,
        '，测试数据特征',X_test.shape )

    print('原始数据标签',exam_y.shape ,
        '，训练数据标签', y_train.shape ,
        '，测试数据标签',y_test.shape )
    >> 原始数据特征 (20,)，训练数据特征 (16,)，测试数据特征 (4,)
    原始数据标签 (20,)，训练数据标签 (16,)，测试数据标签 (4,)

    #绘制训练和测试数据的散点图
    plt.scatter(X_train, y_train, color='b',label='train data')
    plt.scatter(X_test, y_test,color='red',label='test data')
    #添加图标标签
    plt.legend(loc=2)
    plt.xlabel('Hours')
    plt.ylabel('Score')
    plt.show()

    #训练模型
    '''
    scikit-learn要求输入的特征必须是二维数组类型，需要进行转换(将训练数据特征和标签转换为二维数组XX行XX列)

    reshape(-1,1) 是一个常用的方法，用于将一维数组转换为二维数组，其中 -1 表示自动计算行数以匹配原始数组中的元素总数。

    import pandas as pd
	import numpy as np

	# 创建一个简单的 Series
	y_train = pd.Series([0, 1, 0, 1, 0])

	# 转换为 NumPy 数组并重塑
	y_train_np = y_train.values.reshape(-1, 1)

	print(y_train_np)

	[[0]
	 [1]
	 [0]
	 [1]
	 [0]]
    '''
    X_train=X_train.values.reshape(-1,1)
    y_train=y_train.values.reshape(-1,1)

    #开始训练模型
    #第1步：导入线性回归
    from sklearn.linear_model import LinearRegression

    # 第2步：创建模型，线性回归
    model=LinearRegression()
    # 第3步：训练模型
    # fit()函数，传入第一个参数是训练数据的特征X，第二个参数是训练参数的标签y
    model.fit(X_train, y_train)
    >> LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

    # 建立最佳拟合线
    a=model.intercept_
    b=model.coef_
    print('最佳拟合线，截距a=',a,'，回归系数b=',b)
    >> 最佳拟合线，截距a=[7.38860355]，回归系数b=[[16.42445973]]

    #绘制最佳拟合线
    import matplotlib.pyplot as plt
    #训练数据散点图
    plt.scatter(X_train, y_train, color='b',label='train data')

    # 训练数据的预测（通过机器学习sklearn中的Linear Regression,用model.predict）
    y_train_pred=model.predict(X_train)

    #绘制最佳拟合线图,线而不是散点，不用scatter，而用plot
    plt.plot(X_train, y_train_pred, color='black', linewidth='3', label='best line')

    # 添加图标标签
    plt.legend(loc=2)
    plt.xlabel('Hours')
    plt.ylabel('Score')

    #显示图函数show()
    plt.show()

    #评估模型：决定系数R平方为79%，代表着79%的考试成绩y的波动可以由回归线描述。
    # 1.相关系数：corr返回结果是一个数据框，存放的是相关系数矩阵
    rDf=examDf.corr()
    print('相关系数矩阵：')
    rDf
    >> 相关系数矩阵：
    Out[107]:
    学习时间  分数
    学习时间  1.000000 0.923985
    分数 0.923985 1.000000

    # 2.转换矩阵
    X_test=X_test.values.reshape(-1,1)
    y_test=y_test.values.reshape(-1,1)

    # 3.线性回归的score方法得到的是决定系数R平方
    #评估模型:决定系数R平方
    model.score(X_test,y_test)
    >> 0.7920076607404332

    ##逻辑回归
    from collections import OrderedDict
    import Pandas as pd

    # 字典—有序字典—数据框
    examDict={
        '学习时间':[0.50,0.75,1.00,1.25,1.50,1.75,1.75,2.00,2.25,2.50,
              2.75,3.00,3.25,3.50,4.00,4.25,4.50,4.75,5.00,5.50],
        '通过考试':[0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1]
    }
    examOrderdDict=OrderedDict(examDict)
    examDf=pd.DataFrame(examOrderdDict)

    #显示前5行
    examDf.head()
    >> 学习时间    通过考试
    0    0.50     0
    1    0.75     0
    2    1.00     0
    3    1.25     0
    4    1.50     0


    # 1.提取特征和标签
    exam_X=examDf.loc[:,'学习时间']
    exam_y=examDf.loc[:,'通过考试']

    # 2.绘制散点图
    import matplotlib.pyplot as plt
    plt.scatter(exam_X, exam_y, color='b',label='exam data')

    plt.xlabel('Hours')
    plt.ylabel('Pass')
    plt.show()

    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test=train_test_split(exam_X,
                                          exam_y,
                                          train_size=0.8)
    print('原始数据特征：', exam_X.shape,
        '，训练数据特征：', X_train.shape,
        '，测试数据特征：', X_test.shape)

    print('原始数据标签：', exam_y.shape,
        '，训练数据标签：', y_train.shape,
        '，测试数据标签：', y_test.shape)
    >> 原始数据特征 (20,)，训练数据特征 (16,)，测试数据特征 (4,)
    原始数据标签 (20,)，训练数据标签 (16,)，测试数据标签 (4,)

    import matplotlib.pyplot as plt
    plt.scatter(X_train, y_train, color='b', label='train data')
    plt.scatter(X_test, y_test, color='red', label='test data')

    plt.legend(loc=2)
    plt.xlabel('Hours')
    plt.ylabel('Pass')
    plt.show()

    # 逻辑回归函数LogisticRegression
    from sklearn.linear_model import LogisticRegression
    X_train=X_train.reshape(-1,1)
    X_test=X_test.reshape(-1,1)
    Model=LogisticRegression()
    model.fit(X_train, y_train)
    >> LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
            intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
            penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
            verbose=0, warm_start=False)

   	#决定系数R平方为50%，代表着50%的考试成绩y的波动可以由回归线描述。
   	model.score(X_test, y_test)
    >> 0.5

    ##Matplotlib
    #用列表绘制线条
    import matplotlib.pyplot as plt

    #第1步：利用列表信息来定义x和y轴坐标
    x=[1, 2, 3, 4]
    y=[1,4,9,16]

    #第2步：使用plot绘制线条
    #第1个参数是x的坐标值，第2个参数是y的坐标值
    #属性：color:线条颜色，r表示红色；marker:点的形状，o表示点为圆圈标记；linestyle：线条的形状,dashed表示虚线连接各点
    plt.plot(x, y)

    '''
    axis：坐标轴范围
    语法为axis[xmin, xmax, ymin, ymax]，
    也就是axis[x轴最小值, x轴最大值, y轴最小值, y轴最大值]
    '''
    plt.axis([0, 6, 0, 20])
    plt.xlabel('x坐标轴')
    plt.ylabel('y坐标轴')
    plt.title('标题')
    # text: 注释文本。
    # x, y: 注释文本在图中的位置。
    # xytext: 注释框的偏移位置。
    # arrowprops: 箭头属性，可以是字典，用于设置箭头的样式。
    plt.annotate('这里是注释',xy=(2,5),xytext=(2,10),
            arrowprops=dict(facecolor='black',shrink=0.01),)

    #第3步：显示图形
    plt.show()


    #用数组绘图
    import Numpy as np
    t=np.arrange(0, 5, 0.2) #等差数列arrange([start,]stop,[step,])
    t
    >> array([0. , 0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6, 1.8, 2. , 2.2, 2.4,
          2.6, 2.8, 3. , 3.2, 3.4, 3.6, 3.8, 4. , 4.2, 4.4, 4.6, 4.8])

    # 线条1
    x1=y1=t

    # 线条2
    x2=x1
    y2=t**2

    # 线条3
    x3=x1
    y3=t**3

    # 3条线绘图
    LineList=plt.plot(x1, y1,
                  x2, y2,
                  x3, y3)

    #用setp方法可以同时设置多条线条的属性
    plt.setp(LineList, color='blue')
    plt.show()


    #多个图的绘制
    #创建画板:
    plt.figure(1)
    #创建画纸，并选择画纸1
    #参数的前2个数字代表要生成几行几列的子图矩阵，第3个数字代表选中的子图位置
    plt.subplot(2, 1, 1)

    #在画纸1上绘图
    plt.plot([1, 2, 3])

    #选择画纸2
    plt.subplot(2, 1, 2)

    #在画纸2上绘图
    plt.plot([4, 5, 6])

    plt.show()

```
