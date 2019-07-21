# 学习率
学习率 learning_rate ： 表示了 每次参数更新的幅度大小 。 学习率过大， 会 导致 待优化的参数 在最小值附近波动 ，不收敛 ；学习率过小， 会导致 待优化的参数收敛缓慢 。在训练过程中， 参数的更新向着损失函数梯度下降的方向 。  
参数的更新公式 为：  
$$
w_{n+1} = w_{n} - learingRate*∇
$$
假设损失函数为 
$$
loss = (w + 1)^2
$$
梯度是损失函数 loss 的导数为 ∇=2w+2。如参数初值为 5，学习率为 0.2，则参数和损失函数更新如下：  
1 次 参数 w：5 5 - 0.2 * (2 * 5 + 2) = 2.6  
2 次 参数 w：2.6 2.6 - 0.2 * (2 * 2.6 + 2) = 1.16  
3 次 参数 w：1.16 1.16 – 0.2 * (2 * 1.16 + 2) = 0.296  
4 次 参数 w：0.296  
损失函数 loss 的最小值会在(-1,0)处得到，此时损失函数的导数为 0,得到最终参数 w = -1。代码如下：

~~~
#coding:utf=8

#设损失函数 loss = (w+1)^2,令w初值是常数5.方向传播就是求最优w,即求最小loss对应的w值
import tensorflow as tf

#定义待优化参数w初值5
w = tf.Variable(tf.constant(5, dtype = tf.float32))

#定义损失函数loss
loss = tf.square( w + 1 )

#定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize( loss )

#生成会话,训练40轮
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run( init_op )
	for i in range( 40 ):
		sess.run(train_step)
		w_val = sess.run(w)
		loss_val = sess.run(loss)
		print "After %s steps: w is %f, loss is %f." %(i,w_val,loss_val) 
~~~

由结果可知，随着损失函数值的减小，w 无限趋近于-1，模型计算推测出最优参数 w = -1。  


## 学习率的设置
> 学习率过大，会导致 待优化的参数 在最小值附近波动，不收敛；学习率过小， 会导致 待优化的参数收敛缓慢。  

例如：  
① 对于上例的损失函数 loss = (w + 1)^2 。则将上述代码中学习率修改为 1，其余内容不变。  
> 由运行结果可知，损失函数 loss 值并没有收敛，而是在 5 和-7 之间波动。   

② 对于上例的损失函数 loss = (w + 1)^2 。则将上述代码中学习率修改为 0.0001，其余内容不变。  
> 由运行结果可知，损失函数 loss 值缓慢下降，w 值也在小幅度变化，收敛缓慢。 

## 指数衰减学习率
指数衰减学习率 ： 学习率随着 训练轮数变化而动态更新  
学习率计算公司如下：  
Learning_rate = LEARNING_RATE_BASE*LEARNING_RATE_DECAY*global_step/LEARNING_RATE_BATCH_SIZE  
用Tensorflow的函数表示为：  
global_step = tf.Variable(0, trainable=False)  
learning_rate = tf.train.exponential_decay( 
LEARNING_RATE_BASE,
global_step,
LEARNING_RATE_STEP, LEARNING_RATE_DECAY,
staircase=True/False)  
其中，LEARNING_RATE_BASE 为学习率初始值，LEARNING_RATE_DECAY 为学习率衰减率,global_step 记录了当前训练轮数，为不可训练型参数。学习率 learning_rate 更新频率为输入数据集总样本数除以每次喂入样本数。若 staircase 设置为 True 时，表示global_step/learning rate step 取整数，学习率阶梯型衰减；若 staircase 设置为 false 时，学习率会是一条平滑下降的曲线。  
例如：  
> 在本例中，模型训练过程不设定固定的学习率，使用指数衰减学习率进行训练。其中，学习率初值设置为 0.1，学习率衰减率设置为 0.99，BATCH_SIZE 设置为 1

~~~
#coding:uft-8
#设损失函数 loss = (w + 1)^2,令w初值是常数10。方向传播就是求最优w,即求最小loss对应的w值
#使用只是衰减的学习率,在迭代初期得到最高的下降速度,可以在较小的训练轮数下取得更有收敛度。
LEARNING_RATE_BASE = 0.1 #最初学习率
LEARNING_RATE_DECAY = 0.99 #学习率衰减率
LEARNING_RATE_STEP = 1 #喂入多少轮BATCH_SIZE后,跟新一次学习率，一般设置为总样本/BATCH_SIZE

#运行了几轮BATCH_SIZE的计数器,初值为0,设为不被训练
global_step = tf.Variable(0,trainable = False)

#定义指数下降学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_STEP,LEARNING_RATE_DECAY,staircase=True)

#定义待优化参数w初值10
w = tf.Variable(tf.consant( 5, dtype = tf.float32 ))

#定义损失函数loss
loss = tf.square(w+1)

#定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize( loss, global_step = global_step )

#生成会话,训练40轮
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	for i in range(40):
		sess.run( train_step )
		learing_rate_val = sess.run( learning_rate )
		w_val = sess.run( w )
		loss_val = sess.run( loss )
		print "After %s steps:global_step is %f,w is %f,learing rate is %f,loss is %f"%(i,learing_rate_val,w_val,loss_val)
~~~

