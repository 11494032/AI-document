# `mnist实例训练`

熟悉基本流程
参考资料：caffe-master/examples/mnist/readme.md
## 下载数据源
~~~
./data/mnist/get_mnist.sh 
~~~
## 转换成lmdb数据
~~~
./example/mnist/create_mnist.sh
~~~
## 搭建神经网络
~~~
./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt
~~~
## 训练
~~~
./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt 2>1 | tee a.log
~~~
## 查看训练的效果图
~~~
python tools/extra/plot_training_log.py.example     python tools/extra/plot_training_log.py.example 6 aa.png a.log
~~~
## 查看神经网络系统的层次关系
~~~
python ~/caffe/python/draw_net.py cc.prototxt xx.png  --rankdir=BT #从BOTTEM 到TOP
~~~