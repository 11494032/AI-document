# `Caffe安装步骤`

服务器配置和Caffe配置
## 服务器配置

 - [服务器配置参考](https://www.cnblogs.com/banju/p/7918895.html) 
 - GPU安装  
    1.关闭bios里的secure boot(否则无法使用显卡驱动)  
    2.设置----软件和更新----附加驱动----使用NVIDIA....(专有)----应用更改----（之后为半小时的等待）-----提示重新启动。  
    3.检测显卡驱动是否安装成功 nvidia-smi  
    4.安装依赖关系包   
~~~
	sudo apt-get install vim python-pip git    
   	sudo apt-get install build-essential libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
   	sudo apt-get install libopenblas-dev liblapack-dev libatlas-base-dev libgflags-dev libgoogle-glog-dev liblmdb-dev
   	sudo apt-get install libboost-all-dev
   	sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev
~~~
## 安装CUDA
 - 下载安装文件。首先去英伟达官网下载cuda安装包  
 	https://developer.nvidia.com/cuda-toolkit-archive  
- 安装包  
		~~~ 
		sudo sh ./cuda_8.0.61_375.26_linux.run 
		~~~
- 检测安装    
~~~
	sudo nvcc -V  
	cuda samplecd /usr/local/cuda-7.5/samples/1_Utilities/deviceQuery  
	make -j4  
	./deviceQuery  
~~~  
## 安装CUDNN
 - 下载cudnn的安装文件  
	https://developer.nvidia.com/rdp/cudnn-archive  
 - 安装  
  ~~~
	tar -zxvf cudnn-8.0-linux-x64-v5.1.tgz  
    cd cuda    
    sudo cp lib64/lib* /usr/local/cuda/lib64/   
    sudo cp include/cudnn.h /usr/local/cuda/include/   
    cd /usr/local/cuda/lib64/  
    sudo chmod +r libcudnn.so.x.x.x  # x替换为自己查看.so的版本 ，如libcudnn.so.5.1.10  
    sudo ln -sf libcudnn.so.x.x.x libcudnn.so.x  
    sudo ln -sf libcudnn.so.x libcudnn.so  
    sudo ldconfig  
~~~
   
## 安装caffe-master
	https://github.com/BVLC/caffe  
 - 编译  
 ~~~
    cp Makefile.config.example Makefile.config      
    vim Makefile.config  
    	启用cuDNN,去掉'#' USE_CUDNN :=1
    	INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/lib/x86_64-linux-gnu/hdf5/serial/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial
	export PATH=/usr/local/cuda/bin:$PATH
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    make all -jx  
    make test -j16  
    make runtest -jx  
    make pycaffe -jx 
~~~
## 常见问题
 - 1.ImportError: No module named skimage.io
~~~
	pip install scikit-image
~~~
- 2.ImportError: No module named google.protobuf
~~~
	sudo pip install protobuf
~~~
3.ImportError: No module named caffe
~~~
	export PYTHONPATH=/work/project/caffe/python:$PYTHONPATH
~~~
4. ImportError: libcudart.so.8.0: cannot open shared object file: No such file or directory
~~~
	export PYTHONPATH=/work/project/caffe/python:$PYTHONPATH
	1. sudo ldconfig /usr/local/cuda-8.0/lib64
	2.LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64。如果仍然不行，再尝试执行:
	export PATH=$PATH:/usr/local/cuda-8.0/bin
	export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-8.0/lib64
	source /etc/profile
~~~    