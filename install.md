# `Caffe安装步骤`

服务器配置和Caffe配置
## 服务器配置

 - [服务器配置参考](https://www.cnblogs.com/banju/p/7918895.html) 
 - GPU安装  
    1.关闭bios里的secure boot(否则无法使用显卡驱动)  
    2.设置----软件和更新----附加驱动----使用NVIDIA....(专有)----应用更改----（之后为半小时的等待）-----提示重新启动。  
    3.检测显卡驱动是否安装成功 nvidia-smi  
    4.安装依赖关系包  
         sudo apt-get install vim python-pip git    
         sudo apt-get install build-essential libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler  
        sudo apt-get install libopenblas-dev liblapack-dev libatlas-base-dev libgflags-dev libgoogle-glog-dev liblmdb-dev
        sudo apt-get install libboost-all-dev
        sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev

## 安装CUDA
 -下载安装文件。首先去英伟达官网下载cuda安装包
 	https://developer.nvidia.com/cuda-toolkit-archive
-安装包
	sudo sh ./cuda_8.0.61_375.26_linux.run
-检测安装
	sudo nvcc -V
	cuda samplecd /usr/local/cuda-7.5/samples/1_Utilities/deviceQuery
	make -j4
	./deviceQuery
	