# DQN by Matlab
DQN实现避障控制
###实现环境：
Matlab 2019a及以上，因为使用了deeplearning toolbox搭建网络
###结果
小地图：  
![](./image/simplemap.jpg)  

稍复杂点的地图：  
![](./image/bigmap1.jpg)  

![](./image/bigmap2.jpg)
  

#DQN by Python
使用Pytorch框架搭建神经网络

##性能对比（episode-step图）  
传统DQN算法：  
![](./image/oldDQN.jpg)


基于优先级采样的DQN算法：  
![](./image/optimizedDQN.jpg)

DQN + 人工势场：
![](./image/DQN_with_potential_field.jpg)