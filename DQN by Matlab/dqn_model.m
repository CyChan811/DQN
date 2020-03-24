function layers = dqn_model()
   %% 创建神经网络
    %包括两个卷积层,两个全连接层
    %图层数组
    global map;
    layers = [     
        imageInputLayer([size(map,1),size(map,2),1], 'Name', 'input') % 建立图像输入层

        convolution2dLayer(3, 16, 'Name', 'con1')   %卷积层1  卷积核大小为5x5,卷积核数目为16
        reluLayer('Name','relu1') %激活函数

        convolution2dLayer(3, 32, 'Name', 'con2')  %%卷积层2  卷积核大小为5x5,卷积核数目为32
        reluLayer('Name','relu2')    %激活函数

        fullyConnectedLayer(64, 'Name', 'fc1') %全连接层  有 64 个神经元的输出
        reluLayer('Name','relu3')  %激活函数

        fullyConnectedLayer(4, 'Name', 'fc2') %输出每个动作的Q值
        regressionLayer('Name', 'output') %回归层
%          softmaxLayer
%          classificationLayer('Name', 'classification')
        ];

%     model = layerGraph;    %创建空网络图层
%     model = addLayers(model,layers);  %按图层数组的顺序加入网络图层架构中  
end