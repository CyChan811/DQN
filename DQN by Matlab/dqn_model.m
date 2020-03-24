function layers = dqn_model()
   %% ����������
    %�������������,����ȫ���Ӳ�
    %ͼ������
    global map;
    layers = [     
        imageInputLayer([size(map,1),size(map,2),1], 'Name', 'input') % ����ͼ�������

        convolution2dLayer(3, 16, 'Name', 'con1')   %�����1  ����˴�СΪ5x5,�������ĿΪ16
        reluLayer('Name','relu1') %�����

        convolution2dLayer(3, 32, 'Name', 'con2')  %%�����2  ����˴�СΪ5x5,�������ĿΪ32
        reluLayer('Name','relu2')    %�����

        fullyConnectedLayer(64, 'Name', 'fc1') %ȫ���Ӳ�  �� 64 ����Ԫ�����
        reluLayer('Name','relu3')  %�����

        fullyConnectedLayer(4, 'Name', 'fc2') %���ÿ��������Qֵ
        regressionLayer('Name', 'output') %�ع��
%          softmaxLayer
%          classificationLayer('Name', 'classification')
        ];

%     model = layerGraph;    %����������ͼ��
%     model = addLayers(model,layers);  %��ͼ�������˳���������ͼ��ܹ���  
end