function state = matrix_to_img(row, col)
%% ����ͼ����תΪͼƬ��ʽ��shape(height,width,channel)
    global POS_VALUE;
    global map;
    state = map;
    state(row,col) = POS_VALUE;
    state = reshape(state,[size(state,1),size(state,2),1]); %ά��ת��
end