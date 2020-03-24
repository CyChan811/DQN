function state = matrix_to_img(row, col)
%% 将地图矩阵转为图片格式的shape(height,width,channel)
    global POS_VALUE;
    global map;
    state = map;
    state(row,col) = POS_VALUE;
    state = reshape(state,[size(state,1),size(state,2),1]); %维数转换
end