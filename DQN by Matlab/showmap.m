function showmap(map,start_state_pos,target_state_pos,state_mark)
%% 地图可视化
n_x = size(map,1);
n_y = size(map,2);
b = zeros(n_x,n_y);
for i = 1:n_x
    for j = 1: n_y
        if map(i,j) == 1
           b(n_x + 1 - i,j) = 1; 
        end
    end
end
b(end+1,end+1) = 0;
figure;
colormap([1 1 1;0 0 0])
pcolor(b); % 赋予栅格颜色
set(gca,'XTick',10:10:n_x,'YTick',10:10:n_y);  % 设置坐标
axis image xy
 
text(start_state_pos(2) + 0.3,n_x + 1 - start_state_pos(1) +0.75,'START','Color','red','FontSize',5);%显示start字符
text(target_state_pos(2) + 0.3,n_x + 1 - target_state_pos(1) + 0.75,'GOAL','Color','red','FontSize',5);%显示goal字符
 
hold on
for i = 1:size(state_mark,1)
    scatter(state_mark(i,2) + 0.5,n_x + 1 - state_mark(i,1) +0.5,20 ,[0 0 0]);%显示轨迹
    if i < size(state_mark,1)
        if (state_mark(i,1) - state_mark(i+1,1)) == 0
            scatter((state_mark(i,2) + state_mark(i+1,2)) / 2 + 0.5,n_x + 1 - state_mark(i,1) +0.5,20,[0 0 0]);%显示轨迹
        elseif (state_mark(i,2) - state_mark(i+1,2)) == 0
            scatter(state_mark(i,2) + 0.5,n_x + 1 - (state_mark(i,1) + state_mark(i+1,1)) / 2 +0.5,20,[0 0 0]);%显示轨迹
        end
    end
    hold on
end
%pin strat goal positon
scatter(start_state_pos(2) + 0.5,n_x + 1 - start_state_pos(1) +0.5,'MarkerEdgeColor',[1 0 0],'MarkerFaceColor',[1 0 0], 'LineWidth',1);%start point
scatter(target_state_pos(2) + 0.5,n_x + 1 - target_state_pos(1) + 0.5,'MarkerEdgeColor',[0 1 0],'MarkerFaceColor',[0 1 0], 'LineWidth',1);%goal point
 
hold on
end