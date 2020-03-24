% 人工势场,返回引力和动作
function [pf_f, pf_action] = potential_field(y, x, target_state_pos)
    global map;
    pf_maze_size = sqrt(power(size(map, 1), 2) + power(size(map, 2), 2));   %地图斜长度
    pf_k = 0.9; %引力尺度因子
    threshold_value = 3.0;     % 阈值
    current_y = y;   % 当前位置坐标,y是行，x是列
    current_x = x;
    distance = sqrt(power((target_state_pos(1) - current_y), 2) + power((target_state_pos(2) - current_x), 2));
    if distance >= threshold_value
        pf_f = (pf_maze_size - pf_k * distance) * 0.9;  % 引力
    else
        pf_f = (pf_maze_size - pf_k * threshold_value) * 0.9;
    end
    % 根据目标的位置选择动作
    if current_x <= target_state_pos(2) && current_y > target_state_pos(1)
        pf_action = 1;   % 上
    elseif current_x >= target_state_pos(2) && current_y < target_state_pos(1)
        pf_action = 2;   % 下
    elseif current_x > target_state_pos(2) && current_y >= target_state_pos(1)
        pf_action = 3;   % 左
    elseif current_x < target_state_pos(2) && current_y <= target_state_pos(1)
        pf_action = 4;   % 右
    end
end