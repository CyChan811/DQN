function action = choose_action(~, state, target_state_pos)
    %选择动作，epsilon为动作探索阈值
    global dqn_net;
    global epsilon;
    global POS_VALUE;
    [row,col] = find(state == POS_VALUE);
    if rand() < epsilon
        [pf_f, pf_action] = potential_field(row, col, target_state_pos);
        pf_add_num = int8(pf_f);
        choose_num = round(rand() * (100 + pf_add_num));
        if (0 <= choose_num) && (choose_num < 25)
            action = 1;
        elseif (25 <= choose_num) && (choose_num < 50)
            action = 2;
        elseif (50 <= choose_num) && (choose_num < 75)
            action = 3;
        elseif (75 <= choose_num) && (choose_num < 100)
            action = 4;
        else
            action = pf_action;
        end
    else
        act_values = predict(dqn_net, state);
        action = find(act_values == max(act_values));   %取最大值
        %最大值索引可能有多个，取第一个索引
        action = action(1);
    end
end