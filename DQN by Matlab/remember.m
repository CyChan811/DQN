function remember(current_state, action, reward, next_state, done)
   global memory_count1;
   global memory_count2;
   global memory_size1;
   global memory_size2;
   global memory1;
   global memory2;
   transition = cell(1, 5);
   transition = {current_state, action, reward, next_state, done};  %��¼״̬
   index1 = rem(memory_count1, memory_size1) + 1;
   index2 = rem(memory_count2, memory_size2) + 1;
   %������������, �͸���������
   if reward > 0    %�洢����������
      memory1{index1,1} =  transition{1,1};
      memory1{index1,2} =  transition{1,2};
      memory1{index1,3} =  transition{1,3};
      memory1{index1,4} =  transition{1,4};
      memory1{index1,5} =  transition{1,5};
      memory_count1 = memory_count1 + 1;
   else  %�洢���������㽱������
      memory2{index2,1} =  transition{1,1};
      memory2{index2,2} =  transition{1,2};
      memory2{index2,3} =  transition{1,3};
      memory2{index2,4} =  transition{1,4};
      memory2{index2,5} =  transition{1,5};
      memory_count2 = memory_count2 + 1;
   end
end