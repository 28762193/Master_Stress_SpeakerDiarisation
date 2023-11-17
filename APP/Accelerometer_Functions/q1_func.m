function q1Acc = q1_func(arr_Sort)
% q1_func: get q1 value from sorted array:
    n1 = length(arr_Sort);
    q1_pointer = floor((n1+1)/4);
    if mod(n1+1,4) == 0
        q1Acc = arr_Sort(q1_pointer);
    elseif q1_pointer == 0
        q1Acc = arr_Sort(1);
    else
        q1Acc = (arr_Sort(q1_pointer)+arr_Sort(q1_pointer+1))/2;
    end
end