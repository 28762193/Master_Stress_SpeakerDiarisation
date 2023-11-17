function q3Acc = q3_func(arr_Sort)
% q3_func: get q3 value from sorted array:
    n1 = length(arr_Sort);
    q3_pointer = floor(((n1+1)*3)/4);
    if mod(((n1+1)*3),4) == 0
        q3Acc = arr_Sort(q3_pointer);
    elseif q3_pointer == n1
        q3Acc = arr_Sort(2);
    else
        q3Acc = (arr_Sort(q3_pointer)+arr_Sort(q3_pointer+1))/2;
    end
end