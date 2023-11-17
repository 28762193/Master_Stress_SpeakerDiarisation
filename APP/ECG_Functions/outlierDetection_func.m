function Sig = outlierDetection_func(Sig,desired_wnd,Sens_Type)
%outlierDetection_func: This function acts as an outlier detector and
%corrector. The outlier detector is based on IQR ranges and uses the same
%principle as a simple moving average window to progress through the data.
%Each data point gets its own IQR ranges. The window size is adjusted for
%the end points where the desired window size cannot be met.

% Sens_Type = 1 -> For HRV data correction; uses a simple average of the
% point before and after the outlier.
% Sens_Type = 2 -> For Acc data correction; uses a more complex average and
% std as to ensure that a group of consecutive outliers does not influence
% the correction.

%% Outlier Detection and Removal:
%desired_wnd = 8;
n = size(Sig,1);
chg_point1 = (desired_wnd+1)/2;
chg_point2 = n-((desired_wnd+1)/2)+1;
wnd_size = 0;
q1Acc = zeros(size(Sig));
q3Acc = zeros(size(Sig));
iqrAcc = zeros(size(Sig));
LB_Acc = zeros(size(Sig));
UB_Acc = zeros(size(Sig));
for i=1:size(Sig,2)
    for j=1:n
        if j == 1
            wnd_size = 2;
        elseif j < chg_point1
            wnd_size = j*2-1;
        elseif j == n
            wnd_size = 2;
        elseif j > chg_point2
            wnd_size = (n-j)*2+1;
        else
            wnd_size = desired_wnd;
        end
        % Get Boundary Limits:
        if j ~= n
            range_begin = j-floor((wnd_size-1)/2);
            range_end = j+ceil((wnd_size-1)/2);
        else
            range_begin = j-ceil((wnd_size-1)/2);
            range_end = j+floor((wnd_size-1)/2);
        end
        tempSort = sort(Sig(range_begin:range_end,i));
        q1Acc(j,i) = q1_func(tempSort);
        q3Acc(j,i) = q3_func(tempSort);
        iqrAcc(j,i) = q3Acc(j,i) - q1Acc(j,i);
        LB_Acc(j,i) = q1Acc(j,i) - 3*iqrAcc(j,i);
        UB_Acc(j,i) = q3Acc(j,i) + 3*iqrAcc(j,i);

        % Check whether the current value exceeds its boundaries:
        if Sig(j,i) < LB_Acc(j,i) || Sig(j,i) > UB_Acc(j,i)
            % Change outlier to its mean value of all previous values within
            % the window ± its std:
            if Sens_Type == 1 % HRV
                Sig(j,i) = (Sig(j-1,i) + Sig(j+1,i))*0.5;
            elseif Sens_Type == 2 % Acc
                Sig(j,i) = mean(Sig(range_begin:j-1,i)) + (-1)^(randi([1 2]))*(std(Sig(range_begin:j-1,i)));
            end
            
        end
    end
end
%EOF
end