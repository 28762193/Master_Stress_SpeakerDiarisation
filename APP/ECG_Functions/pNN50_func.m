function [pNN50] = pNN50_func(inputSig)
%pNN50_func: Provides the pNN50 time-domain feature based upon the input
%siganl.

% Calculate the difference between consecutive RR-intervals:
rr_diff = zeros(length(inputSig)-1,1);

for i=2:length(inputSig)
    rr_diff(i-1) = inputSig(i)-inputSig(i-1);
end

% Number of RR intervals that differ more than 50 ms. (NN50) [count]:
count_rr_diff50 = length(rr_diff(abs(rr_diff)>=0.05)); 

% Number of RR intervals that differ more than 50 ms divided by the total 
% number of consecutive RR-intervals. (NN50/#RR-interval) (pNN50) [%]:
pNN50 = count_rr_diff50/length(rr_diff) * 100; 

end