function [RMSSD] = rmssd_func(inputSig)
%rmssd_func: Provides the RMSSD value of the input signal.

rr_diff = zeros(length(inputSig)-1,1);

for i=2:length(inputSig)
    rr_diff(i-1) = inputSig(i)-inputSig(i-1);
end
RMSSD = rms(rr_diff*1000); % RMSSD of the entire HRV sequence in ms.

end