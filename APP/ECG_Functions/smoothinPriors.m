function varargout = smoothinPriors(hrv_reSampled,lamda,fs)
%smoothinPriors Summary: This function takes raw HRV data, lamda, and 
% re-sampling rate as input and produces a detrended version thereof for 
% proper time and frequency (mostly frequency) analysis. The detrend 
% function works almost like a time-varying FIR Highpass Filter. lamda is 
% the only parameter that needs to be adjusted and it determines the cutoff 
% frequency of the filter.
% Default lamda=500 -> cutoff=0.035 .

% Detrend Algorithm:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = length(hrv_reSampled);
I = speye(T);
D2 = spdiags(ones(T-2,1)*[1 -2 1],0:2,T-2,T);
hrv_stat = (I-inv(I+lamda^2*(D2')*D2))*hrv_reSampled; %% Detrended/(almost)-Stationary HRV Data.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trend = hrv_reSampled - hrv_stat; % Trend line
temp = (I-inv(I+lamda^2*(D2')*D2));
temp =full(temp(fix(length(hrv_reSampled)/2),:));
[h,w] = freqz(temp,1,2^12,fs); % Frequency Response.

varargout{1} = hrv_stat;
varargout{2} = trend;
varargout{3} = h;
varargout{4} = w;
end