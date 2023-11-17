function [varargout] = RpeakDetect(Sig,Fs)
% RpeakDetect() takes a raw ECG signal and Sampling Rate as
% inputs and provides the HRV data in [seconds] as the output. The evaluation
% parameters (Sensitivity, PPV, DER) can also be outputted.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform Noise Suppression using SWT:

wname = 'sym4'; % Mother Wavelet 
L = 10; % Level of decompistion.
N = length(Sig);

% Add data to signal:
temp = ceil(N/2^L);
newLength = temp*2^L;
newSig = zeros(newLength,1);
newSig(1:N) = Sig;
newSig(N:end) = Sig(1:newLength-N+1);

% Perform SWT:
s_wt = swt(newSig,L,wname);
% Reconstruct ECG(sort of) using only detail coefficients 4 and 5:
re_swt = zeros(size(s_wt));
re_swt(4,:) = s_wt(4,:);
re_swt(5,:) = s_wt(5,:);
reconSig = iswt(re_swt,wname)';

% Remove the extra data:
reconSig = reconSig(1:N,1);
% Delete unneeded variables:
clear newSig newLength s_wt re_swt temp L;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Enhance Signal:
% Non-linear transform of the signal:
En_reconSig = sign(reconSig).*(reconSig.^2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Zero-Crossing: Addition of High-Frequency Signal

K_func = zeros(size(En_reconSig));
lamda_k = 0.2; % Forgetting factor element of (0,1);
c = 1.1412; % Constant gain
c_adder = norm(En_reconSig)/N; % Constant added to signal.

for i=2:N
    K_func(i,1) = lamda_k*K_func(i-1,1) + (1-lamda_k)*abs(En_reconSig(i,1))*c;
end

b_func = K_func;
for i=1:N-1
    b_func(i+1,1) = ((-1)^i)*K_func(i+1,1) + ((-1)^i)*c_adder;
end

z_func = En_reconSig + b_func; % Addition of high-frequency function to ensure zero-crossing counting

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Zero-Crossing: COUNTING: 0 - No-change, 1 - Change/Crossing

lamda_count = 0.2; % Forgetting Factor
count_func = zeros(size(En_reconSig));
for i=2:N
    count_func(i,1) = abs((sign(z_func(i,1)) - sign(z_func(i-1,1)))/2);
end

% Number of Zero-Crossings Per Segment:
Count_Func = zeros(size(count_func));

for i=2:N
    Count_Func(i,1) = lamda_count*Count_Func(i-1,1) + (1-lamda_count)*count_func(i,1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Zero-Crossing: Adaptive Threshold Q(n):

lamda_q = 0.2;  % Forgetting Factor
Q_func = zeros(size(Count_Func));

for i = 2:N
    Q_func(i,1) = lamda_q*(Q_func(i-1,1)) + (1-lamda_q)*(Count_Func(i,1));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Event Detection: Boundary Creation

boundary_width = 0.05*Fs; % 50 ms Boundary width for QRS-complex.
inbetween_width = 0.25*Fs; % 250 ms width before next possible beat.

% Locate where Count_Func < Threshold (Q_Func):
loc = find(Count_Func<Q_func);
loc_counter = 1; % Holds location index
boundaries_counter = 1;
i = 1; % General counter
boundaries = {};

while loc_counter < length(loc)
    % Make Boundaries:
    % Search within big window (250 ms) for multiple events:
    while loc(i) <= loc(loc_counter) + inbetween_width && i < length(loc)
        i = i+1;
    end
    % Does the events exceed the small window size (50 ms), if so use the first and last events for boundaries:
    if (loc(i-1)+1) - (loc(loc_counter)-1) > (boundary_width)
        boundaries(boundaries_counter,1) = {[floor(loc(loc_counter)) ceil(loc(i-1))]};
    else
        % Use 50 ms boundary:
        temp = boundary_width - (loc(i-1) - loc(loc_counter));
        boundaries(boundaries_counter,1) = {[floor(loc(loc_counter)-temp/2) ceil(loc(i-1)+temp/2)]};
    end
    if boundaries{boundaries_counter}(2) > size(En_reconSig,1)
        boundaries{boundaries_counter}(2) = size(En_reconSig,1);
    end
    if boundaries{boundaries_counter}(1) > 0
        boundaries_counter = boundaries_counter + 1;
    end
    loc_counter = i;
end
clear boundaries_counter boundary_width inbetween_width loc_counter

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% R-peak Detection:

% Find R-peaks in each boundary
m = zeros(size(boundaries));
index = zeros(size(boundaries));
Rpeak_loc = zeros(size(boundaries));
for j = 1:length(boundaries)
    [m(j),index(j)] = max(abs(En_reconSig(boundaries{j,1}(1):boundaries{j,1}(2))));
    Rpeak_loc(j) = boundaries{j,1}(1) + index(j) - 1;
    Rpeak_loc(j) = Rpeak_loc(j)/Fs;
end
clear m index boundaries;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Amplitude threshold on detected R-peaks:

threshold_amp = rms(c*En_reconSig); % The Amplitude Threshold is choosen 
% to be the rms value of the enhanced signal.
i = 1;
n = length(Rpeak_loc);

for j =1:n
    if abs(c*En_reconSig(fix(Rpeak_loc(i)*Fs))) < threshold_amp
        Rpeak_loc(i) = [];
    else 
        i = i+1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

varargout{1} = Rpeak_loc;


end