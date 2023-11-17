function [totalAcc,bodyMovementAcc,gravityAcc,timestamps,Fs] = preProcessingAcc_func(rawAccData,timestamps,Fs)
%preProcessingAcc_func: The function accepts the raw accelerometer data and
%sampling frequency for preprocessing. The preprocessing entails
%downsampling, outlier detection (based on moving window) and removal with 
%interpolation, median filter to reduce noise, 3rd order lowpass butterworth 
%fitler with a 20 Hz cutoff as most bodymovements are below 15 Hz, moving 
%average filter (MAF) for baseline wondering removal (as the filter acts as 
%a FIR highpass filter) therby creating the bodymovement signal, gravity 
%signal creation by removing bodymovement from the raw siganl, and lastly 
%checking results via visual aid.

%% Filter/Pre-processing:
% Test Accelerometer on Smartphone Data:
% DownSample Data if Fs > 50 Hz
% Filtering: Total_Acc_Data = Body_Movement + Gravity + Noise
% We want linearAcceleration (Body_Movement): Bmove =_appr hpf(Total_Acc_data);
% cutoff Freq = 0.3 Hz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Down Sample Data with Cubic Spline:
if Fs > 50
    t_old = (0:1/Fs:(length(rawAccData)-1)/Fs)';
    Fs = 50; % New Sampling Rate - 50 Hz
    t_new = (0:1/Fs:t_old(end))';
    timestamps = seconds(t_new)+timestamps(1);
    rawAccData_ReSamp = zeros(length(t_new),size(rawAccData,2));
    for j=1:size(rawAccData_ReSamp,2)
        rawAccData_ReSamp(:,j) = spline(t_old,rawAccData(:,j),t_new);
    end
    rawAccData = rawAccData_ReSamp;
    clear t_new t_old rawAccData_ReSamp
end

%% Outlier Detection and Removal:

totalAcc = rawAccData;

desired_wnd = 129;
n = size(totalAcc,1);
chg_point1 = (desired_wnd+1)/2;
chg_point2 = n-((desired_wnd+1)/2)+1;
q1Acc = zeros(size(totalAcc));
q3Acc = zeros(size(totalAcc));
iqrAcc = zeros(size(totalAcc));
LB_Acc = zeros(size(totalAcc));
UB_Acc = zeros(size(totalAcc));
for i=1:size(totalAcc,2)
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
        tempSort = sort(totalAcc(range_begin:range_end,i));
        q1Acc(j,i) = q1_func(tempSort);
        q3Acc(j,i) = q3_func(tempSort);
        iqrAcc(j,i) = q3Acc(j,i) - q1Acc(j,i);
        LB_Acc(j,i) = q1Acc(j,i) - 3*iqrAcc(j,i);
        UB_Acc(j,i) = q3Acc(j,i) + 3*iqrAcc(j,i);

        % Check whether the current value exceeds its boundaries:
        if totalAcc(j,i) < LB_Acc(j,i) || totalAcc(j,i) > UB_Acc(j,i)
            % Change outlier to its mean value of all previous values within
            % the window ± its std:
            totalAcc(j,i) = mean(totalAcc(range_begin:j-1,i)) + (-1)^(randi([1 2]))*(std(totalAcc(range_begin:j-1,i)));
        end
    end
end

clear q1Acc q3Acc iqrAcc LB_Acc UB_Acc desired_wnd wnd_size range_begin 
clear range_end chg_point1 chg_point2 n

%% Create Filter objects and totalAcc signal:
lpfiltBW = lpbutw_order3_cf20(Fs); % 3-order Lowpass Butterworth filter 
% with 20 Hz cutoff frequency as most body movements are below 15 Hz.
% hpfiltCheb2 = hpfilter(Fs); % Highpass Chebyshev Type 2 filter with 0.3 Hz 
% cutoff frequency as to remove gravitional forces. MATCH EXACTLY @
% stopband.

totalAcc = medfilt1(totalAcc); % Median Filter to reduce noise.
totalAcc = filtfilt(lpfiltBW.sosMatrix,lpfiltBW.ScaleValues,totalAcc);  

clear lpfiltBW;

%% Adaptive Mean Filtering for Baseline Drift:
wind_length = floor(Fs*1.28); % 1.28 s window size for 50 Hz sampling rate - provides a 0.34 Hz cutoff

bodyMovementAcc = totalAcc - movmean(totalAcc,wind_length); % MAF FIR filter with a 0.34 Hz cutoff
gravityAcc = movmean(totalAcc,wind_length);   %totalAcc - bodyMovementAcc;

%% Check Frequency Response of Moving Average Filter:

% cutoffMA = (0.442947/sqrt(wind_length^2 - 1)) * Fs; % Aprroximation to 
% moving average cutoff frequency in Hz. Cutoff at -3.01 dB

% Show frequency response of moving average filter:
% a = zeros(wind_length,1);
% b = ones(wind_length,1);
% a(1,1) = wind_length;
% 
% [h,w] = freqz(b,a,2^12);
% w = (w*Fs)/(2*pi);
% figure('Name','MAF Frequency Response');
% subplot(2,1,1);
% plot(w,20*log10(abs(h)))
% ylabel('Magnitude (dB)');
% xlabel('Frequency (Hz)');
% title('Magnitude Response');
% subplot(2,1,2);
% plot(w,180*angle(h)/pi)
% title('Phase Response');
% ylabel('Phase (deg)');
% xlabel('Frequency (Hz)');
% grid on;
% clear h w a b

% fprintf('Cutoff Frequency: %0.4f Hz\n',cutoffMA);

%% Plot Filter comparison:
% figure('Name','Filter Comparison: Time-Domain');
% plot(rawAccData(1:end,1));
% hold on;
% grid on;
% plot(totalAcc(1:end,1));
% plot(bodyMovementAcc(1:end,1));
% legend('rawAcc','totalAcc','bodyMoveAcc');
% title('Accelerometer Data (X-Axis)');
% xlabel('Samples');
% ylabel('Acceleration [g=9.80665 m/s^2]');

%% PSD of Filter:
% figure('Name','Filter Comparison2: PSD');
% Y = [fft(rawAccData(1:end,1)) fft(bodyMovementAcc(1:end,1)) fft(gravityAcc(1:end,1))];
% L = [length(rawAccData(1:end,1)) length(bodyMovementAcc(1:end,1)) length(gravityAcc(1:end,1))];
% for j=1:size(Y,2)
%     P2 = abs(Y(1:end,j)/L(j));
%     P1 = P2(1:L(j)/2+1);
%     P1(2:end-1) = 2*P1(2:end-1);
% 
%     f = Fs*(0:(L(j)/2))/L(j);
%     plot(f,P1);
%     hold on;
% end
% title('Single-Sided Amplitude Spectrum of Acc (X-Axis)');
% xlabel('Frequency [Hz]');
% ylabel('Magnitude');
% % legend('rawAcc','bodyMoveAcc');
% legend('rawAcc','bodyMoveAcc','gravityAcc');
% ylim([0 0.2]);
% clear P1 P2 Y L f;

end