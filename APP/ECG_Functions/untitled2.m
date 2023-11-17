
hrv = cell(size(ECG_Cell));
hrv_time = cell(size(ECG_Cell));
ECGStress_Fit = cell(size(ECG_Cell));
stressTime_Regions = cell(size(ECG_Cell));

for ii=1:size(ECG_Cell,1)
    %% Rpeak Detection and HRV creation:
    % ECG_Cell{ii} = ECG_Cell{ii}/1000; % ECG uV->mV
    % Fs = round(1/mean(seconds(diff(ecgTimestamps{ii}))),2);

    %% Detrend ECG:
    % Adaptive Mean Filtering for Baseline Drift:
    wind_length = floor(Fs*0.35); % 0.35 s window size for 130 Hz sampling rate - provides a 1.266 Hz cutoff

    ECG_Cell{ii} = ECG_Cell{ii} - movmean(ECG_Cell{ii},wind_length); % MAF FIR filter with a 1.266 Hz cutoff
    % Trim Edges:
    ecgTimestamps{ii}(end-Fs+1:end) = [];
    ECG_Cell{ii}(end-Fs+1:end) = [];
    ecgTimestamps{ii}(1:Fs) = [];
    ECG_Cell{ii}(1:Fs) = [];
    %% Resample ECG:
    if Fs~=360 % Resample
        t = ((0:size(ECG_Cell{ii},1)-1)/Fs)';
        Fs = 360;
        t_new = (0:1/Fs:t(end))';
        ECG_Cell{ii} = makima(t,ECG_Cell{ii},t_new);
        ecgTimestamps{ii} = ecgTimestamps{ii}(1)+seconds(t_new);
    end
    %% Check Frequency Response of Moving Average Filter:
    %
    % cutoffMA = (0.442947/sqrt(wind_length^2 - 1)) * Fs; % Aprroximation to
    % % moving average cutoff frequency in Hz. Cutoff at -3.01 dB
    %
    % % Show frequency response of moving average filter:
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
    %
    % fprintf('Cutoff Frequency: %0.4f Hz\n',cutoffMA);

    %% Rpeak Detection:
    Rpeak_Loc = RpeakDetect(ECG_Cell{ii},Fs);

    % HRV Creation
    hrv{ii} = zeros(length(Rpeak_Loc)-1,1);
    for jj=1:length(Rpeak_Loc)-1
        hrv{ii}(jj) = Rpeak_Loc(jj+1)-Rpeak_Loc(jj);
    end

    % HRV Correction:
    hrv{ii} = outlierDetection_func(hrv{ii},16,1); % For HRV data - Sens_Type=1

    % HRV timestamps:
    hrv_time{ii} = zeros(size(hrv{ii}));
    hrv_time{ii}(1) = hrv{ii}(1);
    for jj=2:length(hrv_time{ii})
        hrv_time{ii}(jj) = hrv_time{ii}(jj-1) + hrv{ii}(jj);
    end

    %% Re-sample HRV Data with Cubic Spline Interpolation - To equally space the data
    re_sampleRate = 4;
    xx = (0:1/re_sampleRate:hrv_time{ii}(end))';
    hrv_equid = spline(hrv_time{ii},hrv{ii},xx);

    %% Detrending HRV Data with Smoothin Priors

    lamda = 500; % Default value - provides an cutoff frequency of 0.035 Hz
    [hrv_stat,trend,~,~] = smoothinPriors(hrv_equid,lamda,re_sampleRate);

    %% Create 5minute HRV time frames:

    wind_size = 5*60; % 5 Minute window in seconds.
    overlay_size = 1 - 0.75; % 75% overlap.
    HRV_Frame_Index = hrvSegmentation(hrv_time{ii},wind_size,overlay_size);

    % Segment HRV:
    hrv_Segment = cell(size(HRV_Frame_Index,1),1);

    % SUBTRACT the BASELINE (mean of the Moving Average):
    hrv_MinusBaseline = hrv{ii} - mean(trend);

    for jj=1:size(HRV_Frame_Index,1)
        hrv_Segment{jj} = hrv_MinusBaseline(HRV_Frame_Index(jj,1):HRV_Frame_Index(jj,2));
    end
    
    % Change HRV Time to datetime:
    hrv_time{ii} = ecgTimestamps{ii}(1)+seconds(hrv_time{ii});
    %% Time-Domain Analysis:
    % Variables of Interest:
    % mean_RR - The Average of the RR-intervals. [ms]
    % RMSSD - The Square root of the mean squared differences between
    %         successive RR intervals. [ms]
    % pNN50 - Number of successive RR interval pairs that differ more than 50
    %         ms (NN50) divided by the total number of RR intervals. [%]
    % peak-valley - Time-domain Filter dynamically centered at the exact
    %               ongoing respiratory frequency.

    % Time Analysis does not require re-sampling or detrending. (Task Force)
    mean_RR = zeros(size(hrv_Segment));
    skew_RR = zeros(size(hrv_Segment));
    RMSSD = zeros(size(hrv_Segment));
    pNN50 = zeros(size(hrv_Segment));

    % Ranges: 1:278 - non-stress; 1492:1913 - stress.
    for jj=1:size(hrv_Segment,1)
        %     x = histogram(hrv{ii},25,'BinLimits',[0.25 1.75]); % Show RR distribution.
        mean_RR(jj) = mean(hrv_Segment{jj})*1000; % mean RR interval in ms.
        skew_RR(jj) = skewness(hrv_Segment{jj}); % Negative skewed equals non-stress and positive skewed equals stress.
        RMSSD(jj) = rmssd_func(hrv_Segment{jj}); % Provides the RMSSD value of the entire HRV signal
        pNN50(jj) = pNN50_func(hrv_Segment{jj}); % Provide the pNN50 value of the entire HRV signal
    end

    %% Get hrv segments for stationary-equidistance hrv samples:

    % Segment hrv for spectral analysis:
    HRV_Frame_Index_stat = hrvSegmentation(xx,wind_size,overlay_size);

    % Segment HRV:
    hrv_stat_Segment = cell(size(HRV_Frame_Index_stat,1),1);

    for jj=1:size(HRV_Frame_Index_stat,1)
        hrv_stat_Segment{jj} = hrv_stat(HRV_Frame_Index_stat(jj,1):HRV_Frame_Index_stat(jj,2));
    end

    %% Spectral Analysis:

    m_order = 16; % Model-order - The prefered order for HRV data is 16.
    range_VLF = 0.04; % Less than 0.04 Hz
    range_LF = 0.15; % Between 0.04 and 0.15 Hz
    range_HF = 0.4; % Between 0.15 and 0.4 Hz

    pxx = cell(size(hrv_stat_Segment));
    f = cell(size(hrv_stat_Segment));
    VLF = zeros(size(hrv_stat_Segment));
    LF = zeros(size(hrv_stat_Segment));
    HF = zeros(size(hrv_stat_Segment));
    TP = zeros(size(hrv_stat_Segment));
    LFHF_ratio = zeros(size(hrv_stat_Segment));

    for jj=1:size(hrv_stat_Segment)
        [pxx{jj},f{jj},VLF(jj),LF(jj),HF(jj),TP(jj),LFHF_ratio(jj)] = spectralAnalysis(hrv_stat_Segment{jj},m_order,re_sampleRate,range_VLF,range_LF,range_HF);
    end

    % % Plot Spectral Analysis:
    % figure('Name','Spectral Analysis Using AR Modelling');
    % plot(f,pxx{1});
    % title('PSD with AR Modelling');
    % ylabel('PSD [s^2/Hz]');
    % xlabel('Frequency [Hz]');
    % legend('pburg PSD estimate');

    %% Create Feature Matrix:
    feat_ECG_Stress = [mean_RR,RMSSD,pNN50,skew_RR,LF,HF,LFHF_ratio];
    %% Classify/Identify Stress:
    % Load ECG-Stress Model:
    load("ECG_Stress_MLModel.mat");
    % Predict:
    ECGStress_Fit{ii} = ECG_Stress_MLModel.predictFcn(feat_ECG_Stress);
    % Time Regions:
    stressTime_Regions{ii} = NaT(size(ECGStress_Fit{ii},1),2);
    stressTime_Regions{ii}(:,1) = hrv_time{ii}(HRV_Frame_Index(:,1));
    stressTime_Regions{ii}(:,2) = hrv_time{ii}(HRV_Frame_Index(:,2));
    %% Plot Stress and NonStress Regions:
%     if ~isempty(ECGStress_Fit{ii})
%         temp = zeros(size(ECGStress_Fit{ii},1),3);
%         temp(:,3) = ECGStress_Fit{ii}(:,end);
%         temp(:,1) = hrv_time{ii}(HRV_Frame_Index(:,1));
%         temp(:,2) = hrv_time{ii}(HRV_Frame_Index(:,2));
%         ECGStress_Fit{ii} = temp;
%         clear temp
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         %% Plot Results versus Time:
%         regions = ECGStress_Fit{ii}(:,1:2);
%         idxClrs = ECGStress_Fit{ii}(:,3)+1;
%         chgPnt1 = ischange(idxClrs);
%         chgPnt2 = find(chgPnt1)-1;
%         chgPnt1(1) = 1; chgPnt2(end+1) = size(regions,1);
%         chgPnt1 = find(chgPnt1);
%         idxClrs = ECGStress_Fit{ii}(chgPnt1,3)+1;
% 
% 
%         xCoords = [regions(chgPnt1,1) regions(chgPnt2,2) regions(chgPnt2,2) regions(chgPnt1,1)];
%         yCoords = zeros(size(xCoords,1),4);
%         yCoords(:,1:2) = min(hrv{ii});
%         yCoords(:,3:4) = max(hrv{ii});
% 
%         clrs = {[0 0.4470 0.7410], [0.8500 0.3250 0.0980]}; % Blue and Red - Nonstress and Stress.
% 
%         figure;
%         hESLine = plot(hrv_time{ii},hrv{ii},'Color','Black'); % Plot HRV versus time (hrv-time)
%         hESAx = ancestor(hESLine,'axes');
%         hold on;
%         for jj=1:size(xCoords,1) % Plot regions
%             patch(hESAx,xCoords(jj,:),yCoords(jj,:),clrs{idxClrs(jj,1)},'FaceAlpha',.3,'EdgeColor',clrs{idxClrs(jj,1)},'LineWidth',1);
%         end
%     end
%EOL
end
