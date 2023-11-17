% WESAD_Stress_To_MAT: This script creates the ECG-Stress dataset from the
% WESAD ECG signals. HRV Features are obtained as well as its corresponding
% stress label and are saved as a table in a .mat file. Referencing to the
% WESAD dataset:
% =========================================================================
% Philip Schmidt, Attila Reiss, Robert Dürichen, Claus Marberger, and 
% Kristof Van Laerhoven. 2018. Introducing WESAD, a Multimodal Dataset for 
% Wear- able Stress and Affect Detection. In 2018 International Conference 
% on Multi- modal Interaction (ICMI ’18), October 16–20, 2018, Boulder, CO, 
% USA. ACM, New York, NY, USA, 9 pages. 
% https://doi.org/10.1145/3242969.3242985
% =========================================================================

% Main Algorithm: Emotion Disregulation (i.e. Stress) Classification
% Use of WESAD Dataset.
clear all;
%% Load Data:
% The data is represented by a table consisting of ECG, XYZ acceleration,
% and respiration data.

fileLoc = '/Datasets/WESAD/MAT_Files/';
cd(fileLoc);
fileList = dir('S*.mat');
fileList = {fileList.name}';
fileName = strcat(fileLoc,fileList);

fileLoc = '/Datasets/WESAD/Subject_Info/';
cd(fileLoc);
fileList = dir('S*.txt');
fileList = {fileList.name}';
fileName_readme = strcat(fileLoc,fileList);

% Move to current running script:
filePath = matlab.desktop.editor.getActiveFilename;
cd(fileparts(filePath));

clear fileList fileLoc;
%%
for index=1:size(fileName,1)
    load(fileName{index});

    %% Rpeak Detection and HRV creation:

    Fs = 700;
    % Rpeak Detection:
    Rpeak_Loc = RpeakDetect(raw_data.ECG,Fs);

    % HRV Creation:
    hrv = zeros(length(Rpeak_Loc)-1,1);
    for i=1:length(Rpeak_Loc)-1
        hrv(i) = Rpeak_Loc(i+1)-Rpeak_Loc(i);
    end

    % HRV Correction:
    hrv = outlierDetection_func(hrv,16,1); % For HRV data - Sens_Type=1

    % HRV timestamps: Important for creating 5 minute HRV segments for
    % feature extraction.
    hrv_time = zeros(size(hrv));
    for i=2:length(hrv_time)
        hrv_time(i) = hrv_time(i-1) + hrv(i);
    end
    
    %% Re-sample HRV Data with Cubic Spline Interpolation - To equally space the data
    re_sampleRate = 4;
    xx = (0:1/re_sampleRate:hrv_time(end))';
    hrv_equid = spline(hrv_time,hrv,xx);

    %% Detrending HRV Data with Smoothin Priors

    lamda = 500; % Default value - provides an cutoff frequency of 0.035 Hz
    [hrv_stat,trend,h,w] = smoothinPriors(hrv_equid,lamda,re_sampleRate);

    %% Create 5minute HRV time frames:
    
    wind_size = 5*60; % 5 Minute window in seconds.
    overlay_size = 1 - 0.5;
    HRV_Frame_Index = hrvSegmentation(hrv_time,wind_size,overlay_size);
 
    % Segment HRV:
    hrv_Segment = cell(size(HRV_Frame_Index,1),1);
    
    % SUBTRACT the BASELINE (mean of the Moving Average): 
    hrv_MinusBaseline = hrv - mean(trend);

    for i=1:size(HRV_Frame_Index,1)
        hrv_Segment{i} = hrv_MinusBaseline(HRV_Frame_Index(i,1):HRV_Frame_Index(i,2));
    end

    %% Getting labels:
    % Lables will have to be averaged - judgement call
    % Rpeak_Loc is used here as it gives clear timing with regards to the
    % original ECG data. hrv_time is merely used for hrv features.
    % 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation

    [HRV_Frame_Index,hrv_Segment,feature_Labels,removal_Index] = getHRV_Labels_func(Rpeak_Loc,hrv,raw_data.Labels,HRV_Frame_Index,hrv_Segment,Fs);


    %% Time-Domain Analysis:
    % Variables of Interest:
    % mean_RR
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

    %short_term_window = length(hrv); % Usually 5-minutes.
    % Ranges: 1:278 - non-stress; 1492:1913 - stress.
    for i=1:size(hrv_Segment,1)
        %x = histogram(hrv,25,'BinLimits',[0.25 1.75]); % Show RR distribution.
        mean_RR(i) = mean(hrv_Segment{i})*1000; % mean RR interval in ms.
        skew_RR(i) = skewness(hrv_Segment{i}); % Negative skewed equals non-stress and positive skewed equals stress.
        RMSSD(i) = rmssd_func(hrv_Segment{i}); % Provides the RMSSD value of the entire HRV signal
        pNN50(i) = pNN50_func(hrv_Segment{i}); % Provide the pNN50 value of the entire HRV signal
    end


    %% Plot Frequency Response of the Smmothin Priors Filter:
    % Plot Frequency Response (testing lamda value: lamda determines the cutoff
    % frequency)
    % figure('Name','Smoothin Prior FIR High Pass Filter Frequency Response');
    % plot(w,db(h));
    % title('Frequency Response');
    % ylabel('Magnitude [db]');
    % xlabel('Frequency [Hz]');
    % legend('lamda\_500');
    % clear h w;

    %% Plot Data:

%     figure('Name','HRV vs Detrended HRV');
%     subplot(2,1,1);
%     plot(xx,hrv_equid);
%     hold on;
%     plot(xx,trend,'LineWidth',2);
%     title('HRV and Detrend Line');
%     
%     subplot(2,1,2);
%     plot(xx,hrv_stat);
%     title('(Almost) Stationary/Detrended HRV')
    
    %% Get hrv segments for stationary-equidistance hrv samples:
    % Segment hrv for spectral analysis:
    HRV_Frame_Index_stat = hrvSegmentation(xx,wind_size,overlay_size);
    % Remove segments where labels cannot be matched:
    HRV_Frame_Index_stat(removal_Index,:) = [];

    % Segment HRV:
    hrv_stat_Segment = cell(size(HRV_Frame_Index_stat,1),1);
    
    for i=1:size(HRV_Frame_Index_stat,1)
        hrv_stat_Segment{i} = hrv_stat(HRV_Frame_Index_stat(i,1):HRV_Frame_Index_stat(i,2));
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
    
    for i=1:size(hrv_stat_Segment)
        [pxx{i},f{i},VLF(i),LF(i),HF(i),TP(i),LFHF_ratio(i)] = spectralAnalysis(hrv_stat_Segment{i},m_order,re_sampleRate,range_VLF,range_LF,range_HF);
    end

    % % Plot Spectral Analysis:
    % figure('Name','Spectral Analysis Using AR Modelling');
    % plot(f,pxx);
    % title('PSD with AR Modelling');
    % ylabel('PSD [s^2/Hz]');
    % xlabel('Frequency [Hz]');
    % legend('pburg PSD estimate');

    %% Create and save feature table:
    feat_ECG_Stress = table(mean_RR,RMSSD,pNN50,skew_RR,LF,HF,LFHF_ratio,feature_Labels, ...
        'VariableNames',{'Mean RR','RMSSD','pNN50','Skewness RR','Abs Power LF', ...
        'Abs Power HF','LF HF Ratio','Label'});
    fileNameSave = strcat('/database/WESAD_ECG_Stress/','feat_ECG_S',string(str2double(regexp(fileName{index},'\d*','Match')))','.mat');
    save(fileNameSave,'feat_ECG_Stress');

    % Add subject information to database:
    sub_info = importfile_readme(fileName_readme{index});

    fileNameSave_info = strcat('/database/WESAD_ECG_Stress/','info_ECG_S',string(str2double(regexp(fileName_readme{index},'\d*','Match')))','_');
    temp = char(sub_info.Answ(4));
    temp = string(temp(1)) + '_A' + string(sub_info.Answ(1)) + '_H' + string(sub_info.Answ(2)) + '_W' + string(sub_info.Answ(3));
    fileNameSave_info = strcat(fileNameSave_info,temp);
    save(fileNameSave_info,'sub_info');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load all data and save as a single .csv file:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fileLoc = '/database/WESAD_ECG_Stress/';
cd(fileLoc)
fileList = dir('feat_ECG_S*.mat');
fileList = {fileList.name}';
fileName = strcat(fileLoc,fileList);
Feat_Matr = [];
for ii=1:size(fileName,1)
    temp = load(fileName{ii});
    tempName = fieldnames(temp);
    Feat_Matr = [Feat_Matr; temp.(tempName{:})];
end

Feat_Matr{Feat_Matr{:,end}~=2,end} = 0;
Feat_Matr{Feat_Matr{:,end}==2,end} = 1;

%% Testing Features: Prediction Models
% cats = categorical(Feat_Matr{:,end});
% nPlots = size(Feat_Matr,2)-1;
% figure('Name','ECG-Stress Variable Eval');
% tiledlayout(ceil(nPlots/2),2);
% 
% for ii=1:ceil(nPlots/2)
%     nexttile
%     gscatter(Feat_Matr{:,ii},Feat_Matr{:,ii},cats);
%     xlabel(Feat_Matr.Properties.VariableNames{ii});
%     ylabel(Feat_Matr.Properties.VariableNames{ii});
%     title('Model Predictions');
%     legend('NonStress','Stress');
%     nexttile
%     gscatter(Feat_Matr{:,ii},Feat_Matr{:,ii+1},cats);
%     xlabel(Feat_Matr.Properties.VariableNames{ii});
%     ylabel(Feat_Matr.Properties.VariableNames{ii+1});
%     title('Model Predictions');
%     legend('NonStress','Stress');
% end
% figure('Name','ECG-Stress Variable Eval');
% tiledlayout(floor(nPlots/2),2);
% 
% for ii=ceil(nPlots/2)+1:nPlots-1
%     nexttile
%     gscatter(Feat_Matr{:,ii},Feat_Matr{:,ii},cats);
%     xlabel(Feat_Matr.Properties.VariableNames{ii});
%     ylabel(Feat_Matr.Properties.VariableNames{ii});
%     title('Model Predictions');
%     legend('NonStress','Stress');
%     nexttile
%     gscatter(Feat_Matr{:,ii},Feat_Matr{:,ii+1},cats);
%     xlabel(Feat_Matr.Properties.VariableNames{ii});
%     ylabel(Feat_Matr.Properties.VariableNames{ii+1});
%     title('Model Predictions');
%     legend('NonStress','Stress');
% end
% 
% nexttile([1 2]);
% gscatter(Feat_Matr{:,end-1},Feat_Matr{:,end-1},cats);
% title('Model Predictions');
% legend('NonStress','Stress');

%% Create Classification Model:
Feat_Matr = table2array(Feat_Matr);
ECG_Stress_MLModel = SVMCubicTrainer(Feat_Matr);

%% Save Classification Model:
% Move to current running script:
filePath = matlab.desktop.editor.getActiveFilename;
cd(fileparts(filePath));
cd('..');
save("ECG_Stress_MLModel.mat","ECG_Stress_MLModel");