function [featTable,VADidx_Speech] = speakerDiar_longTermFE_func(audioIn,Fs,VADidx_Speech,wndPar,SpeechWndDur)
% Long-term Feature extraction:
% This function extracts the prosodic features from the obtained speech
% regions of the audiotrack. These features are to be used in the initial
% pre-clustering phase of the AHC speaker-clustering.

% INPUT: audioIn=current audiotrack
%        FS=sampling rate of the audio
%        VADidx_Speech=The speech regions within audioIn
%        wndPar=Window parameters for features (Struct)
%        SpeechWndLen=Min speech window length

% OUTPUT: featTable=Table containing the prosodic features (Pitch[median,min,mean];
% Harmonic-to-noise ratio[mean]; 4thFormant[std,min,mean];
% 5thFormant[std,min,mean]; FormantDispersion[mean])

% Parameters:|
%-------------
if isempty(wndPar)
% Pitch feature window: Default Values - corresponds to 98 pitch and harmonic values
    windowDuration = 0.03; % 30 ms window
    hopDuration = 0.01; % 10 ms shift (hop)
    wndPar.windowSamples = round(windowDuration*Fs);
    hopSamples = round(hopDuration*Fs);
    wndPar.overlapSamples = wndPar.windowSamples - hopSamples;
end
if ~(fieldnames(wndPar)=="windowSamplesForm")
    % Formant feature window: Default values - corresponds to 78 formant values
    hopDur = 0.0125; % 12.5ms hop
    hopSamplesForm = hopDur*Fs;
    windowSamplesForm = 3*hopSamplesForm; % 37.5ms analysis window
    overlapSamplesForm = windowSamplesForm-hopSamplesForm;
end
if isempty(SpeechWndDur)
% Speech segment duration:
    SpeechWndDur = 1; % 1sec segments (rectangular-window)
end
% Segment parameters: with the addition of a Hamming window
segPar.windowSamples = round(SpeechWndDur*Fs);
segPar.overlapSamples = 0.5*segPar.windowSamples; % 50% overlap

%% Create speech analysis segments:
% Filter speech segments - ignore segments that are less than
% SpeechWndLen
% The segments that are included are Seg ~[1000ms; 1500ms) (due to 50%
% overlap)
SpeechWndLen = SpeechWndDur*Fs;
if isempty(VADidx_Speech)
    % Perform VAD:
    VADidx_Speech = speakerDiar_VAD_func(audioIn,Fs,wndPar);
end
if size(VADidx_Speech,2) > 2
    % Select merely the speech regions:
    VADidx_Speech = VADidx_Speech(VADidx_Speech(:,3)==1,1:2);
end
% Remove speech regions that are < SpeechWndDur*Fs
VADidx_Speech(VADidx_Speech(:,2)-VADidx_Speech(:,1)+1 < SpeechWndLen,:) = [];

%% Adding Overlap:
nSegs = zeros(size(VADidx_Speech,1),1);
for ii=1:size(VADidx_Speech,1)
    nSegs(ii) = floor((VADidx_Speech(ii,2)-VADidx_Speech(ii,1)+1)/segPar.overlapSamples)-1;
end

tempSpeechVAD = zeros(sum(nSegs),2);
% Segment Speech VADidx into overlapping segments with a minimum duration
% of SpeechWndDur:
kk = 1;
for ii=1:size(VADidx_Speech,1)
    if nSegs(ii)>1
    tempSpeechVAD(kk,1) = VADidx_Speech(ii,1);
    tempSpeechVAD(kk,2) = VADidx_Speech(ii,1)+SpeechWndLen-1;
    kk = kk+1;
    for jj=2:nSegs(ii)-1
        tempSpeechVAD(kk,1) = tempSpeechVAD(kk-1,2)-segPar.overlapSamples+1;
        tempSpeechVAD(kk,2) = tempSpeechVAD(kk,1)+SpeechWndLen-1;
        kk = kk+1;
    end
    tempSpeechVAD(kk,1) = tempSpeechVAD(kk-1,2)-segPar.overlapSamples+1;
    tempSpeechVAD(kk,2) = VADidx_Speech(ii,2);
    kk = kk+1;
    else
        tempSpeechVAD(kk,1:2) = VADidx_Speech(ii,1:2);
        kk = kk+1;
    end
end

VADidx_Speech = tempSpeechVAD;

audio_Speech = cell(1,size(VADidx_Speech,1));

for ii=1:size(VADidx_Speech,1)
    audio_Speech{ii} = audioIn(VADidx_Speech(ii,1):VADidx_Speech(ii,2));
    audio_Speech{ii} = hamming(length(audio_Speech{ii}),"periodic").*audio_Speech{ii};
end
%% Extract Features: Pitch and Harmonic-to-noise ratio

aFE = audioFeatureExtractor(SampleRate = Fs,...
    Window=hann(wndPar.windowSamples,'periodic'),...
    OverlapLength = wndPar.overlapSamples,...
    pitch=true,...
    harmonicRatio=true);
setExtractorParameters(aFE,'pitch','MedianFilterLength',10); % Add movmedian filter to smooth pitch values
% Default pitch range will suffice (50-400 Hz) for adult male and female,
% as well as for children. A 10ms hopDuration will provide 98 pitch values
% per second - which will be the length of each analysed segment (1-second).
longTermFeatures = cell(1,size(audio_Speech,2));
for ii=1:size(audio_Speech,2)
    longTermFeatures{ii} = extract(aFE,audio_Speech{ii});
end

% plotFeatures(aFE,audio_Speech{1});

%% Formant Detection:

% Apply pre-emphasis filter:
formFreq = cell(1,size(audio_Speech,2));
for ii=1:size(audio_Speech,2)
    preemph = [1 0.63]; % Pre-emphasis filter - Emphasises high freq and attenuates low frequencies
    x1 = filter(1,preemph,audio_Speech{ii});
    
    % Analysis windows:
    xBuf = buffer(x1,windowSamplesForm,overlapSamplesForm,'nodelay');
    nFrames = floor(size(x1,1)/hopSamplesForm) - 2;
    xBuf = xBuf(:,1:nFrames);
    
    % Get Linear prediction filter
    nCoeff=2+Fs/1000;           % Rule of thumb for formant estimation
    a=lpc(xBuf,nCoeff);
    
    formFreq{ii} = cell(1,size(a,1));
    % Find frequencies by root-solving
    for jj=1:size(a,1)
        try
        r=roots(a(jj,:));                  % find roots of polynomial a
        r=r(imag(r)>0.01);           % only look for roots >0Hz up to Fs/2
        % Get formant frequencies:
        formFreq{ii}{jj}=sort(atan2(imag(r),real(r))*Fs/(2*pi)); % convert to Hz and sort
        catch
            disp(ii);
            disp(jj);
        end
    end
end
% Get formant dispersion:
% Formant dispersion is the average differnce between succesive formant
% frequencies.

formDispersion = cell(size(formFreq));
for ii=1:size(formFreq,2)
    tempfFreq = formFreq{ii};
    % Get number of min formFreqs:
    tempLen = cell2mat(cellfun(@length,tempfFreq,'UniformOutput',false));
    minLen = min(tempLen);
    % Limit formant freqs to minLen:
    formDispersion{ii} = cell(size(tempfFreq)); % Formant Dispersion
    for jj=1:size(tempfFreq,2)
        tempfFreq{jj} = tempfFreq{jj}(1:minLen,1);
        formDispersion{ii}{jj} = mean(diff(tempfFreq{jj}));
        % Formant Frequencies: Limit to 4th and 5th
        formFreq{ii}{jj} = formFreq{ii}{jj}(4:5,1);
    end
end

%% Feature Extraction:
%--------------------------------------------------------------------------
% Mean - The average value.
% Median - The value of the 50th percentile.
% Min, max - Instead of the actual min and max values, the 5th and 95th
% percentiles, respectively, are taken to avoid an otherwise large impact
% of outliers caused by artifacts.
% Diff - The difference between the max and min value.
% Stdev - The standard deviation as a measure of the variance.
% Swoj - The slope of the pitch curve ignoring octave jumps.
%--------------------------------------------------------------------------
% Pitch Features: Median, min, and mean.
feat_Pitch = zeros(size(longTermFeatures,2),3);
% Harmonic-to-noise ratio Feature: Mean
feat_Harm  = zeros(size(longTermFeatures,2),1);
for ii=1:size(longTermFeatures,2)
    feat_Pitch(ii,1) = quartile_func(longTermFeatures{ii}(:,1),0.5);
    feat_Pitch(ii,2) = quartile_func(longTermFeatures{ii}(:,1),0.05);
    feat_Pitch(ii,3) = mean(longTermFeatures{ii}(:,1));

    feat_Harm(ii,1) = mean(longTermFeatures{ii}(:,2));
end

% Formant Features: Std, min, mean.
feat_4Formant = zeros(size(formFreq,2),3);
feat_5Formant = zeros(size(formFreq,2),3);
feat_FDispersion = zeros(size(formFreq,2),1);
for ii=1:size(formFreq,2)
    temp = cell2mat(formFreq{ii})';
    feat_4Formant(ii,1) = std(temp(:,1));
    feat_4Formant(ii,2) = quartile_func(temp(:,1),0.05);
    feat_4Formant(ii,3) = mean(temp(:,1));
    feat_5Formant(ii,1) = std(temp(:,2));
    feat_5Formant(ii,2) = quartile_func(temp(:,2),0.05);
    feat_5Formant(ii,3) = mean(temp(:,2));
    feat_FDispersion(ii,1) = mean(cell2mat(formDispersion{ii}));
end

featTable = table(feat_Pitch,feat_Harm,feat_4Formant,feat_5Formant,feat_FDispersion);
%EOF
end
