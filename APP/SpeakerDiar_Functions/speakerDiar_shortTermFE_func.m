function [shortTermFeat] = speakerDiar_shortTermFE_func(audioIn,Fs,wndPar)
% This function creates short-term acoustic features (i.e., 19-MFCCs) and
% indicates the regions it has been obtained from within the audioIn data.
% INPUT: audioIn=audio track
%        Fs=sampling rate
%        VADidx_Speech=speech regions within the audio track
%        wndPar=window parameters for feature extraction
%        minDur=minimum enforced duration.
% OUTPUT: shortTermFeat=19-MFCCs feature matrix

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

% if isempty(minDur)
% % Minimum speech duration:
%     error('Error: Minimum duration unspecified');
% end

% 19-MFCCs feature extraction setup:
aFE = audioFeatureExtractor(SampleRate = Fs,...
    Window=hann(wndPar.windowSamples,'periodic'),...
    OverlapLength = wndPar.overlapSamples,...
    mfcc = true);
setExtractorParams(aFE,"mfcc",NumCoeffs=19);

shortTermFeat = extract(aFE,audioIn);

%EOF
end