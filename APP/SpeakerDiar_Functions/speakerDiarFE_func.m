function features = speakerDiarFE_func(Data,Fs,wndPar)
% Feature Creation for newly trained GMM models for Speaker Diarization
% This script creates the training data for the GMM models that form part
% of the VAD HMM.

    % Parameters:
    if isempty(wndPar)
        windowDuration = 0.03; % 30 ms window
        hopDuration = 0.01; % 10 ms shift (hop)
        wndPar.windowSamples = round(windowDuration*Fs);
        hopSamples = round(hopDuration*Fs);
        wndPar.overlapSamples = wndPar.windowSamples - hopSamples;
    end

    aFE = audioFeatureExtractor(SampleRate = Fs,...
        Window=hann(wndPar.windowSamples,'periodic'),...
        OverlapLength = wndPar.overlapSamples,...
        mfcc = true,...
        zerocrossrate=true);
    setExtractorParams(aFE,"mfcc",NumCoeffs=12);

    % Extract features:
    Data = cat(2,Data(:));
    features = extract(aFE,Data);
    
%     % Remove MFCC2 from feature set:
%     features(:,2) = [];
%     % Seperate SpectralEntropy from set:
%     specEntro = features(:,12);
%     features(:,12) = [];

    % Get delta and delta-delta of all features:
    tempDelta = audioDelta(features);
    tempDelta_Delta = audioDelta(tempDelta);
    
    features = [features,tempDelta,tempDelta_Delta];
    
%EOF
end