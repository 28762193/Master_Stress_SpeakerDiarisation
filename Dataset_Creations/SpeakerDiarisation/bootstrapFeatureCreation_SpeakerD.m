% Bootstrap Feature Creation:
% This script creates the training data for the GMM models that form part
% of the Bootstrap HMM. Three models will have to be created so three sets
% of data will be used namely; speech, silence, and noise.
clear all;
loadOrCreate = 0; % 1=Load, 0=Create Features
modelSSN = 0; %0/Otherwise=Speech, 1=Silence, 2=Noise
normFeatures = 0; % 1=Normalize features (only after all seperate features are obtained).
if loadOrCreate == 0
    normFeatures = 0;
end
if loadOrCreate
    % Move to current running script:
    filePath = matlab.desktop.editor.getActiveFilename;
    cd(fileparts(filePath));
    load('allFeatures_Speech.mat');
    %load('normFactors_Speech.mat');
    load('allFeatures_Silence.mat');
    %load('normFactors_Silence.mat');
    load('allFeatures_Noise.mat');
    %load('normFactors_Noise.mat');
else
    % Creating Datastores for each Model/String:
    fileSavePath = '/Speaker_Diarization/';
    switch modelSSN
        case 1
            filepath = '/Datasets/Commonvoice/train/Silence/';
            newNameAF = 'allFeatures_Silence';
            newNameNF = 'normFactors_Silence';
        case 2
            filepath = '/Datasets/Commonvoice/train/Noise/';
            newNameAF = 'allFeatures_Noise';
            newNameNF = 'normFactors_Noise';
        otherwise
            filepath = '/Datasets/Commonvoice/train/Speech/';
            newNameAF = 'allFeatures_Speech';
            newNameNF = 'normFactors_Speech';
    end
    fileSaveNameAF = strcat(fileSavePath,newNameAF);
    %fileSaveNameNF = strcat(fileSavePath,newNameNF);

    adsTrain = audioDatastore(filepath);


    % Feature Extraction from entire dataset:
    Fs = 16e3;
    windowDuration = 0.03; % 30 ms window
    hopDuration = 0.01; % 10 ms shift (hop)
    wndPar.windowSamples = round(windowDuration*Fs);
    hopSamples = round(hopDuration*Fs);
    wndPar.overlapSamples = wndPar.windowSamples - hopSamples;
    % Features consist of 12-MFCCs + ZCRate, and their 
    % first and second derivative(39-Features):

%     aFE = audioFeatureExtractor(SampleRate = Fs,...
%         Window=hann(windowSamples,'periodic'),...
%         OverlapLength = overlapSamples,...
%         mfcc = true);
%     setExtractorParams(aFE,"mfcc",NumCoeffs=12); % SpectralEntropy causes a high correlation which hinders the covariance matrices of the GMMs.

    featuresAll = [];

    if modelSSN == 0
        % Divide audio (Speech) files between 4 workers:
        numPar = 4;
        p = gcp('nocreate'); % Initialise parpool if one does not exist
        if isempty(p)
            poolsize = 0;
        else
            parpool(numPar);
        end

        parfor i = 1:numPar
            adsPart = partition(adsTrain,numPar,i);
            featuresPart = cell(0,numel(adsPart.Files));
            for ii = 1:numel(adsPart.Files)
                audioData = read(adsPart);
                featuresPart{ii} = speakerDiarFE_func(audioData,Fs,wndPar);
                %featuresPart{ii} = helperFeatureExtraction(audioData,aFE);
            end
            featuresAll = [featuresAll,featuresPart];
        end

    else
        featuresPart = cell(0,numel(adsTrain.Files));
        for i = 1:numel(adsTrain.Files)
            audioData = read(adsTrain);
            featuresPart{i} = speakerDiarFE_func(audioData,Fs,wndPar);
            %featuresPart{i} = helperFeatureExtraction(audioData,aFE);
        end
        featuresAll = [featuresAll,featuresPart];
    end
    allFeatures = cat(1,featuresAll{:});
    %     % Calculate mean and std for all the features: (Normalisation Step)
    %     normFactors.Mean = mean(allFeatures,2,"omitnan");
    %     normFactors.STD = std(allFeatures,[],2,"omitnan");
    %
    %     % Normalise features:
    %     allFeatures = (allFeatures-normFactors.Mean)./normFactors.STD;
    %     % Cepstral mean subtraction (for channel noise):
    %     allFeatures = allFeatures - mean(allFeatures,"all");

    % Save Variables:
    S1.(newNameAF) = allFeatures'; %S2.(newNameNF) = normFactors;
    save(fileSaveNameAF,'-struct','S1'); %save(fileSaveNameNF,'-struct','S2');
end

if normFeatures % Normalize features:
    % Obtain smallest array:
    norm_featLen = min([size(allFeatures_Speech,2), size(allFeatures_Silence,2),...
        size(allFeatures_Noise,2)]);
    norm_feats = [allFeatures_Speech(:,1:norm_featLen),...
        allFeatures_Silence(:,1:norm_featLen),...
        allFeatures_Noise(:,1:norm_featLen)];

    % Calculate mean and std for all the features: (Normalisation Step)
    normFactors.Mean = mean(norm_feats,2,"omitnan");
    normFactors.STD = std(norm_feats,[],2,"omitnan");

    % Normalise features:
    allFeatures_Speech = (allFeatures_Speech-normFactors.Mean)./normFactors.STD;
    allFeatures_Silence = (allFeatures_Silence-normFactors.Mean)./normFactors.STD;
    allFeatures_Noise = (allFeatures_Noise-normFactors.Mean)./normFactors.STD;
    % Cepstral mean subtraction (for channel noise):
    norm_feats = [allFeatures_Speech(:,1:norm_featLen),...
        allFeatures_Silence(:,1:norm_featLen),...
        allFeatures_Noise(:,1:norm_featLen)];
    allFeatures_Speech = allFeatures_Speech - mean(norm_feats,"all");
    allFeatures_Silence = allFeatures_Silence - mean(norm_feats,"all");
    allFeatures_Noise = allFeatures_Noise - mean(norm_feats,"all");

    save("normFactors.mat",'normFactors');
    save("allFeatures_Speech.mat","allFeatures_Speech");
    save("allFeatures_Silence.mat","allFeatures_Silence");
    save("allFeatures_Noise.mat","allFeatures_Noise");
%EOC
end

%% Helper Functions:
function [features,numFrames] = helperFeatureExtraction(audioIn,aFE)
    % Feature extraction
    features = extract(aFE,audioIn); % Extract 12 MFFCs and spectralEntropy.
    tempZCR = zerocrossrate(audioIn,'WindowLength',size(aFE.Window,1),...
        'OverlapLength',aFE.OverlapLength); % Extract zerocrossrate
    features = [features,tempZCR];
    
    % Get delta and delta-delta of all features:
    tempDelta = audioDelta(features);
    tempDelta_Delta = audioDelta(tempDelta);
    
    features = [features,tempDelta,tempDelta_Delta];
    
    features = features';
    
    numFrames = size(features,2);
end