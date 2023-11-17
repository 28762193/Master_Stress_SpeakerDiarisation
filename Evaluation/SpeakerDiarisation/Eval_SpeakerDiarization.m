% Evaluate Speaker Diarization:

%% Load Audio:
[audioData,Fs] = audioread("tempAudioMix_SuperLong_Processed.wav");
%% Parameters:
tol_wnd = ceil(0.250*Fs); % Tolerance window of ±250ms (on a side), as given by NIST
minDur_speakerSeg = 1.5;

%% Test Accuracy: Get trueLabels:
load("speech_SuperLongSig_Rev5_Processed.mat");
% load("tempAudioMixed_Libri_LongExt_Labeled.mat");

%% Get Speech Regions:
speechRegions = floor(ls.Labels.Speech{:,:}.ROILimits*Fs);

%% Get Track Regions:
load("tempAudioMixed_LibriSpeech_SuperLong_Info.mat")
load("tempAudioMixed_LibriSpeech_SuperLong_Spk.mat")

trackRegions = zeros(size(trackInformation,2),3);
trackRegions(1,1:2) = [1 trackInformation{1,1}.TotalSamples];
for ii=2:size(trackRegions,1)
    trackRegions(ii,1:2) = [trackRegions(ii-1,2)+1 trackInformation{1,ii}.TotalSamples+trackRegions(ii-1,2)];
end
trackRegions(:,3) = spkIdx;
%% Get Ground Truth:
speakerRegions = speechRegions;
speakerRegions(:,3) = zeros(size(speakerRegions,1),1);
for ii=1:size(trackRegions,1)
    speakerRegions(speakerRegions(:,1)>=trackRegions(ii,1)&speechRegions(:,2)<=trackRegions(ii,2),3) = trackRegions(ii,3);
end
%% Create LabelDefitions:
speakerLbDef = signalLabelDefinition('Speakers','LabelType','roi',...
    'LabelDataType','categorical','Categories',{'1','2','3','4','5'});
% Add Speaker Labels:
ls.addLabelDefinitions(speakerLbDef);
setLabelValue(ls,1,'Speakers',speakerRegions(:,1:2),string(speakerRegions(:,3)));

%% Create TruthRegions:
t = getLabeledSignal(ls);
data = t.Signal{:};
tempLabels = t.Speakers{:};
trueLabels = floor(tempLabels.ROILimits);
trueLabels(:,end+1) = tempLabels.Value;  %cell2mat(tempLabels.Value);
[~,sortIdx] = sort(trueLabels(:,1));
trueLabels = trueLabels(sortIdx,:);

%% Plot GroundTruth Speakers:
maskLabels = trueLabels(:,3);
regionsVAD = trueLabels(:,1:2);
msk = signalMask(table(regionsVAD,categorical(maskLabels)));

figure('Name','AudioMixed_LibriSpeech_LongExt_Truth');
plotsigroi(msk,data,true)
axis([0 numel(data) -1 1])
title('Ground Truth')



%% Test Accuracy: Hypothesized Labels
% load('SegmentsEval.mat');

% Change cluster 'names':
[~,~,tempIC] = unique(Segments(:,3)); % Match guess labels (from random GMM) to the true labels.
tempNewC = [5 2 1 4 3]; % MANUALY MATCH CLSUTERS FROM GUESS TO TRUTH. (e.g., guess 1 to truth 5 is [5 0 0 0 0])
Segments(:,3) = tempNewC(tempIC);
guessSegments = Segments;

%% Plot Hypothesized Speakers:
maskLabels = guessSegments(:,3);
regionsVAD = guessSegments(:,1:2);
msk = signalMask(table(regionsVAD,categorical(maskLabels)));
%% Test:
figure('Name','AudioMixed_LibriSpeech_LongExt_Hypothesized');
plotsigroi(msk,data,true)
axis([0 numel(data) -1 1])
title('Hypothesized')

%% Calculate Accuracy:

MISS = 0;   % Missed speaker segment - Speaker in Ref, but not in Hypo; False Negative
FA = 0;     % False Alarm - Speaker in Hypo, but not in Ref - False Positive
SE = 0;     % Speaker Error - Mapped Ref/true is not the same as hypo spreaker.
SE_IncClust = 0; % Speaker Error (Error caused by incorrect clustering)
% The above measurements are given in [s] - time variables
% DER = (MISS+FA+SE+OVL)/length of the Ref - given as a percentage of time
% OVL - overlap speakers - is not measured here.

% Create fixed.Interval variables:
trueInterval = fixed.Interval(trueLabels(:,1),trueLabels(:,2));
guessInterval = fixed.Interval(guessSegments(:,1),guessSegments(:,2));

trueLen = diff(trueLabels(:,1:2),[],2);
trueLen = sum(trueLen);

% VAD Eval:
%MISS Calc:
for ii=1:size(trueLabels,1)
    intersection = intersect(trueInterval(ii),guessInterval(:));
    if isempty(intersection)
        tempMISS = (trueLabels(ii,2)-trueLabels(ii,1)) - tol_wnd;
        if tempMISS >= minDur_speakerSeg*Fs
            MISS = MISS + tempMISS;
        else
            % Obtain Completely missed segments - Usually caused by
            % quantization error from Segmentation and Clustering, thus not
            % a VAD error but a Speaker Error: Indicates that the
            % limitation lies with S+C and not the VAD algorithm
            SE = SE + tempMISS;
        end
    else
        tempMISS= double(intersection(1).LeftEnd - trueInterval(ii).LeftEnd - tol_wnd);
        if tempMISS>0;MISS = MISS+tempMISS;end

        tempMISS = double(trueInterval(ii).RightEnd - intersection(end).RightEnd - tol_wnd);
        if tempMISS>0;MISS = MISS+tempMISS;end

        for jj=2:size(intersection,1)
            tempMISS = double(intersection(jj).LeftEnd - intersection(jj-1).RightEnd - tol_wnd);
            if tempMISS>0;MISS = MISS+tempMISS;end
        end
    end
end
%FA Calc:
for ii=13:size(guessSegments)
    intersection = intersect(guessInterval(ii),trueInterval(:));
    if isempty(intersection)
        tempFA = (guessSegments(ii,2)-guessSegments(ii,1)) - tol_wnd;
        if tempFA >= minDur_speakerSeg*Fs
            FA = FA + tempFA;
        else
            SE = SE + tempFA;
        end
    else
        tempFA = double(intersection(1).LeftEnd - guessInterval(ii).LeftEnd - tol_wnd);
        if tempFA>0;FA = FA+tempFA;end

        tempFA = double(guessInterval(ii).RightEnd - intersection(end).RightEnd - tol_wnd);
        if tempFA>0;FA = FA+tempFA;end

        for jj=2:size(intersection,1)
            tempFA = double(intersection(jj).LeftEnd - intersection(jj-1).RightEnd - tol_wnd);
            if tempFA>0;FA = FA+tempFA;end
        end
    end
end
%SE Calc:
trueInterLabel = 0;
guessInterLabel = 0;
for ii=1:size(trueInterval,1)
    guessSECount = 1;
    while guessSECount <= size(guessInterval,1)
        intersection = intersect(trueInterval(ii),guessInterval(guessSECount));
        if ~isempty(intersection)
            % Compare:
            if (trueLabels(ii,3) ~= guessSegments(guessSECount,3))
                SE_IncClust = SE_IncClust + trueInterval(ii).RightEnd - trueInterval(ii).LeftEnd;
            end
        end
        guessSECount = guessSECount + 1;
    end
end
SE = SE + SE_IncClust; % SE - The error caused by the Segmentation+Clustering
% portion of the Speaker Diarization algorithm. It includes the
% quantization error (limitation from large window sizes) and the error
% associated with incorrect clustering.
MISS = (MISS/trueLen) * 100;
FA = (FA/trueLen) * 100;
SE = (SE/trueLen) * 100;
SE_IncClust = (SE_IncClust/trueLen) * 100;
DER = MISS+FA+SE; % Speaker Diarization Error
% fprintf('MISS Error: %d%% \n',MISS);
% fprintf('FA Error: %d%% \n',FA);
% fprintf('SE Error: %d%% \n',SE);
% fprintf('SE by incorrect Clustering: %d%% \n',SE_IncClust);
% fprintf('DER Error: %d%% \n',DER);

EvalTab = table(MISS,FA,SE,SE_IncClust,DER); % Measured in percentage.
disp(EvalTab);

%% Evaluate Measurements:
% Turn-taking
% Total Spoken Duration
% Longest Phrase
% Longest Spoken Turn
[speakTurns_Ground,speakDur_Ground,longPhrase_Ground,longTurn_Ground] = getSpeakStats_func(trueLabels,Fs);
speakStats_Ground = table(speakTurns_Ground,speakDur_Ground,longPhrase_Ground,longTurn_Ground);
disp(speakStats_Ground)

[speakTurns_Hypo,speakDur_Hypo,longPhrase_Hypo,longTurn_Hypo] = getSpeakStats_func(guessSegments,Fs);
speakStats_Hypo = table(speakTurns_Hypo,speakDur_Hypo,longPhrase_Hypo,longTurn_Hypo);
disp(speakStats_Hypo)




%EOS