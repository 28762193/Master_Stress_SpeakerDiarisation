function [audioData,Segments] = speakerDiar_func(app,audioDS,Fs)


% Testing Speaker Diarization:
% tic
%% Set Parameters:
% Feature Window:
windowDuration = 0.03; % 30 ms window
hopDuration = 0.01; % 10 ms shift (hop)
wndPar.windowSamples = round(windowDuration*Fs);
hopSamples = round(hopDuration*Fs);
wndPar.overlapSamples = wndPar.windowSamples - hopSamples;
% Audio Window:
chunkWndLen = 5*60*Fs;   % 5-min window


%% Load audioData into Tall Array:
audioIn = tall(read(audioDS));
reset(audioDS);

% chunkWndLen = gather(length(audioIn));

%% Create 5-min chunks:
audioChunk = cell(1,ceil(gather(length(audioIn))/chunkWndLen));
chunk_K = 1;
for ii=1:size(audioChunk,2)-1
    chunk_K = ii*chunkWndLen+1;
    audioChunk{ii} = audioIn((ii-1)*chunkWndLen+1:chunk_K-1);
end
audioChunk{end} = audioIn(chunk_K:end);
nChunks = gather(size(audioChunk,2));
% Speaker Diarization Container:
Segments = cell(nChunks,1);
HMM_speakerSeg = cell(nChunks,1);
shortTermFeat = cell(nChunks,1);
SpeechIdx = cell(nChunks,1);
audioData = cell(nChunks,1);

maxMergeIters = 25;

% Check for closeReq before executing speaker diarisation:
pause(2);
if app.closeReq; return; end
p = gcp('nocreate');
if isempty(p); parpool(4); end
%% Run Speaker Diarization:
parfor chunkIter=1:nChunks
%     VADidx = [];
    SpeechIdx{chunkIter} = [];
    audioData{chunkIter} = [];
    audioPre = gather(audioChunk{chunkIter});
    
    %% Perform pre-processing:
    audioPre = audioPre./max(abs(audioPre));
    % tic
    audioPre = v_ssubmmse(audioPre,Fs); % Obtained from voice-box
    % toc
    audioPre = sign(audioPre).*(abs(audioPre).^0.75); % Mitigate distant speakers.
    % Add white Gaussian noise with a high SNR to the audio data as to ensure
    % proper zerocrossingrates and other features:
    audioPre = awgn(audioPre,120); % Add white Gaussian noise.
    % highFreqComp = 0.0001*(2*rem(1:numel(audioPre),2) - 1)';
    % audioData = audioPre+highFreqComp;

    %% Perform VAD:
    for ii=1:1
%         tic
        VADidx = speakerDiar_VAD_func(audioPre,Fs,wndPar,1); % If ignoreNoiseFlag then the VAD algorithm is based on detectSpeech rather than Viterbi.
%         toc
        SpeechIdx{chunkIter} = VADidx(VADidx(:,3)==1,:);
%         SilIdx = VADidx(VADidx(:,3)==2,:);
%         NoiseIdx = VADidx(VADidx(:,3)==3,:);
        % %% Plot segments from VAD:
        % plotRegions_func(VADidx,audioData);
    end
    audioData{chunkIter} = audioPre;
    disp('VAD Done')
    disp(chunkIter);
end
    % %% Extract and plot specific labels:
    % labelToInspect = 1;
    % cutOutSilenceFromAudio = 1;
    %
    % bmsk = binmask(msk,numel(audioData));
    % t = (0:size(audioData,1)-1)/Fs;
    %
    % audioToPlay = audioData;
    % if cutOutSilenceFromAudio
    %     audioToPlay(~bmsk(:,labelToInspect)) = [];
    % end
    % audioToPlay = gather(audioToPlay);
    %
    % figure;
    % tiledlayout(2,1)
    %
    % nexttile
    % plot(t,gather(audioData))
    % axis tight
    % ylabel('Amplitude')
    %
    % nexttile
    % plot(t,gather(audioData).*bmsk(:,labelToInspect))
    % axis tight
    % xlabel('Time (s)')
    % ylabel('Amplitude')
    % title("Speaker Group "+labelToInspect)
%% Execute Speaker Diarisation
% Check for closeReq before executing speaker diarisation:
pause(3);
if app.closeReq; return; end
parfor chunkIter=1:nChunks
    [HMM_speakerSeg{chunkIter},shortTermFeat{chunkIter},Segments{chunkIter}] = speakerDiar_SegClustCore_func(gather(audioData{chunkIter}),Fs,SpeechIdx{chunkIter},wndPar,maxMergeIters);
end
% Check for closeReq before executing Speaker Merging and Final Segmentation:
pause(3);
if app.closeReq; return; end
%% Merge Speaker Diarization Audio Segments (and resegment and cluster)
% Merge Segments and AudioChunks:
nClusts = 0;
for ii=2:size(Segments,1)
    Segments{ii}(:,1:2) = Segments{ii}(:,1:2)+(ii-1)*chunkWndLen;
    SpeechIdx{ii}(:,1:2) = SpeechIdx{ii}(:,1:2)+(ii-1)*chunkWndLen;
    nClusts = nClusts+length(unique(Segments{ii-1}(:,3)));
    Segments{ii}(:,3) = Segments{ii}(:,3)+nClusts;
end
Segments = cat(1,Segments{:});
SpeechIdx = cat(1,SpeechIdx{:});
temp_shortTermFeat = cat(2,shortTermFeat{:});
audioData = gather(cat(1,audioData{:}));
% audioData = gather(cat(1,audioChunk{:}));
[HMM_speakerSeg,shortTermFeat,Segments] = speakerDiar_SegClustCore_func(audioData,Fs,SpeechIdx,wndPar,maxMergeIters,HMM_speakerSeg,temp_shortTermFeat,Segments);

%% Final Segmentation and Output:
% Last Iteration using a smaller min duration:
% Parameters:
minDur_speakerSeg = 1.5; % minimum segment duration
minSize_speakerSeg = minDur_speakerSeg/hopDuration; % minimum number off feature samples
%--------------------------------------------------------------------------
% Obtain short-term feature set on speech only data:|
%----------------------------------------------------
tempIdx = SpeechIdx;
tempIdx(tempIdx(:,2)-tempIdx(:,1)<minDur_speakerSeg*Fs,:) = [];
shortTermFeat = cell(1,size(tempIdx,1)); % Will be used throughout the segmentation process
for ii=1:size(tempIdx,1)
    shortTermFeat{ii} = speakerDiar_shortTermFE_func(audioData(tempIdx(ii,1):tempIdx(ii,2)),Fs,wndPar);
end

% Train Model:
% Create HMM for speaker segmentation:|
%--------------------------------------
% ASPG:
speechSeconds = diff(SpeechIdx(:,1:2),[],2)+1;
speechSeconds = (sum(speechSeconds)-1)/Fs;
secpergauss = 0.01*speechSeconds+2.6;
g = floor(speechSeconds/(secpergauss*nClusts)); % Number of Gaussians per GM

prevClusts = nClusts;
nClusts = length(unique(Segments(:,3)));

HMM_speakerSeg = speakerDiar_SegClustTrain_func(audioData,Fs,nClusts,minSize_speakerSeg,Segments,wndPar,g,1);

% Segmentation:
[Segments,~] = speakerDiar_SegClustSeg_func(HMM_speakerSeg,shortTermFeat,hopSamples,tempIdx);
nClusts = length(unique(Segments(:,3)));
if nClusts~=prevClusts
%         reCreateFlag=1; prevClusts=nClusts;
    % Change cluster 'names':
    [~,~,tempIC] = unique(Segments(:,3));
    tempC = 1:nClusts; Segments(:,3) = tempC(tempIC);
end
% toc

%% Plot segments from Final Segmentation:
% plotRegions_func(Segments,audioData);
% toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%EOS
end