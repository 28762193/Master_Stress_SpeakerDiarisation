function [HMM_speakerSeg,shortTermFeat,Segments] = speakerDiar_SegClustCore_func(audioData,Fs,SpeechIdx,wndPar,maxMergeIters,HMM_speakerSeg,shortTermFeat,Segments)
%% Parameters:
hopSamples = (wndPar.windowSamples - wndPar.overlapSamples);
hopDuration = hopSamples/Fs;
minDur_speakerSeg = 2.5; % minimum segment duration
minSize_speakerSeg = minDur_speakerSeg/hopDuration; % minimum number off feature samples
filtSpeechIdx = SpeechIdx;  % temporary speech vad-index which is < minDur_speakerSeg
filtSpeechIdx(filtSpeechIdx(:,2)-filtSpeechIdx(:,1)<minDur_speakerSeg*Fs,:) = [];
if ~exist("shortTermFeat","var") || ~exist("Segments","var") || ~exist("HMM_speakerSeg","var")
%% Initialization: Segmentation and Clustering
%     tic
    [speakerIdx,Comps,g] = speakerDiar_SegClustIni_func(audioData,Fs,SpeechIdx,wndPar,1);
%     toc
    % %% Plot segments from initial long-term overlapped seg+clustering:
    % plotRegions_func(speakerIdx,gather(audioData));

    %% Segmentation and Clustering: Core Algorithm
    %----------------------------------------
    % 1.Resegmentation and Retraining:      |
    % 2.Model Merging                       |
    % 3.Stopping Condition                  |
    %----------------------------------------
%     tic
    % Parameters:
    % Comps = number of states (modelled via GMM)
    % g = number of gaussians per GMM
    prevClusts = Comps;
    reCreateFlag = 0;

    %--------------------------------------------------------------------------
    % Obtain short-term feature set on speech only data:|
    %----------------------------------------------------

    shortTermFeat = cell(1,size(filtSpeechIdx,1)); % Will be used throughout the segmentation process

    for ii=1:size(filtSpeechIdx,1)
        shortTermFeat{ii} = speakerDiar_shortTermFE_func(audioData(filtSpeechIdx(ii,1):filtSpeechIdx(ii,2),1),Fs,wndPar);
    end
%     toc
    %% Train: Obtain the initial Model
    % Use the initial segments from pre-clustering (Overlapped segments that are
    % based upon long-term features)
    % Create HMM for speaker segmentation:|
    %--------------------------------------
%     tic
    HMM_speakerSeg = speakerDiar_SegClustTrain_func(audioData,Fs,Comps,minSize_speakerSeg,speakerIdx,wndPar,g,1);
%     toc
    %% Segmentation:
%     tic
    [Segments,~] = speakerDiar_SegClustSeg_func(HMM_speakerSeg,shortTermFeat,hopSamples,filtSpeechIdx);
    nClusts = length(unique(Segments(:,3)));
    if nClusts~=prevClusts
        reCreateFlag=1; prevClusts=nClusts;
        % Change cluster 'names':
        [~,~,tempIC] = unique(Segments(:,3));
        tempC = 1:nClusts; Segments(:,3) = tempC(tempIC);
    end
    disp('Initial Done');
%     toc
    % %% Plot segments from Core-clustering:
%     plotRegions_func(Segments,gather(audioData));
else
    reCreateFlag = 1;
    nClusts = length(unique(Segments(:,3)));
    prevClusts = nClusts;
    % ASPG:
    speechSeconds = diff(SpeechIdx(:,1:2),[],2)+1;
    speechSeconds = (sum(speechSeconds)-1)/Fs;
    secpergauss = 0.01*speechSeconds+2.6;
    
    g = floor(speechSeconds/(secpergauss*nClusts)); % Number of Gaussians per GM
    
    % Enforce minimum Gaussians:
    minGauss = 3;
    if g<minGauss; g=minGauss;end
end
    %% Core Algorithm:
    mergeIter = 1;
    %%
%     tic
    while mergeIter<=maxMergeIters
        % Re-iterate Training and Segmentation twice before moving to merging:
        for jj=1:1
            % Use the newly segmented data to retrain the GMMs:
            % Get new #ofGaussiansPerGMM
            speechSeconds = diff(Segments(:,1:2),[],2);
            speechSeconds = sum(speechSeconds)/Fs -1/Fs;
            secpergauss = 0.01*speechSeconds+2.6;
%             g = floor(speechSeconds/(secpergauss*nClusts));
%             g = zeros(1,nClusts);
%             for jjj=1:nClusts
%                 speechSeconds = diff(Segments(Segments(:,3)==jjj,1:2),[],2)+1;
%                 speechSeconds = sum(speechSeconds)/Fs -1/Fs;
%                 secpergauss = 0.01*speechSeconds+2.6;
%                 g(jjj) = floor(speechSeconds/(secpergauss)); % Number of Gaussians per GM
%             end
            % tic
%             g=5;
            if reCreateFlag
                HMM_speakerSeg = speakerDiar_SegClustTrain_func(audioData,Fs,nClusts,minSize_speakerSeg,Segments,wndPar,g,reCreateFlag);
                reCreateFlag = 0;
            else
                HMM_speakerSeg.E = speakerDiar_SegClustTrain_func(audioData,Fs,nClusts,minSize_speakerSeg,Segments,wndPar,g,reCreateFlag);
            end
%             toc

            %% Segment Again:
            [Segments,~] = speakerDiar_SegClustSeg_func(HMM_speakerSeg,shortTermFeat,hopSamples,filtSpeechIdx);
            nClusts = length(unique(Segments(:,3)));
            if nClusts~=prevClusts
                reCreateFlag=1; prevClusts=nClusts;
                % Change cluster 'names':
                [~,~,tempIC] = unique(Segments(:,3));
                tempC = 1:nClusts; Segments(:,3) = tempC(tempIC);
            end
            %% Plot segments from Core-clustering:
%             plotRegions_func(Segments,gather(audioData));
        end
        %% Model Merging - BIC Merging:
%         mergeSeg = [];
%         mergeGMM = [];
        kk = 1;
        BICValues = zeros(nClusts*(nClusts-1)/2,3);
        NLogValues = zeros(nClusts*(nClusts-1)/2,3);
        NumIters = zeros(nClusts,1);
        mergeGMM = cell(nClusts*(nClusts-1)/2,1);
        for ii=1:nClusts
            NumIters(ii,1) = HMM_speakerSeg.E{ii}.NumIterations;
            for jj=ii+1:nClusts
                % Create combined model:
                % Merge segments
                mergeSeg = Segments((Segments(:,3)==ii | Segments(:,3)==jj),:);
                mergeSeg(:,3) = 1;
                mergeGMM(kk) = speakerDiar_SegClustTrain_func(audioData,Fs,1,minSize_speakerSeg,mergeSeg,wndPar,2*g,0);
                BICValues(kk) = mergeGMM{kk}.BIC - (HMM_speakerSeg.E{ii}.BIC+HMM_speakerSeg.E{jj}.BIC);
                BICValues(kk,2) = ii; BICValues(kk,3) = jj;
                NLogValues(kk) = mergeGMM{kk}.NegativeLogLikelihood - (HMM_speakerSeg.E{ii}.NegativeLogLikelihood+HMM_speakerSeg.E{jj}.NegativeLogLikelihood);
                NLogValues(kk,2) = ii; NLogValues(kk,3) = jj;
                kk = kk+1;
            end
        end
        %% Stopping Condition:
        %if less than zero -> Merge
        chngSegs = BICValues(BICValues(:,1)<=0,:);
        if ~isempty(chngSegs)
            % Merge
            [~,mergeIdx] = min(chngSegs(:,1));
            Segments(Segments(:,3)==chngSegs(mergeIdx,3),3) = chngSegs(mergeIdx,2);
            nClusts = nClusts-1;
            reCreateFlag=1; prevClusts=nClusts;
            % Change cluster 'names':
            [~,~,tempIC] = unique(Segments(:,3));
            tempC = 1:nClusts; Segments(:,3) = tempC(tempIC);
        else
            % Stop Merging - DONE...
            mergeIter = maxMergeIters;
        end
        mergeIter = mergeIter+1;
    end
end