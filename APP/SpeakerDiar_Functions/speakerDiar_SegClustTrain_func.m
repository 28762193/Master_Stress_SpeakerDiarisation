function HMM = speakerDiar_SegClustTrain_func(audioIn,Fs,nClusts,minSize,speakerIdx,wndPar,g,reCreateFlag)
% This function trains an HMM for the speaker segmentation.
% INPUT: audioIn=audio track,
%        Fs=sampling rate,
%        nClusts=number of clusters,
%        speakerIdx=speaker regions belonging to a cluster,
%        wndPar=window parameter,
%        g=number of Gaussians per GMM/cluster,
%        reCreateFlag=re-create HMM's initial and transition matrix (flag).

% OUTPUT: HMM=HMM model (struct) - contains initial matrix, transition
% matrix, and a GMM per cluster (possible speaker) as to estimate the
% emision matrix.
%--------------------------------------------------------------------------
if reCreateFlag
    % Create HMM with initial and transition matrices:
    HMM = speakerDiar_HMMCreate_func(nClusts,minSize);
end

if isscalar(g)
    g = ones(1,nClusts)*g;
end

% Create GMMs per Cluster:
GMMs_speakerSeg = cell(1,nClusts);
gmmCount = 1;
for ii=1:nClusts
%     tempAudioSeg = [];
    tempRanges = speakerIdx(speakerIdx(:,3)==ii,1:2);
    tempAudioSeg = cell(size(tempRanges,1),1);
    for jj=1:size(speakerIdx(speakerIdx(:,3)==ii),1)
        tempAudioSeg{jj} = audioIn(tempRanges(jj,1):tempRanges(jj,2));
    end
    tempAudioSeg = cat(1,tempAudioSeg{:});
    % Obtain short-term features:
    Feats_speakerSeg = speakerDiar_shortTermFE_func(tempAudioSeg,Fs,wndPar);
    if (size(Feats_speakerSeg,1) < size(Feats_speakerSeg,2)); Feats_speakerSeg = [];end
    if ~isempty(Feats_speakerSeg)
    % Obtain GMMs:
        GMMs_speakerSeg{gmmCount} = speakerDiar_fitGMMs_func(1,{Feats_speakerSeg,g(ii)});
        gmmCount = gmmCount+1;
    else
        GMMs_speakerSeg(gmmCount) = [];
    end

end
if ~reCreateFlag
    HMM = GMMs_speakerSeg'; % Add GMMs to HMM (For Resegmentation and Retraining)
else
    HMM.E = GMMs_speakerSeg'; % Add GMMs to HMM (For Resegmentation and Retraining)
end
%EOF
end