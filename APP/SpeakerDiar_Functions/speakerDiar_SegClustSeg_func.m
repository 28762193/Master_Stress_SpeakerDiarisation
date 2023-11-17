function [segments,llh] = speakerDiar_SegClustSeg_func(HMM,Feat,hopSamples,speechIdx)
% This function executes the segmentation procedure of the Segmentation
% and Clustering section of the Speaker Diarization algorithm.
% INPUT: HMM=Hidden Markov Model containing an initial and transition
% matrices, as well as GMMs per cluster as to obtain the emission matrix.
%        Feat=Feature set to be segmented
%        hopSamples=sample size of each feature window.
%        speechIdx=speech indices as reference to the original audio track.

% OUTPUT: segments=the segmentation of the feature set.
%         llh=likelihood of each segmentation.


segments = cell(size(Feat,2),1);
llh = cell(size(Feat,2),1);

% q = parallel.pool.DataQueue;
% afterEach(q,@disp);
for ii=1:size(Feat,2)
    [segments{ii},llh{ii}] = hmmViterbi_func(HMM,Feat{ii});
%     send(q,ii);
    segments{ii}(1,1) = speechIdx(ii,1);
    segments{ii}(1,2) = (segments{ii}(1,2)+2)*hopSamples+segments{ii}(1,1)-1;
    for jj=2:size(segments{ii},1)
        segments{ii}(jj,1) = segments{ii}(jj-1,2)+1;
        segments{ii}(jj,2) = (segments{ii}(jj,2)+2)*hopSamples+segments{ii}(1,1)-1;
    end
end
segments = cell2mat(segments);
%EOF
end