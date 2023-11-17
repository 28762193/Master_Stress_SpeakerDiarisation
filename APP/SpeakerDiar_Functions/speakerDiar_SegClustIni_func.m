function [speakerIdx,Comps,g] = speakerDiar_SegClustIni_func(audioIn,Fs,VADidx,wndPar,minDur)
% This function performs the initialisation step for the segmentation and
% clustering part of the speaker diarization algorithm. It entails
% pre-clustering and ASPG.
% Pre-clustering obtains the initial number of clusters (speakers - 
% overestimated, ideally) and provides an initial speaker-segmentation of
% the audio.
% ASPG - Adaptive seconds per Gaussian: Gives an indication of the ideal
% number of Gaussians to be used per cluster given the amount of available
% speech and the number of clusters to be used.
%--------------------------------------------------------------------------
% INPUT: audioIn=audio track
%        Fs=sampling rate
%        VADidx=voice activity detection indices
%        wndPar=window parameter (struct)
%        minDur=minimum duration to be used in long-term feature extraction

% OUTPUT: speakerIdx=cluster regions
%         g=number of gausssians per cluster
%--------------------------------------------------------------------------
%% 1.Pre-Clustering:
% Obtain long-term features:
[longTermFeat_Tab,VADidx_Speech] = speakerDiar_longTermFE_func(audioIn,Fs,VADidx,wndPar,minDur);
featArr = table2array(longTermFeat_Tab);

%% 1.Pre-Clustering:
% Use long-term features for model creation.
n_ini = 2; % Initial number of Gaussians (One for a single speaker and one for background noise)
n_max = 20; % Maximum allowed number of Gaussians to try
n_mid = 6; % Check atleast n_mid amount of Gaussians
% The amount of Gaussians will provide the initial amount of speaker
% clusters.

[GMM_preCluster,Comps] = speakerDiar_OptiGauss_func(featArr,n_ini,n_max,n_mid);

%% Create univariant GMMs:
gmm_singles = cell(1,Comps);
for ii=1:Comps
    gmm_singles{ii} = gmdistribution(GMM_preCluster.mu(ii,:),GMM_preCluster.Sigma(1,:,ii));
end
%% Initial Segmentation:
% Get best fitting model logpdfs:
Nlogpdf = zeros(size(featArr,1),Comps);
for ii=1:Comps
    Nlogpdf(:,ii) = -1*log(gmm_singles{ii}.pdf(featArr));
end
[~,speakerIdx] = min(Nlogpdf,[],2);
speakerIdx = [VADidx_Speech,speakerIdx];

%% 2.ASPG:
speechSeconds = diff(VADidx_Speech(:,1:2),[],2)+1;
speechSeconds = sum(speechSeconds)/Fs -1/Fs;
secpergauss = 0.01*speechSeconds+2.6;

g = floor(speechSeconds/(secpergauss*Comps)); % Number of Gaussians per GM

% Enforce minimum Gaussians:
minGauss = 3;
if g<minGauss; g=minGauss;end
end