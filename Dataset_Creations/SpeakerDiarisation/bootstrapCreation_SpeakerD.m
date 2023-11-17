% Bootstrap Creation:
% The bootstrap component is a rough HMM containing 2 string-states one for
% silence and sound, and the other for speech. Each string holds a unique
% GMM for that string that is shared among its substates.
% This script pretrains the GMM models and forms the HMM.
%% Set Parameters:
testGMMs = 0; % 1=Find best mixture for each model
FitGMMs = 1; % 1=Fit features to GMMs
compSize_Speech = 12; % Number of components/mixtures for speech GMM
compSize_Silence = 6; % Number of components/mixtures for silence GMM
compSize_Noise = 8; % Number of components/mixtures for noise GMM
if testGMMs; FitGMMs = 0;end
%% Load Training data:
% Move to current running script:
filePath = matlab.desktop.editor.getActiveFilename;
cd(fileparts(filePath));

load('allFeatures_Speech.mat');
load('normFactors_Speech.mat');
load('allFeatures_Silence.mat');
load('normFactors_Silence.mat');
load('allFeatures_Noise.mat');
load('normFactors_Noise.mat');

% Create a single struct for allFeatures and normFactors:
AllFeatures.Speech = allFeatures_Speech;
AllFeatures.Silence = allFeatures_Silence;
AllFeatures.Noise = allFeatures_Noise;

normFactors.Speech = normFactors_Speech;
normFactors.Silence = normFactors_Silence;
normFactors.Noise = normFactors_Noise;

clear allFeatures_Speech allFeatures_Silence allFeatures_Noise
clear normFactors_Speech normFactors_Silence normFactors_Noise

%% Test Different Mixtures for each Model:
% Tested speech model with 4:2:16 components - 12 proven to be okay with
% much less iters than 12. 8 components will also suffice.
% Silence Model - 2:2:8 - 6 proven to be good enough with good AIC and min
% difference between AICs.
% Noise Model - 2:2:10 - 8 proven to be okay with good AIC

% !!!!!!The lower the AIC value the better, and a delta-AIC of more than -2 is
% considered significantly better than the model it is being compared
% to.!!!!!!

if testGMMs
    tic
    %[~,X] = pca(AllFeatures.Noise');
    X = AllFeatures.Speech';
    compSize = 4:2:16;
    AIC = zeros(1,size(compSize,2));
    GMModels = cell(1,size(compSize,2));
    options = statset('Display','final','MaxIter',250);
    
    for k = 1:size(compSize,2)
        GMModels{k} = fitgmdist(X(1:60e3,:),compSize(k),'Options',options,'CovarianceType','diagonal');
        AIC(k)= GMModels{k}.AIC;
    end
    
    [minAIC,numComponents] = min(AIC);
    AIC
    diff(AIC)
    toc
end


%% Fit GMMs:
% Observation Parameters: HMM
% The observation parameters are the GMMs that will be used to generate the
% observation-likelihood/emission-probability.

if FitGMMs
    tic
    options = statset('MaxIter',250);
    GMM_Speech = fitgmdist(AllFeatures.Speech',compSize_Speech,...
        'CovarianceType','diagonal','Options',options);
    GMM_Silence = fitgmdist(AllFeatures.Silence',compSize_Silence,...
        'CovarianceType','diagonal','Options',options);
    GMM_Noise = fitgmdist(AllFeatures.Noise',compSize_Noise,...
        'CovarianceType','diagonal','Options',options);
    toc
    save('GMM_Speech.mat','GMM_Speech');
    save('GMM_Silence.mat','GMM_Silence');
    save('GMM_Noise.mat','GMM_Noise');
else 
    load('GMM_Speech.mat'); load('GMM_Silence.mat'); load('GMM_Noise.mat');
end
EPar.Speech = GMM_Speech; % Emission Parameters for HMM
EPar.Sil = GMM_Silence;
EPar.Noise = GMM_Noise;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HMM:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters: Model-Based Method
N_Speech = 75; % Number of substates for each string
N_Silence = 25; % The number of substates enforces a min duration for each class 
N_Noise = 25; % 75 substates = 750ms min duration (with 10ms hop).
N_States = N_Speech + N_Silence + N_Noise;

string_begin = [1,N_Speech+1,N_Speech+N_Silence+1];
string_end = [N_Speech,N_Speech+N_Silence,N_States];

% Initialise PI (initial probability distribution):
init_prob = 1/length(string_begin);
pi_int = zeros(N_States,1); 
pi_int(1,1) = init_prob; % First state in Speech string
for i=2:length(string_begin)
    pi_int(string_begin(i),1) = init_prob;
end

% Create Transition Matrix:
% row->column transitions
A_trans = zeros(N_States,N_States);
for i=1:N_States
    if ~isempty(string_end(string_end==i))
        A_trans(i,i) = 1/(length(string_begin)+1);
        for j=1:length(string_begin)
            A_trans(i,string_begin(j)) = 1/(length(string_begin)+1);
        end
    else
        A_trans(i,i+1) = 1;
    end
end

% Bootstap Component:
HMM_Bootstrap.s = pi_int; % Start probability vector
HMM_Bootstrap.A = A_trans; % Transition matrix
HMM_Bootstrap.E = EPar; % Emission parameters (GMMs)
save('HMM_Bootstrap.mat','-struct',"HMM_Bootstrap");

