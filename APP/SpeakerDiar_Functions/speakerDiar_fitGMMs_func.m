%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------fitGMMs_func-------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function varargout = speakerDiar_fitGMMs_func(forceFit,varargin)
% This function fits Feat to a GMM model with maxComp components, from
% {{Feat};{maxComp}}. forceFit enforces the use of the number components to
% be used given by maxComp.
% INPUT: varargin -> {{SilFeat};{maxSilComp}}, {{NoiseFeat};{maxNoiseComp}}, {{SpeechFeat};{maxSpeechComp}}
% OUTPUT: varargout (depends on varargin) -> SilGMM, NoiseGMM, SpeechGMM

options = statset('MaxIter',350);
varargout = cell(1,nargin-1);

for i=1:nargin-1
    Feat = varargin{i}{1};
    if ~isempty(Feat)
        maxComp = varargin{i}{2};
        if ~forceFit
            Comp = round(size(Feat,1)/(size(Feat,2)*4));
            if Comp<2;Comp=2;elseif Comp>maxComp; Comp=maxComp;end
        else 
            Comp = maxComp;
        end
        varargout{i} = fitgmdist(Feat,Comp,'CovarianceType','diagonal','Options',options,'Replicates',3);
    else
        varargout{i} = [];
    end
end
%EOF
end