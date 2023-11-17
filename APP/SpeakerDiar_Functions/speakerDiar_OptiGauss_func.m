function [GMM,Comps] = speakerDiar_OptiGauss_func(Feat,iniComps,maxComps,minCompCheck)
% This function determines the best number of gaussians given the
% obtained features. This is achieved by using the BICriteria.
% INPUT: Feat=feature set;
%        iniComps=initial number of Components to test from.
%        maxComps=maximum number of Components to be tested.

if isempty(iniComps); iniComps=2;end
if isempty(maxComps); maxComps=50;end
if isempty(minCompCheck); minCompCheck=5+1;end


n = iniComps; % Initial number of Gaussians
tempGMM = cell(1,minCompCheck-iniComps+1);
tempBIC = zeros(1,minCompCheck-iniComps+1);

for ii=n:minCompCheck
    try
        tempGMM{ii-1} = speakerDiar_fitGMMs_func(1,{Feat,ii});
        tempBIC(ii-1) = tempGMM{ii-1}.BIC;
    catch
        tempGMM{ii-1} = [];
        tempBIC(ii-1) = inf;
    end
end
[minBIC,minIdx] = min(tempBIC);
%Comps = minIdx+n-1;
GMM = tempGMM{minIdx}; % +1 for possible background noise
% Compare further for possible improvement:
GMM_BICCompare = minBIC;
Iter = minCompCheck+1;
prevGMM = GMM;
while Iter <= maxComps
    try
        GMM = speakerDiar_fitGMMs_func(1,{Feat,Iter});
        if ~(GMM.BIC < GMM_BICCompare)
            Iter = maxComps+1;
            GMM = prevGMM;
        else 
            GMM_BICCompare = GMM.BIC;
            prevGMM = GMM;
            Iter = Iter+1;
        end
    catch
        % Catch the fitgmdist() error
        Iter = maxComps+1;
    end
end
Comps = GMM.NumComponents;
%EOF
end