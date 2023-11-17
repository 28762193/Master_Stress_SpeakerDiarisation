%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------------------hmmViterbi_func------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [segmentSSN, llh] = hmmViterbi_func(model, x)
% Viterbi algorithm (calculated in log scale to improve numerical stability).
% Input:
%   x: o x f feature matrix (obs x features)
%   model: model structure which contains
%       model.s: n x 1 start probability vector
%       model.A: n x n transition matrix (row->column)
%       model.E: M GMMs (M=3)
% Output:
%   z: 1 x o latent state
%   llh:  loglikelihood
% o -> #obs; f-> #features; n -> #states; M -> #classes (Speech, Silence, Noise)
    o = size(x,1); % #observations
    n = size(model.A,1); % #states

    % Sparse matrix model.s for efficiency
    s = log(model.s);
    A = log(model.A);
    B = zeros(n,o);
    % Get number of GMMs:
    k = 1;

    while k <= length(model.E)
        if isempty(model.E{k})
            model.E(k) = [];
        elseif isa(model.E{k},"gmdistribution"); k=k+1;
        end
    end
    k = k-1;

    %Get logpdfs from GMMs
    logpdf = cell(1,k);
    % [idx,nlogpdf,P,logpdf,d2] = cluster():
    for i=1:k
        [~,~,~,logpdf{i}] = cluster(model.E{i},x);
    end
    % Fill Emission Probability for all states (shared-states):
    temp = find(model.s ~= 0);
    for i=1:length(temp)-1
        B(temp(i):temp(i+1)-1,:) = repmat(logpdf{i}',temp(i+1)-temp(i),1);
    end
    B(temp(end):end,:) = repmat(logpdf{end}',size(B,1)-temp(end)+1,1);
   %---------------------------------------------------------------
    %Enforce change at the end: As to ensure the minimum duration is
    %enforced at the end of the audio track.
%     tempMaxregions = B(:,end-1)==max(B(:,end-1));
%     B(tempMaxregions,end) = -inf;
    %----------------------------------------------------------------
    % Viterbi:|
    % ---------
    % Initialisation:
    Delt = zeros(n,o);
    Delt(:,1) = 1:n;
    v = s(:) + B(:,1);
    % Inductive Step:
    for t=2:o
        [v,idx] = max(bsxfun(@plus,A,v),[],1);
        v = v(:)+B(:,t);
        Delt = Delt(idx,:);
        Delt(:,t) = 1:n;
    end
    % Termination:
    [llh,idx] = max(v);
    z = Delt(idx,:);

    % Further Segmentation (Segment after Viterbi):|
    % ----------------------------------------------
    % Assign labels to states:
    temp = find(model.s ~= 0);

    for i = 1:length(temp)-1
        z(z>=temp(i) & z<temp(i+1)) = i;
    end
    z(z>=temp(end)) = length(temp);

    z(1,end) = z(1,end-1);
    % Enforce minimum duration:
    chgPoint = find(ischange(z));
    if ~isempty(chgPoint)
        tempDiff = diff(temp);
        tempDiff(end+1) = size(s,1)-temp(end)+1;
        tempChgSize = zeros(numel(chgPoint)+1,1);
        tempChgSize(1) = chgPoint(1);
        for i=2:numel(chgPoint)
            tempChgSize(i) = chgPoint(i)-chgPoint(i-1);
        end
        tempChgSize(end) = numel(z)-chgPoint(end)+1;
        for i=1:numel(tempChgSize)-1
            if tempChgSize(i)<tempDiff(z(chgPoint(i)-1))-5
                % Too short...
                z(chgPoint(i):chgPoint(i+1)) = z(chgPoint(i)-1);
            end
        end
        if tempChgSize(end)<tempDiff(z(end))-5
            % Too short...
            z(chgPoint(end):end) = z(chgPoint(end)-1);
        end
    end
    chgPoint = find(ischange(z));
    segmentSSN = zeros(length(chgPoint)+1,3); % Segment Array: Begin|End|Label
    segmentSSN(1,1) = 1;
    segmentSSN(2:end,1) = chgPoint;
    segmentSSN(1:end-1,2) = chgPoint-1; segmentSSN(end,2) = o;
    segmentSSN(:,3) = z(segmentSSN(:,2));
%EOF
end