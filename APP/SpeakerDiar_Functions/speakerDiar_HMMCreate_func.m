function HMM = speakerDiar_HMMCreate_func(nStates,varargin)
% This function creates a model-based HMM with equal initial probability
% and enforced minimum duration per state.
% INPUT: nStates=number of states;
%        varargin=variable input for enforced minimum state duration: 
%       single input - equal number of substates for each state;
%       multiple inputs - enforced segment duration of each state
%       (length(multipleInputs)=nStates).
% OUTPUT: HMM=HMM struct containing the initial probability and transition
% matrix.

N_Values = zeros(1,nStates);
if length(varargin) == 1
    N_Values(1:end) = varargin{1};
elseif length(varargin) == nStates
    N_Values = cell2mat(varargin);
else
    error('Error: Number of arguments are not equal to the number of states');
end

N_Values = N_Values(N_Values>0);
N_States = sum(N_Values);
string_begin = zeros(1,length(N_Values));
string_end = zeros(1,length(N_Values));
string_begin(1) = 1; string_end(1) = N_Values(1);

for i=2:nStates
    string_begin(i) = string_begin(i-1)+N_Values(i-1);
    string_end(i) = string_end(i-1)+N_Values(i);
end


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
HMM.s = pi_int; % Start probability vector
HMM.A = A_trans; % Transition matrix

%EOF
end