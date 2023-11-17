function quart = quartile_func(Data,p)
% Quartile function to obtain divitions
% INPUT: Data=input data in the form of a vector,
%        p=percentile as a decimal.
% OUTPUT: quart=quartile value.

% Order data in ascending order:
if ~issorted(Data)
    Data = sort(Data);
end

n = length(Data);
if mod(n+1,1/p) ~= 0
    m = (n+1)*p;
    pos = floor(p*(n+1));
    quart = Data(pos)+(Data(pos+1)-Data(pos))*(m-floor(m));
else 
    quart = Data(floor(p*(n+1)));
end
%EOF
end