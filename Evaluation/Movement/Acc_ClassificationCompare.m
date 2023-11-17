% Classification Comparisons:
% This script compares different classification models against one another.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load Data:
data = load("featAcc_Rev2.mat");
names = fieldnames(data);
data = data.(names{:});

%% Partition Data:
cv = cvpartition(size(data,1),"HoldOut",0.2);
idx = cv.test;
dataTrain = data(~idx,:);
dataTest = data(idx,:);
%% ML Model:
% Train Model:
tic
[AccModelTrained,AccVal] = trainAccClassifier_LinSVM_func(dataTrain);
toc
%% Test Model:
tic
yfit = AccModelTrained.predictFcn(dataTest);
toc
% True Positive, False Positive and False Negative:
testLabels = dataTest.Label;
TP_ML = size(yfit(yfit==testLabels & testLabels==1),1);
FP_ML = size(yfit(yfit~=testLabels & yfit==1),1);
FN_ML = size(yfit(yfit~=testLabels & yfit==0),1);
totTrue_ML = size(yfit(yfit==testLabels),1);
testLabSize = size(testLabels,1);
disp([TP_ML FP_ML FN_ML totTrue_ML testLabSize]);
%% Threshold Method:
% Define Thresholds: (Obtained empirically)
tic
TM_Para.enerThres = 0.8/100;
TM_Para.absMeanThres = 0.1;
TM_Para.stdThres = 0.12;
yfit = zeros(size(dataTest,1),1);
for ii=1:size(dataTest,1)
    testArr = [dataTest{ii,"BodyEnergyAve"}>=TM_Para.enerThres, dataTest{ii,"BodyAbsMean"}>=TM_Para.absMeanThres,...
        dataTest{ii,"BodyStd"}>=TM_Para.stdThres];
    if size(testArr(testArr==1),2)>=2
        yfit(ii) = 1;
    else
        yfit(ii) = 0;
    end
end
toc
% True Positive, False Positive and False Negative:
testLabels = dataTest.Label;
TP_TM = size(yfit(yfit==testLabels & testLabels==1),1);
FP_TM = size(yfit(yfit~=testLabels & yfit==1),1);
FN_TM = size(yfit(yfit~=testLabels & yfit==0),1);
totTrue_TM = size(yfit(yfit==testLabels),1);
disp([TP_TM FP_TM FN_TM totTrue_TM testLabSize]);
%% Plot Regions:
samples = (1:size(yfit,1))';
% Find changes in the labels:
abruptChg = 0;
% yfit = testLabels;
if abruptChg
chgPnt1 = zeros(size(yfit));
for ii=2:size(yfit,1)
    chgnd = isequal(yfit(ii-1),yfit(ii));
    if ~chgnd
        chgPnt1(ii) = 1;
    end
end
else 
    chgPnt1 = ischange(yfit,'linear');
end
chgPnt2 = find(chgPnt1)-1;
chgPnt1(1) = 1; chgPnt2(end+1) = size(samples,1);
chgPnt1 = find(chgPnt1);
idxClrs = yfit(chgPnt1,1)+1;

xCoords = [samples(chgPnt1,1) samples(chgPnt2,1) samples(chgPnt2,1) samples(chgPnt1,1)];
yCoords = zeros(size(xCoords,1),4);
yCoords(:,1:2) = min(min(dataTest{:,1:3}));
yCoords(:,3:4) = max(max(dataTest{:,1:3}));

clrs = {[0 0.4470 0.7410], "green"}; % Blue and Green - Non-movement and Movement.
labelMovement = {'NonMove','Movement'};

figure;
hAccLine = plot(dataTest{:,1:3},'Color','Black'); % Plot triaxial accelerometer versus time.
lines = {'-';':';'-.'};
[hAccLine(:).LineStyle] = lines{:};
hAccAx = ancestor(hAccLine,'axes');
hold on;
for ii=1:size(xCoords,1) % Plot regions
    patch(hAccAx{1},xCoords(ii,:),yCoords(ii,:),clrs{idxClrs(ii,1)},'FaceAlpha',.3,'EdgeColor',clrs{idxClrs(ii,1)},'LineWidth',1);
end
legend('bEner','bAbsMean','bStd',labelMovement{unique(idxClrs)});
title('Test Accelerometer Label Classification')
%% Save ML Model:
save('AccMovementModel_Rev2.mat','AccModelTrained');
%% Save TM Parameters:
save('TM_Para.mat','TM_Para');
