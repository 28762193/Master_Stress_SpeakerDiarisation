function [processedAcc,bodyAcc_Summed,timeAcc_Seg,moveYN] = polarMoveClassi_func(ACC_tab_Cell,AccTimeStamps)
% Movement-Classification/Identification:
% Function Script for bodymovement classification.
%% Parameters:
processedAcc = cell(size(ACC_tab_Cell));
bodyAcc_Summed = cell(size(ACC_tab_Cell));
timeAcc_Seg = cell(size(ACC_tab_Cell));
moveYN = cell(size(ACC_tab_Cell));
% Classification Parameters:
AccMovementModel_LinSVM = load('AccMovementModel_Rev2.mat');
varName = fieldnames(AccMovementModel_LinSVM);
varName = varName{1};
AccMovementModel_LinSVM = AccMovementModel_LinSVM.(varName);
for ii=1:size(ACC_tab_Cell,1)
%% Perform pre-processing:
% The acceleration data is given and should be in g's (gravitational
% force).
% The sampling rate must be 50 Hz or higher.

% Determine Sample Rate:
Fs = round(1/mean(seconds(diff(AccTimeStamps{ii}))),2);
% ACC Data given in mG's:
[totalAcc,bodyMovementAcc,~,AccTimeStamps{ii},Fs] = preProcessingAcc_func(ACC_tab_Cell{ii}/1000,AccTimeStamps{ii},Fs);
processedAcc{ii} = totalAcc;
bodyAcc_Summed{ii} = sum(abs(bodyMovementAcc),2);
%% Get Features:
[featAcc,timeAcc] = createFeaturesAcc_func(bodyMovementAcc(:,1:3),[],Fs);
timeAcc = seconds(timeAcc)+AccTimeStamps{ii}(1);
timeAcc_Seg{ii} = timeAcc;
%% Classify Movement:
moveFit = AccMovementModel_LinSVM.predictFcn(featAcc);
%% Threshold Method:
% % Define Thresholds: (Obtained empirically)
% load('TM_Para.mat');
% moveFit = zeros(size(featAcc,1),1);
% for ii=1:size(featAcc,1)
%     testArr = [featAcc{ii,"BodyEnergyAve"} >=TM_Para.enerThres, featAcc{ii,"BodyAbsMean"}>=TM_Para.absMeanThres,...
%         featAcc{ii,"BodyStd"}>=TM_Para.stdThres];
%     if size(testArr(testArr==1),2)>=2
%         moveFit(ii) = 1;
%     else
%         moveFit(ii) = 0;
%     end
% end
moveYN{ii} = moveFit;

% %% Plot Results:
% t = AccTimeStamps{ii};
% 
% % Find changes in the labels:
% abruptChg = 1;
% 
% if abruptChg
% chgPnt1 = zeros(size(moveFit));
% for jj=2:size(moveFit,1)
%     chgnd = isequal(moveFit(jj-1),moveFit(jj));
%     if ~chgnd
%         chgPnt1(jj) = 1;
%     end
% end
% else 
%     chgPnt1 = ischange(moveFit,'linear');
% end
% chgPnt2 = find(chgPnt1)-1;
% chgPnt1(1) = 1; chgPnt2(end+1) = size(timeAcc,1);
% chgPnt1 = find(chgPnt1);
% idxClrs = moveFit(chgPnt1,1)+1;
% 
% 
% xCoords = [timeAcc(chgPnt1,1) timeAcc(chgPnt2,2) timeAcc(chgPnt2,2) timeAcc(chgPnt1,1)];
% yCoords = zeros(size(xCoords,1),4);
% yCoords(:,1:2) = min(min(totalAcc));
% yCoords(:,3:4) = max(max(totalAcc));
% 
% clrs = {[0 0.4470 0.7410], "green"}; % Blue and Green - Non-movement and Movement.
% labelMovement = {'NonMovement','Movement'};
% figure;
% hAccLine = plot(t,totalAcc,'Color','Black'); % Plot triaxial accelerometer versus time.
% lines = {'-';':';'-.'};
% [hAccLine(:).LineStyle] = lines{:};
% hAccAx = ancestor(hAccLine,'axes');
% hold on;
% for jj=1:size(xCoords,1) % Plot regions
%     patch(hAccAx{1},xCoords(jj,:),yCoords(jj,:),clrs{idxClrs(jj,1)},'FaceAlpha',.3,'EdgeColor',clrs{idxClrs(jj,1)},'LineWidth',1);
% end
% legend('X','Y','Z',labelMovement{unique(idxClrs)});
% xlabel('Time [sec]');
% ylabel("Acceleration in g's [g=9.80665 m/s^2]");
% title('Triaxial Acceleration')

%EOL
end
%EOF
end