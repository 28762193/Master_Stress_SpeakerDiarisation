function [feat,AccTime] = createFeaturesAcc_func(bodyMovementAcc,labelarr,Fs)
% This function is mainly meant for database creation of acceleration features:
% createFeaturesAcc is a function that obtains the bodyMovementAcc signal 
% with corresponding labels as input and uses this to create
% features and save the results as a feat.mat table file in the database
% folder.
%% Segmentation:
% Create Windows that will be used in the Feature Extraction section.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BodyMoveAcc_Windowed = segmentSig(bodyMovementAcc,Fs,2,6,50);
WndSize = floor(0.5*Fs); % 0.5sec window
OverlapSize = floor(0.5*WndSize);
Wnd = hamming(WndSize,'periodic');
xbuf = cell(size(bodyMovementAcc,2),1);
 
for ii=1:size(bodyMovementAcc,2)
[xbuf{ii},~] = buffer(bodyMovementAcc(:,ii),WndSize,OverlapSize,"nodelay");
end
xbuf = cell2mat(reshape(xbuf,1,1,[]));
BodyMoveAcc_Windowed = Wnd.*xbuf;
BodyMoveAcc_Windowed = permute(BodyMoveAcc_Windowed,[2 1 3]);
% Time Intervals:
AccTime = zeros(size(BodyMoveAcc_Windowed,1),2);
dt = (WndSize-OverlapSize)/Fs;
AccTime(:,1) = 0:dt:(size(AccTime,1)-1)*dt;
%AccTime(:,2) = dt:dt:size(AccTime,1)*dt;
AccTime(:,2) = (WndSize-1)/Fs:dt:(size(AccTime,1)+1)*dt;
tempTime = (0:size(bodyMovementAcc,1)-1)/Fs;
tempTimeBuf = buffer(tempTime,WndSize,OverlapSize,"nodelay");
tempTimeBuf = tempTimeBuf';
bufSize = floor((size(bodyMovementAcc,1)-OverlapSize)/(WndSize-OverlapSize));
tempTimeBuf = tempTimeBuf(1:bufSize,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Segment Labels and Delete Windows:
% Segment the label array and deleting observations if a clear decision on
% the label cannot be reached.
if ~isempty(labelarr)
    [lBuf,~] = buffer(labelarr,WndSize,OverlapSize,"nodelay");
    labelarr_Windowed = zeros(size(lBuf,2),1);
    lCounter = 1;
    for ii=1:size(lBuf,2)
        u = unique(lBuf(:,ii));
        uCnts = histcounts(lBuf(:,ii));
        if any(uCnts>floor(0.70*WndSize))
            % Keep Window
            labelarr_Windowed(lCounter,1) = u(uCnts>floor(0.70*WndSize));
            lCounter = lCounter+1;
        else
            % Remove Window
            labelarr_Windowed(lCounter,:) = [];
            BodyMoveAcc_Windowed(lCounter,:,:) = [];
        end
    end
end
%% Feature Extraction:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Energy of the bodymovement/linear acceleration.
% Absolute Mean of the linear acceleration.
% Std of the linear acceleration. - Might not use, might be damaging.
% Max of the linear acceleration. - Not in use (susceptible to outliers)
% Min of the linear acceleration. - Not in use (susceptible to outliers)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate the Energy of the Signal (per Window):
% RMS Method:
% Ener_rms = zeros(size(BodyMoveAcc_Windowed,1),size(BodyMoveAcc_Windowed,3));
% f_Body_Ener = Ener_rms;
% for i=1:size(Ener_rms,2)
%     Ener_rms(:,i) = rms(BodyMoveAcc_Windowed(:,:,i),2);
%     f_Body_Ener(:,i) = (Ener_rms(:,i).^2)*size(BodyMoveAcc_Windowed,2);
% end

% Norm2 Method:
ft_Body_EnerAve = zeros(size(BodyMoveAcc_Windowed,1),size(BodyMoveAcc_Windowed,3));

for j=1:size(ft_Body_EnerAve,2)
    ft_Body_EnerAve(:,j) = vecnorm(BodyMoveAcc_Windowed(:,:,j),2,2).^2;
end
ft_Body_EnerAve = sum(ft_Body_EnerAve,2)/WndSize;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mean:
ft_Body_absMean = zeros(size(BodyMoveAcc_Windowed,1),size(BodyMoveAcc_Windowed,3));
for j=1:size(ft_Body_absMean,2)
    ft_Body_absMean(:,j) = mean(abs(BodyMoveAcc_Windowed(:,:,j)),2);
end
ft_Body_absMean = sum(ft_Body_absMean,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Std:
ft_Body_Std = zeros(size(BodyMoveAcc_Windowed,1),size(BodyMoveAcc_Windowed,3));
for j=1:size(ft_Body_Std,2)
    ft_Body_Std(:,j) = std((BodyMoveAcc_Windowed(:,:,j)),0,2);
end
ft_Body_Std = sum(ft_Body_Std,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Move Features to a table for saving:
if ~isempty(labelarr)
feat = table(ft_Body_EnerAve,ft_Body_absMean,ft_Body_Std,labelarr_Windowed,'VariableNames',{'BodyEnergy','BodyAbsMean','BodyStd','Label'});
else
    feat = table(ft_Body_EnerAve,ft_Body_absMean,ft_Body_Std,'VariableNames',{'BodyEnergyAve','BodyAbsMean','BodyStd'});
%EOF
end