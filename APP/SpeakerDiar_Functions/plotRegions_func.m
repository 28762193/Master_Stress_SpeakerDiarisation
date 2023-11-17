function plotRegions_func(VADidx,data)
% plotRegions() plots the regions of grouped pairs.

%% Create Masks:
maskLabels = VADidx(:,3);
regionsVAD = VADidx(:,1:2);
msk = signalMask(table(regionsVAD,categorical(maskLabels)));
%% Plot:
figure;
plotsigroi(msk,data,true)
axis([0 numel(data) -1 1]);


end