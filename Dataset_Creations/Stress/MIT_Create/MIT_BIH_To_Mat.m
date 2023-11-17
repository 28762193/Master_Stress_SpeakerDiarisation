% Matlab script to create .mat files from wfdb .dat files of the MIT-BIH 
% Arrhythmia Database
recordNum = load('RECORDS');
recordLoc = 'mitdb/';
recordName = convertStringsToChars(strcat(recordLoc,string(recordNum)));
BeatAnnList = {'N','L','R','A','a','J','S','V','F','e','j','E','/','f','Q'};
fileLoc = '/database/MIT-BIH/';
%%
%temp = cell(48,1);
for i=1:length(recordName)
    [Sig,Fs,Time] = rdsamp(recordName{i},1);
    [Ann,AnnType] = rdann(recordName{i},'atr');

    BeatAnnTable = table(Ann,AnnType,'VariableNames',{'AnnTime','AnnType'});
    BeatAnnTable(not(ismember(BeatAnnTable.AnnType,BeatAnnList)),:) = [];
 
    m = struct('Fs',Fs,'Sig',Sig,'Time',Time,'BeatAnnTable',BeatAnnTable);
    fileName = strcat(fileLoc,recordName{i},'m.mat');
    save(fileName,'-struct','m');
    %temp{i} = {AnnType(not(ismember(AnnType,BeatAnnList)))};
end
%%
% figure;
% plot(Time,Sig);
% hold on;
% grid on;
% plot(Time(BeatAnnTable.AnnTime),Sig(BeatAnnTable.AnnTime),'ro','MarkerSize',5);
% hold off;