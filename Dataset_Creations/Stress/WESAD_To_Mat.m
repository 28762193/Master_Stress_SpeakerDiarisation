% Save WESAD .csv data as .mat:
% A python script is used to convert the pickle (.pkl) file to a .csv file
% - only the required signals were extracted. The python script can be
% found in: 
% /main.py
% It is self written and makes use of pandas, numpy, and glob.

fileLoc = '/Datasets/WESAD/MAT_Files/';
cd(fileLoc);
fileList = dir('S*.csv');
fileList = {fileList.name}';
fileName = strcat(fileLoc,fileList);
fileName2 = strrep(fileName,'.csv','');
%%
for i=1:length(fileList)
    raw_data = readtable(fileName{i},"VariableNamingRule","preserve");
    save(fileName2{i},"raw_data");
end
clear all;
