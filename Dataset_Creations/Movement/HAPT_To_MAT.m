% HAPT_To_MAT creates a feature and label table for Human Activity 
% Recognition (HAR) ML capabilities. The dataset used is the Smartphone-Based 
% Recognition of Human Activities and Postural Transitions Data Set
% Version 2.1 from: 
% =========================================================================
% Jorge L. Reyes-Ortiz(1,2), Davide Anguita(1), Luca Oneto(1) and Xavier Parra(2)
% 1 - Smartlab, DIBRIS - Universit‡  degli Studi di Genova, Genoa (16145), Italy. 
% 2 - CETpD - Universitat PolitËcnica de Catalunya. Vilanova i la Geltr˙ (08800), Spain
% har '@' smartlab.ws 
% www.smartlab.ws
% =========================================================================
clear all;
%% Load Data From HAPT Dataset:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fileLoc = '/Datasets/HAPT Data Set/RawData/';
cd(fileLoc);
fileList = dir('acc_exp*.txt');
fileList = {fileList.name}';
fileName = strcat(fileLoc,fileList);

% Load label data:
fileLabel = 'labels.txt';
fileName2 = strcat(fileLoc,fileLabel);
rawLables = readmatrix(fileName2); % Obtain mixed labels and setpoints per file.

Fs = 50; % Sampling rate
rawAccData = [];

for i=1:length(fileName)
    temp = readmatrix(fileName{i});
    labelarr = zeros(length(temp),1);
    % Obtain only the required label data for the current file:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    curExpSub = str2double(regexp(fileName{i},'\d*','Match'))';
    labels = rawLables(rawLables(:,1)==curExpSub(1) & rawLables(:,2)==curExpSub(2),3:5);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Create label array: Match labels to corresponding data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for j=1:length(labels)
        labelarr(labels(j,2):labels(j,3)) = labels(j,1);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Remove unlabelled data:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    temp(labelarr(:,1)==0,:) = [];
    labelarr(labelarr(:,1)==0,:) = [];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    temp(:,end+1) = labelarr; % Add label to rawAccData matrix
    rawAccData = cat(1,rawAccData,temp); % Concat raw triaxial accelerometer data (x,y,z)
end

clear fileLoc fileList fileName fileLabel fileName2 rawLables temp labelarr curExpSub labels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Move to current running script:
filePath = matlab.desktop.editor.getActiveFilename;
cd(fileparts(filePath));
%% Perform pre-processing:
[totalAcc,bodyMovementAcc,gravityAcc,Fs] = preProcessingAcc_func(rawAccData(:,1:3),Fs);
%% Add Movement [1] and Non-movement [0] labels:
labelarr = rawAccData(:,4);
labelarr(labelarr~=4 & labelarr~=5 & labelarr~=6,2) = 1;
%% Create Acceleration Feature Dataset:
tic
[featAcc,~] = createFeaturesAcc_func(bodyMovementAcc(:,1:3),labelarr(:,2),Fs);
toc
cd('..');
save('database/HAPT/featAcc_Rev2.mat','featAcc');
cd(fileparts(filePath));