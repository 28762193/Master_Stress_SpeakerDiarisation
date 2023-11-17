% Create Audio Tester Files:

fileLoc = cell(5,1);
fileName = cell(5,1);
fileLoc{1} = '/LibriSpeech/dev-clean/84/121123/';
fileLoc{2} = '/LibriSpeech/dev-clean/174/50561/';
fileLoc{3} = '/LibriSpeech/dev-clean/251/118436/';
fileLoc{4} = '/LibriSpeech/dev-clean/652/129742/';
fileLoc{5} = '/LibriSpeech/dev-other/700/122866/';

for ii=1:size(fileLoc,1)
    cd(fileLoc{ii});
    fileList = dir('*.flac');
    fileList = {fileList.name}';
    fileName{ii} = strcat(fileLoc{ii},fileList);
end

%% Move to current running script:
filePath = matlab.desktop.editor.getActiveFilename;
cd(fileparts(filePath));
%% Create Audio Track:
rng(3); % Control random number generator
spkIdx = randi([1 size(fileName,1)],1,85); % Speaker Index - Whos turn is it to speak.

audioTrack = [];
trackInformation = cell(size(spkIdx));
Fs = 16e3;
trackIdx = zeros(1,length(spkIdx));
for ii=1:length(spkIdx)
    trackIdx(ii) = randi([1 length(fileName{spkIdx(ii)})],1); % Which track to use.
    [tempAu,~] = audioread(fileName{spkIdx(ii)}{trackIdx(ii)});
    trackInformation{ii} = audioinfo(fileName{spkIdx(ii)}{trackIdx(ii)});
    audioTrack = [audioTrack;tempAu];
end
%% Save Audio Track for Testing:
filename = 'tempAudioMixed_LibriSpeech_SuperLong.wav';
audiowrite(filename,audioTrack,Fs);
filenameInfo = 'tempAudioMixed_LibriSpeech_SuperLong_Info.mat';
save(filenameInfo,"trackInformation");
filenameSpk = 'tempAudioMixed_LibriSpeech_SuperLong_Spk.mat';
save(filenameSpk,"spkIdx");
%%


% fileLoc = {};
% filenum_read = {};
% % Female Voice - ID:84
% fileLoc{1} = '/LibriSpeech/dev-clean/84/121123/';
% filenum_read{1} = '84-121123-0001.flac';
% % Male Voice - ID:174
% fileLoc{2} = '/LibriSpeech/dev-clean/174/50561/';
% filenum_read{2} = '174-50561-0000.flac';
% fileLoc{3} = fileLoc{2};
% filenum_read{3} = '174-50561-0003.flac';
% % Male Voice - ID:251
% fileLoc{4} = '/LibriSpeech/dev-clean/251/118436/';
% filenum_read{4} = '251-118436-0001.flac';
% % Male Voice - ID:652
% fileLoc{5} = '/LibriSpeech/dev-clean/652/129742/';
% filenum_read{5} = '652-129742-0000.flac';
% % Female Voice - ID:700
% fileLoc{6} = '/LibriSpeech/dev-other/700/122866/';
% filenum_read{6} = '700-122866-0001.flac';
% % % Male Voice - ID:251
% % fileLoc{7} = fileLoc{4};
% % filenum_read{7} = '251-118436-0000.flac';
% % % Female Voice - ID:700
% % fileLoc{8} = fileLoc{6};
% % filenum_read{8} = '700-122866-0003.flac';
% % % Female Voice - ID:84
% % fileLoc{9} = fileLoc{1};
% % filenum_read{9} = '84-121123-0003.flac';
% 
% 
% audioData = [];
% Fs = 16e3;
% for i=1:length(fileLoc)
%     filename_read = strcat(fileLoc{i},filenum_read{i});
%     [audio1,~] = audioread(filename_read);
%     audioData = [audioData;audio1];
% end
% 
% filename = 'tempAudioMixed_LibriSpeech_Short.wav';
% audiowrite(filename,audioData,Fs);