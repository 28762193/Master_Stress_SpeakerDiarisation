function VADidx = speakerDiar_VAD_func(audioIn,Fs,wndPar,ignoreNoiseFlag)
% This function performs voice activity detection (VAD) based on a
% model-based HMM-GMM.
% INPUT: audioIn := The input audio data to be segmented (vector).
%        Fs := The audio sampling rate (scalar-integer).
%        wndPar := windowing parameters for feature extraction (in the form
%        of a struct).
%% Initialise:
HMM_Bootstrap = load('HMM_Bootstrap.mat'); % Load Bootstrap-HMM for initial segmentation

%-------------
% Parameters:|
%-------------
% Create feature extraction windowing:
if isempty(wndPar)
    windowDuration = 0.03; % 30 ms window
    hopDuration = 0.01; % 10 ms shift (hop)
    wndPar.windowSamples = round(windowDuration*Fs);
    hopSamples = round(hopDuration*Fs);
    wndPar.overlapSamples = wndPar.windowSamples - hopSamples;
end

% Create sil_len sec segments: For Silence-Noise Confidence Measures
sil_len = find(HMM_Bootstrap.s ~= 0);
sil_len = diff(sil_len); sil_len = sil_len(2);
SN_PAR.wndSize = floor((sil_len/100)*Fs); % 0.25sec window size
SN_PAR.win = hamming(SN_PAR.wndSize,"periodic");  % hamming window
SN_PAR.overlapWnd = floor(0.5*SN_PAR.wndSize); % 50%-overlap

% Number of iterations for Step 2: (Retraining silence and noise models)
step2NumIters = 3; % Number of repeats/retraining
% Number of iterations for Step 3: (Retraining all models-Speech included)
step3NumIters = 3;
% Number of iterations for Step 4: (Lose noise model - retrain speech and
% silence models)
step4NumIters = 4;

%% Main Algorithm:
VADidx = cell(1,1);
for chunkIter=1:size(VADidx,2)
    %% Extract Features:
    features = speakerDiarFE_func(audioIn,Fs,wndPar);
    try
        if ignoreNoiseFlag
            error('Ignore Noise distinction. Use detectSpeech() for efficiency.');
        end
    %% Step 1: Segment using Bootstrap
    % Viterbi First Iter:
    [z,~] = hmmViterbi_func(HMM_Bootstrap,gather(features));
%     z(:,1) = (z(:,1)-1)*(Fs*0.01)+1;
%     z(:,2) = z(:,2)*(Fs*0.01);
    z(:,2) = (z(:,2)-1)*(wndPar.windowSamples-wndPar.overlapSamples)+wndPar.windowSamples;
    z(2:end,1) = z(1:end-1,2)+1;

    %% Step 2: Retrain Silence and Noise/Sound Models
    % Retraining silence and noise models on the current audio data
    % First Iteration: (More strict)

    % Silence and Noise Segmentation for Confidence Measures:
    SN_segments = cell(1,2);
    SN_segments{1} = z(z(:,3)==2,1:2); % 2=silence, 3=noise
    SN_segments{2} = z(z(:,3)==3,1:2); % 2=silence, 3=noise
    audio_SN = cell(1,2);
    audio_SN_seg = [];
    for ii=1:2
        % Get corresponding audio data:
        for jj=1:length(SN_segments{ii})
            audio_SN{ii} = [audio_SN{ii};audioIn(SN_segments{ii}(jj,1):SN_segments{ii}(jj,2),1)];
        end
        % Segment silence and noise audio data: (Hamming not applied here)
        [xbuf,~] = buffer(audio_SN{ii},SN_PAR.wndSize,SN_PAR.overlapWnd,"nodelay");
        if SN_PAR.overlapWnd~=0; nFrames = floor((size(audio_SN{ii},1)-SN_PAR.wndSize)/SN_PAR.overlapWnd)+1; 
        else; nFrames = floor(size(audio_SN{ii},1)/SN_PAR.wndSize); end
        
        xbuf = xbuf(:,1:nFrames);
        xbuf = SN_PAR.win.*xbuf;
        audio_SN_seg = [audio_SN_seg,xbuf];
    end

    % Extract Short-time Energy:
    SN_ener = sum(audio_SN_seg.^2,1)';

    %--------------------------------------------------------
    % Extract Silence and Noise segments with minimum doubt:|
    %--------------------------------------------------------
    % Select min Energy segments for silence: Selection based on max outlier
    SN_min = []; SN_max = [];
    SN_min.Idx = (SN_ener<=quartile_func(SN_ener,0.25)); % 25% Quantile
    SN_min.Ener = SN_ener(SN_min.Idx);
    SN_min.Q1 = quartile_func(SN_min.Ener,0.25);
    SN_min.Q3 = quartile_func(SN_min.Ener,0.75);
    SN_min.IQR = SN_min.Q3 - SN_min.Q1;
    SN_min.Thres = SN_min.Q3+(1.5*SN_min.IQR); % Outlier Detection Threshold
    SN_SilData = audio_SN_seg(:,SN_ener<SN_min.Thres);

    % Select max Energy segments for Noise: Selection based on mean and
    % silence-max-outlier(plus) then zcr is executed on the remaining data
    SN_max.Idx = (SN_ener>=quartile_func(SN_ener,0.75)); % 75% Quantile
    SN_max.Ener = SN_ener(SN_max.Idx);
    SN_max.Thres1 = quartile_func(SN_max.Ener,0.5); % Outlier Detection Threshold
    SN_max.Thres2 = SN_min.Q3+(3*SN_min.IQR); % To not include possible silence segments
    SN_NoiseData = audio_SN_seg(:,SN_ener>=SN_max.Thres1 & SN_ener>SN_max.Thres2);
    %-----------------------------
    % ZCRate for Noise detection:|
    %-----------------------------
    % Selection/Threshold based on silence-max-outlier(plus) - as to ensure not
    % to use possible silence segments for the creation of the Noise Model
    % This selection criteria makes it possible that there is no data to train
    % the noise model...
    SN_SilZCR = zerocrossrate(SN_SilData);
    SN_NoiseZCR = zerocrossrate(SN_NoiseData);

    SN_min.ZCRQ1 = quartile_func(SN_SilZCR,0.25);
    SN_min.ZCRQ3 = quartile_func(SN_SilZCR,0.75);
    SN_min.ZCRIQR = SN_min.ZCRQ3 - SN_min.ZCRQ1;
    SN_min.ZCRThres = SN_min.ZCRQ3+(1.5*SN_min.ZCRIQR);

    SN_NoiseData = SN_NoiseData(:,SN_NoiseZCR>=SN_min.ZCRThres);
    %-----------------------
    % Extract new Features:|
    %-----------------------
    SN_SilFeat = speakerDiarFE_func(SN_SilData,Fs,wndPar);
    SN_NoiseFeat = speakerDiarFE_func(SN_NoiseData,Fs,wndPar);
    if size(SN_SilFeat,1)<size(SN_SilFeat,2); SN_SilFeat=[];end
    if size(SN_NoiseFeat,1)<size(SN_NoiseFeat,2); SN_NoiseFeat=[];end
    %-----------
    % Fit GMMs:|
    %-----------
    % The amount of gaussians are based on the sound complexity (silence
    % less complex than noise) and the amount of data available for each sound
    % type.
    SN_maxSilComp = 6; SN_maxNoiseComp = 8;
    [SN_SilGMM, SN_NoiseGMM] = fitGMMs_func(0,{SN_SilFeat,SN_maxSilComp},{SN_NoiseFeat,SN_maxNoiseComp});
    %---------------
    % Recreate HMM:|
    %---------------
    if isempty(SN_NoiseGMM) % If there is no Noise Model recreate trans and init matrices
        error('No Noise. Use detectSpeech() for efficiency');
        N_Speech = 75; % Number of substates for each string
        N_Silence = 25; % The number of substates enforces a min duration for each class
        N_Noise = 0; % 75 substates = 750ms min duration (with 10ms hop).

        HMM_SN = HMMRecreate_func(N_Speech,N_Silence,N_Noise);
        HMM_SN.E.Speech = HMM_Bootstrap.E.Speech;
    else
        HMM_SN = HMM_Bootstrap;
    end
    % Create new HMM (with new Sil and Noise models, and prior Speech model):
    HMM_SN.E.Sil = SN_SilGMM;
    HMM_SN.E.Noise = SN_NoiseGMM;

    %% Step 2 - Iterative step:
    % First store previous classified speech segments:
    prevSpeech = z(z(:,3)==1,1:2); %1=Speech
    step2Iter = 1;
    Z = cell(1,step2NumIters+1);
    LLH = cell(1,step2NumIters+1);
    prevFeat = {SN_SilFeat,SN_NoiseFeat};

    % Viterbi of first Step2 Retraining on current recorded audio:
    [Z{1},LLH{1}] = hmmViterbi_func(HMM_SN,features);
    Z{1}(:,2) = (Z{1}(:,2)-1)*(wndPar.windowSamples-wndPar.overlapSamples)+wndPar.windowSamples;
    Z{1}(2:end,1) = Z{1}(1:end-1,2)+1;
    % Iteratively perform Step2 on less strict thresholds:
    while step2Iter <= step2NumIters
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Cut Prev Speech from new Silence and Sound Segments:|
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Cut previous classified speech segments from newly classified silence and
        % sound segments (if there is an overlap):
        SN_segments = Z{step2Iter}(Z{step2Iter}(:,3)~=1,:);
        %remove_range = zeros(length(prevSpeech),3);
        cutIter1 = 1;
        cutIter2 = 1;
        cutIter3 = 1;
        while cutIter1<length(prevSpeech) && prevSpeech(cutIter1,1) < SN_segments(end,2)
            while cutIter2<length(SN_segments) && prevSpeech(cutIter1,1) >= SN_segments(cutIter2,1)
                if prevSpeech(cutIter1,1)<=SN_segments(cutIter2,2)
                    % Overlap - Find Range:
                    if prevSpeech(cutIter1,2)<=SN_segments(cutIter2,2)
                        %remove_range(cutIter3,1:3) = [prevSpeech(cutIter1,1) prevSpeech(cutIter1,2) cutIter2];
                        SN_segments = [SN_segments(1:cutIter2-1,:); [0 0 SN_segments(cutIter2,3)]; SN_segments(cutIter2:end,:)];
                        SN_segments(cutIter2,1) = SN_segments(cutIter2+1,1);
                        SN_segments(cutIter2,2) = prevSpeech(cutIter1,1)-1;
                        SN_segments(cutIter2+1,1) = prevSpeech(cutIter1,2)+1;
                    else
                        %remove_range(cutIter3,1:3) = [prevSpeech(cutIter1,1) SN_segments(cutIter2,2) cutIter2];
                        SN_segments(cutIter2,2) = prevSpeech(cutIter1,1)-1;
                    end
                    cutIter3 = cutIter3+1;
                end
                cutIter2 = cutIter2+1;
            end
            cutIter1 = cutIter1+1;
        end
        %remove_range(cutIter3:end,:) = [];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Silence and Noise Model Training:|
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        SN_ener2 = cell(1,2);
        audio_SN_seg = SN_ener2;

        for ii=1:2
            audio_SN = [];
            temp = SN_segments(SN_segments(:,3)==ii+1,1:2); % 2=silence, 3=noise
            for jj=1:length(temp)
                audio_SN = [audio_SN;audioIn(temp(jj,1):temp(jj,2),1)];
            end

            % Segment:
            if ~isempty(audio_SN)
                [audio_SN_seg{ii},~] = buffer(audio_SN,SN_PAR.wndSize,SN_PAR.overlapWnd,"nodelay");
                if SN_PAR.overlapWnd~=0; nFrames = floor((size(audio_SN,1)-SN_PAR.wndSize)/SN_PAR.overlapWnd)+1; 
                else nFrames = floor(size(audio_SN,1)/SN_PAR.wndSize); end
                audio_SN_seg{ii} = audio_SN_seg{ii}(:,1:nFrames);
                audio_SN_seg{ii} = SN_PAR.win.*audio_SN_seg{ii};
    
                % Extract Short-time Energy Per classified silence and noise regions:
                SN_ener2{1,ii} = sum(audio_SN_seg{ii}.^2,1)';
            end
        end

        %-----------------------------------------------------
        % Find Silence and Sound Segments with minimum doubt:|
        %-----------------------------------------------------
        SN_minQ1 = quartile_func(SN_ener2{1},0.25);
        SN_minQ3 = quartile_func(SN_ener2{1},0.75);
        SN_minIQR = SN_minQ3-SN_minQ1;
        SN_minThres = SN_minQ3+1.5*SN_minIQR;
        SN_SilData = audio_SN_seg{1}(:,SN_ener2{1}<=SN_minThres);

        % Add outliers to noise segments (Adds robustness against possible misclassifications):
        if ~isempty(SN_ener2{2})
            SN_maxThres1 = SN_minQ3+3*SN_minIQR; % Silence outlier threshold for Noise.
            audio_SN_seg{2} = [audio_SN_seg{2} audio_SN_seg{1}(:,SN_ener2{1}>SN_maxThres1)];
            SN_ener2{2} = [SN_ener2{2}; SN_ener2{1}(SN_ener2{1}>SN_maxThres1,:)];
            % Select max Energy segments for Noise:
            if step2Iter<4;p = 0.6-((step2Iter-1)*0.1);else; p=0.4;end
            SN_maxThres2 = quartile_func(SN_ener2{2},p); % Outlier Detection Threshold
            SN_NoiseData = audio_SN_seg{2}(:,SN_ener2{2}>=SN_maxThres1 & SN_ener2{2}>=SN_maxThres2);

            %-----------------------------
            % ZCRate for Noise detection:|
            %-----------------------------
    
            SN_SilZCR = zerocrossrate(SN_SilData);
            SN_NoiseZCR = zerocrossrate(SN_NoiseData);
    
            SN_minZCRQ1 = quartile_func(SN_SilZCR,0.25);
            SN_minZCRQ3 = quartile_func(SN_SilZCR,0.75);
            SN_minZCRIQR = SN_minZCRQ3 - SN_minZCRQ1;
            SN_ZCRThres = SN_minZCRQ3+(1.5*SN_minZCRIQR);
    
            SN_NoiseData = SN_NoiseData(:,SN_NoiseZCR>=SN_ZCRThres);
        else;SN_NoiseData = [];
        end

        %-----------------------
        % Extract new Features:|
        %-----------------------
        SN_SilFeat = speakerDiarFE_func(SN_SilData,Fs,wndPar);
        SN_NoiseFeat = speakerDiarFE_func(SN_NoiseData,Fs,wndPar);
        if size(SN_SilFeat,1)<size(SN_SilFeat,2); SN_SilFeat=[];end
        if size(SN_NoiseFeat,1)<size(SN_NoiseFeat,2); SN_NoiseFeat=[];end
        %-----------
        % Fit GMMs:|
        %-----------
        SN_maxSilComp = 6; SN_maxNoiseComp = 8;
        [SN_SilGMM, SN_NoiseGMM] = fitGMMs_func(0,{SN_SilFeat,SN_maxSilComp},{SN_NoiseFeat,SN_maxNoiseComp});
        %----------------------------------------------------------------------
        % Create new HMM (with new Sil and Noise models, and prior Speech
        % model):|
        %----------------------------------------------------------------------
        if ~isequal(SN_SilFeat,prevFeat{1}) || ~isequal(SN_NoiseFeat,prevFeat{2})
            %---------------
            % Recreate HMM:|
            %---------------
            if ~isequal(SN_NoiseFeat,prevFeat{2})
                if isempty(SN_NoiseGMM) % If there is no Noise Model recreate trans and init matrices
                    error('No Noise. Use detectSpeech() for efficiency');
                    N_Speech = 75; % Number of substates for each string
                    N_Silence = 25; % The number of substates enforces a min duration for each class
                    N_Noise = 0; % 75 substates = 750ms min duration (with 10ms hop).
            
                    HMM_SN = HMMRecreate_func(N_Speech,N_Silence,N_Noise);
                    HMM_SN.E.Speech = HMM_Bootstrap.E.Speech;
                else
                    HMM_SN = HMM_Bootstrap;
                end
            end
            % Create new HMM (with new Sil and Noise models, and prior Speech model):
            HMM_SN.E.Sil = SN_SilGMM;
            HMM_SN.E.Noise = SN_NoiseGMM;

            prevFeat = {SN_SilFeat,SN_NoiseFeat};
            % Viterbi:
            [Z{step2Iter+1},LLH{step2Iter+1}] = hmmViterbi_func(HMM_SN,features);
            Z{step2Iter+1}(:,2) = (Z{step2Iter+1}(:,2)-1)*(wndPar.windowSamples-wndPar.overlapSamples)+wndPar.windowSamples;
            Z{step2Iter+1}(2:end,1) = Z{step2Iter+1}(1:end-1,2)+1;
%             Z{step2Iter+1}(:,1) = (Z{step2Iter+1}(:,1)-1)*(Fs*0.01)+1;
%             Z{step2Iter+1}(:,2) = Z{step2Iter+1}(:,2)*(Fs*0.01);
        else
            step2Iter = step2NumIters;
            Z(cellfun(@isempty,Z)) = [];
            LLH(cellfun(@isempty,LLH)) = [];
        end
        step2Iter = step2Iter+1;
    end

    %% Step 3: Training Speech Model
    % Train the speech model on the current audio segmented by the new
    % silence-noise models and the bootstrap speech model.
    speechSegments = Z{end}(Z{end}(:,3)==1,:);
    temp = find(HMM_SN.s~=0);
    temp = diff(temp);
    % Remove speech segments that are less than minSpeech length as defined by
    % the HMM-string:
    speechSegments(speechSegments(:,2)-speechSegments(:,1)<(temp(1)/100)*Fs-1,:) = [];
    audioSpeech = [];
    for jj=1:length(speechSegments)
        audioSpeech = [audioSpeech;audioIn(speechSegments(jj,1):speechSegments(jj,2),1)];
    end
    
    % Apply Hamming window:
    SSN_wndSize = floor((temp(1)*0.01)*Fs); % x sec window size
    SSN_win = hamming(SSN_wndSize,"periodic");  % hamming window
    SSN_overlapWnd = floor(0.5*SSN_wndSize); % 50-overlap
    [xbuf,~] = buffer(audioSpeech,SSN_wndSize,SSN_overlapWnd,'nodelay');
    xbuf = xbuf.*SSN_win;

    % Get speech Features:
    speechFeat = speakerDiarFE_func(xbuf,Fs,wndPar);
    if size(speechFeat,1)<size(speechFeat,2); speechFeat=[];end

    % Fit Speech GMM:
    speech_MaxComp = 8;
    speechGMM = fitGMMs_func(0,{speechFeat,speech_MaxComp});

    % Re-establish HMM:
    HMM_SN.E.Speech = speechGMM;
    %% Step 3 - Iteration: Retrain All Models
    step3Iter = 1;
    Z = cell(1,step3NumIters); LLH = Z;
    HMM_SSN = HMM_SN;
    % Perform Viterbi:
    [Z{step3Iter},LLH{step3Iter}] = hmmViterbi_func(HMM_SSN,features);
    Z{step3Iter}(:,2) = (Z{step3Iter}(:,2)-1)*(wndPar.windowSamples-wndPar.overlapSamples)+wndPar.windowSamples;
    Z{step3Iter}(2:end,1) = Z{step3Iter}(1:end-1,2)+1;
    step3Iter = step3Iter+1;

    %% Step 3: Iterate
    %--------------------------------------------------------------------------
    % Why I believe it is needed to iterate step3 - Some segments can be
    % misclassified (e.g., sound for speech) initially by the bootstrap
    % component, by allowing the sound and speech model, for example, to start
    % and increase with the same number of gaussian components this allows the
    % models to fix these misclassifications.
    %--------------------------------------------------------------------------
    
    SSN_MaxComps = zeros(3,(step3NumIters-step3Iter)+1);
    SSN_MaxComps(1,:) = 4:2:4+(2*(step3NumIters-step3Iter));
    SSN_MaxComps(2,:) = 2:2+(step3NumIters-step3Iter);
    SSN_MaxComps(3,:) = 4:2:4+(2*(step3NumIters-step3Iter)); % Increasing the number of gaussian components per iteratio
    SSN_MaxComps(SSN_MaxComps>10) = 10; % Limit components to 10.
    SSN_GMM = cell(1,3);
    prevFeat = {speechFeat,SN_SilFeat};
    minSegTim = [];

    while step3Iter <= step3NumIters
        SSN_numClass = length(unique(Z{1}(:,3)));
        SSN_segments = cell(1,SSN_numClass);
        SSN_Feat = cell(1,SSN_numClass);
        SSN_audio = cell(1,SSN_numClass);
        
        minSegTim = find(HMM_SSN.s~=0);
        minSegTim(end+1) = length(HMM_SSN.s)+1;
        minSegTim = diff(minSegTim);

        % Get segments:
        for ii=1:SSN_numClass
            SSN_segments{ii} = Z{step3Iter-1}(Z{step3Iter-1}(:,3)==ii,:);
            % Remove false segments:
            SSN_segments{ii}(SSN_segments{ii}(:,2)-SSN_segments{ii}(:,1)<(minSegTim(ii)/100)*Fs-1,:) = [];
        end

        for ii=1:SSN_numClass
            % Get corresponding audio:
            for jj=1:length(SSN_segments{ii})
                SSN_audio{ii} = [SSN_audio{ii};audioIn(SSN_segments{ii}(jj,1):SSN_segments{ii}(jj,2),1)];
            end
            
            % Apply Hamming window:
            SSN_wndSize = floor((minSegTim(ii)/100)*Fs); % x sec window size
            SSN_win = hamming(SSN_wndSize,"periodic");  % hamming window
            SSN_overlapWnd = floor(0.5*SSN_wndSize); % 0-overlap
            [xbuf,~] = buffer(SSN_audio{ii},SSN_wndSize,SSN_overlapWnd,'nodelay');
            xbuf = xbuf.*SSN_win;
            

            % Get Features:
            SSN_Feat{ii} = speakerDiarFE_func(xbuf,Fs,wndPar);
            if size(SSN_Feat{ii},1)<size(SSN_Feat{ii},2); SSN_Feat{ii}=[];end
            % Fit GMM:
%             [SSN_GMM{ii},~] = speakerDiar_OptiGauss_func(SSN_Feat{ii},2,50);
            SSN_GMM{ii} = fitGMMs_func(1,{SSN_Feat{ii},SSN_MaxComps(ii,step3Iter-1)});
        end

        % Check whether there were any changes made:
        if isequal(SSN_Feat(1),prevFeat(1)) || isequal(SSN_Feat(2),prevFeat(2))
            % No changes in the features: Stop iteration
            step3Iter = step3NumIters;
            Z(cellfun(@isempty,Z)) = [];
            LLH(cellfun(@isempty,LLH)) = [];
        else 
            prevFeat = SSN_Feat;
            %---------------
            % Recreate HMM:|
            %---------------
            if isempty(SSN_GMM{3}) % If there is no Noise Model recreate trans and init matrices
                error('No Noise. Use detectSpeech() for efficiency');
                N_Speech = 75; % Number of substates for each string
                N_Silence = 25; % The number of substates enforces a min duration for each class
                N_Noise = 0; % 75 substates = 750ms min duration (with 10ms hop).
                if length(HMM_SSN.s) ~= (N_Speech+N_Silence+N_Noise)
                    HMM_SSN = HMMRecreate_func(N_Speech,N_Silence,N_Noise); % Outputs the A and s matrices
                end
            end
            HMM_SSN.E.Speech = SSN_GMM{1};
            HMM_SSN.E.Sil = SSN_GMM{2};
            HMM_SSN.E.Noise = SSN_GMM{3};
    
            % Perform Viterbi:
            [Z{step3Iter},LLH{step3Iter}] = hmmViterbi_func(HMM_SSN,features);
            Z{step3Iter}(:,2) = (Z{step3Iter}(:,2)-1)*(wndPar.windowSamples-wndPar.overlapSamples)+wndPar.windowSamples;
            Z{step3Iter}(2:end,1) = Z{step3Iter}(1:end-1,2)+1;
%             Z{step3Iter}(:,1) = (Z{step3Iter}(:,1)-1)*(Fs*0.01)+1;
%             Z{step3Iter}(:,2) = Z{step3Iter}(:,2)*(Fs*0.01);
        end
        step3Iter = step3Iter+1;
    end

    %% Step4: Determine if Speech = Noise Model
    % Using BIC
    %--------------------------------------------------------------------------
    % Train a new model on speech and noise data combined, with the combined
    % amount of gaussians as the speech and noise model. Check whether
    % delta-BIC is positive, if so then the models are trained on the same data
    % and thus the noise model needs to be removed and the speech and silence
    % models needs to be retrained.
    %--------------------------------------------------------------------------
    %Formula: delta-BIC = BIC(Combined Segments) - (BIC(Speech Seg) + BIC(Noise Seg))

    % Parameters:
    % Number of components for the combined model must be equal to the sum of
    % its composite parts.
    if ~isempty(HMM_SSN.E.Noise) % The speech and noise check is only relevant if there is a noise model.
        combinedMaxComp = (HMM_SSN.E.Speech.NumComponents + HMM_SSN.E.Noise.NumComponents);
        
        combined_audio = [];
        % Create combined audio data segments:
        % Get segments:
        combined_segments = [];
        combined_segments{1} = Z{end}(Z{end}(:,3)==1,:);
        combined_segments{2} = Z{end}(Z{end}(:,3)==3,:);
        % Remove false segments:
        combined_segments{1}(combined_segments{1}(:,2)-combined_segments{1}(:,1)<(minSegTim(1)/100)*Fs-1,:) = [];
        combined_segments{2}(combined_segments{2}(:,2)-combined_segments{2}(:,1)<(minSegTim(end)/100)*Fs-1,:) = [];
    
        combined_segments = [combined_segments{1}; combined_segments{2}];
        % Get corresponding audio:
        for jj=1:length(combined_segments)
            combined_audio = [combined_audio;audioIn(combined_segments(jj,1):combined_segments(jj,2),1)];
        end
        % Apply Hamming window:
        combined_wndSize = floor((minSegTim(1)/100)*Fs); % x sec window size based on speech min duration
        combined_win = hamming(combined_wndSize,"periodic");  % hamming window
        combined_overlapWnd = floor(0.5*combined_wndSize); % 50-overlap
        [xbuf,~] = buffer(combined_audio,combined_wndSize,combined_overlapWnd,'nodelay');
        xbuf = xbuf.*combined_win;
    
        % Get Features:
        %combinedFeat = speakerDiarFE_func(xbuf,Fs,wndPar);
        combinedFeat = [SSN_Feat{1};SSN_Feat{3}];
        if size(combinedFeat,1)<size(combinedFeat,2);combinedFeat=[];end
        % Fit GMM:
        combinedGMM = fitGMMs_func(1,{combinedFeat combinedMaxComp});
        % Get delta-BIC:
        delta_BIC = combinedGMM.BIC - (HMM_SSN.E.Speech.BIC + HMM_SSN.E.Noise.BIC);
    
        %%
        %if delta-bic is > 0 then delete noise model:
        if delta_BIC > 0
            %delete noise and retrain:
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            step4Iter = 2;
            tempZ = Z{end}; tempLLH = LLH{end};
            Z = cell(1,step4NumIters); Z{1} = tempZ;
            LLH = cell(1,step4NumIters); LLH{1} = tempLLH;
            SSN_numClass = length(unique(Z{1}(:,3)));
            %SSN_MaxComps = cell(1,SSN_numClass); % Speech, Silence, Noise
            SSN_MaxComps = zeros(3,(step4NumIters-step4Iter)+1);
            SSN_MaxComps(1,:) = 6:2:6+(2*(step4NumIters-step4Iter));
            SSN_MaxComps(2,:) = 2:2+(step4NumIters-step4Iter);
            SSN_MaxComps(3,:) = 6:2:6+(2*(step4NumIters-step4Iter)); % Increasing the number of gaussian components per iteration
            SSN_MaxComps(SSN_MaxComps>10) = 10;

            SSN_segments = cell(1,SSN_numClass);
            SSN_Feat = cell(1,SSN_numClass);
            SSN_GMM = cell(1,3); % Hard assigned for Speech,Silence,Noise
    
            minSegTim = find(HMM_SSN.s~=0);
            minSegTim(end+1) = length(HMM_SSN.s)+1;
            minSegTim = diff(minSegTim);
    
            while step4Iter <= step4NumIters
    
                SSN_audio = cell(1,SSN_numClass);
    
                % Get segments:
                %----------------------%----------------------%----------------------
                for ii=1:SSN_numClass
                    SSN_segments{ii} = Z{step4Iter-1}(Z{step4Iter-1}(:,3)==ii,:);
                    if ~isempty(SSN_segments{ii})
                        % Remove false segments:
                        SSN_segments{ii}(SSN_segments{ii}(:,2)-SSN_segments{ii}(:,1)<(minSegTim(ii)/100)*Fs-1,:) = [];
                    end
                end
                % Merge Speech and 'Noise' Segments:
                SSN_segments{1} = [SSN_segments{1};SSN_segments{3}];
                SSN_segments{3} = [];
                %----------------------%----------------------%----------------------
                for ii=1:SSN_numClass
                    % Get corresponding audio:
                    for jj=1:length(SSN_segments{ii})
                        SSN_audio{ii} = [SSN_audio{ii};audioIn(SSN_segments{ii}(jj,1):SSN_segments{ii}(jj,2),1)];
                    end
                    % Get Features:
                    SSN_Feat{ii} = speakerDiarFE_func(SSN_audio{ii},Fs,wndPar);
                    if size(SSN_Feat{ii},1)<size(SSN_Feat{ii},2); SSN_Feat{ii}=[];end
                    % Fit GMM:
                    SSN_GMM{ii} = fitGMMs_func(1,{SSN_Feat{ii},SSN_MaxComps(ii,step4Iter-1)});
                end
    
                %---------------
                % Recreate HMM:|
                %---------------
                if isempty(SSN_GMM{3}) % If there is no Noise Model recreate trans and init matrices
                    N_Speech = 75; % Number of substates for each string
                    N_Silence = 25; % The number of substates enforces a min duration for each class
                    N_Noise = 0; % 75 substates = 750ms min duration (with 10ms hop).
                    if length(HMM_SSN.s) ~= (N_Speech+N_Silence+N_Noise)
                        HMM_SSN = HMMRecreate_func(N_Speech,N_Silence,N_Noise); % Outputs the A and s matrices
                    end
                end
                HMM_SSN.E.Speech = SSN_GMM{1};
                HMM_SSN.E.Sil = SSN_GMM{2};
                HMM_SSN.E.Noise = [];
    
                % Perform Viterbi:
                [Z{step4Iter},LLH{step4Iter}] = hmmViterbi_func(HMM_SSN,features);
                Z{step4Iter}(:,2) = (Z{step4Iter}(:,2)-1)*(wndPar.windowSamples-wndPar.overlapSamples)+wndPar.windowSamples;
                Z{step4Iter}(2:end,1) = Z{step4Iter}(1:end-1,2)+1;
%                 Z{step4Iter}(:,1) = (Z{step4Iter}(:,1)-1)*(Fs*0.01)+1;
%                 Z{step4Iter}(:,2) = Z{step4Iter}(:,2)*(Fs*0.01);
    
                step4Iter = step4Iter+1;
                VADidx{chunkIter} = Z;
                %EOL
            end
    
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        else
            VADidx{chunkIter} = Z;
        end
    else
        VADidx{chunkIter} = Z;
    end
    catch
        % No Noise Present thus Viterbi algorithm no longer needed - Switch
        % to detectSpeech()
        % detectSpeech() is much more efficient as it is algorithmic based
        % running only once as oppose to Viterbi which has to re-iterate.
        mergeDistance = 0.25;
        win = hamming(wndPar.windowSamples,'periodic');
        VADidx{chunkIter} = {detectSpeech(audioIn,Fs,"Window",win,'MergeDistance',mergeDistance*Fs,"OverlapLength",wndPar.overlapSamples)};
        VADidx{chunkIter}{1}(:,end+1) = 1;
        disp('No Noise. Using detectSpeech() for improved efficiency.');
    end
%EOL - chunkIter
end
% Merge Chunks:
temp = [];
for mergeCounter=1:size(VADidx,2)
    temp = [temp;VADidx{mergeCounter}{end}];
end
VADidx = temp;
%EOF
end
%% Helper functions:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------------------hmmViterbi_func------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [segmentSSN, llh] = hmmViterbi_func(model, x)
% Viterbi algorithm (calculated in log scale to improve numerical stability).
% Input:
%   x: o x f feature matrix (obs x features)
%   model: model structure which contains
%       model.s: n x 1 start probability vector
%       model.A: n x n transition matrix (row->column)
%       model.E: M GMMs (M=3)
% Output:
%   z: 1 x o latent state
%   llh:  loglikelihood
% o -> #obs; f-> #features; n -> #states; M -> #classes (Speech, Silence, Noise)
    o = size(x,1); % #observations
    n = size(model.A,1); % #states

    % Sparse matrix model.s for efficiency
    s = log(model.s); % Working in log space to avoid numerical issues
    A = log(model.A);
    B = zeros(n,o);
    % Get number of GMMs:
    k = 0;
    modelNames = fieldnames(model.E);
    for i=1:length(modelNames)
        if isa(model.E.(modelNames{i}),'gmdistribution'); k=k+1;end
    end
    %Get logpdfs from GMMs
    logpdf = cell(1,k);
    % [idx,nlogpdf,P,logpdf,d2] = cluster():
    for i=1:k
        [~,~,~,logpdf{i}] = cluster(model.E.(modelNames{i}),x);
    end
    % Fill Emission Probability for all states (shared-states):
    temp = find(model.s ~= 0);
    for i=1:length(temp)-1
        B(temp(i):temp(i+1)-1,:) = repmat(logpdf{i}',temp(i+1)-temp(i),1);
    end
    B(temp(end):end,:) = repmat(logpdf{end}',size(B,1)-temp(end)+1,1);
    %---------------------------------------------------------------
    %Enforce change at the end: As to ensure the minimum duration is
    %enforced at the end of the audio track.
%     tempMaxregions = B(:,end-1)==max(B(:,end-1));
%     B(tempMaxregions,end) = -inf;
    %----------------------------------------------------------------
    % Viterbi:|
    % ---------
    % Initialisation:
    Delt = zeros(n,o);
    Delt(:,1) = 1:n;
    v = s(:) + B(:,1);
    % Inductive Step:
    for t=2:o
        [v,idx] = max(bsxfun(@plus,A,v),[],1);
        v = v(:)+B(:,t);
        Delt = Delt(idx,:);
        Delt(:,t) = 1:n;
    end
    % Termination:
    [llh,idx] = max(v);
    z = Delt(idx,:);

    % Further Segmentation (Segment after Viterbi):|
    % ----------------------------------------------
    % Assign labels to states:
    temp = find(model.s ~= 0);

    for i = 1:length(temp)-1
        z(z>=temp(i) & z<temp(i+1)) = i;
    end
    z(z>=temp(end)) = length(temp);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    z(1,end) = z(1,end-1);
    % Enforce minimum duration:
    chgPoint = find(ischange(z));
    if ~isempty(chgPoint)
        tempDiff = diff(temp);
        tempDiff(end+1) = size(s,1)-temp(end)+1;
        tempChgSize = zeros(numel(chgPoint)+1,1);
        tempChgSize(1) = chgPoint(1);
        for i=2:numel(chgPoint)
            tempChgSize(i) = chgPoint(i)-chgPoint(i-1);
        end
        tempChgSize(end) = numel(z)-chgPoint(end)+1;
        for i=1:numel(tempChgSize)-1
            if tempChgSize(i)<tempDiff(z(chgPoint(i)-1))-2
                % Too short...
                z(chgPoint(i):chgPoint(i+1)) = z(chgPoint(i)-1);
            end
        end
        if tempChgSize(end)<tempDiff(z(end))-2
            % Too short...
            z(chgPoint(end):end) = z(chgPoint(end)-1);
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    chgPoint = find(ischange(z));
    segmentSSN = zeros(length(chgPoint)+1,3); % Segment Array: Begin|End|Label
    segmentSSN(1,1) = 1;
    segmentSSN(2:end,1) = chgPoint;
    segmentSSN(1:end-1,2) = chgPoint-1; segmentSSN(end,2) = o;
    segmentSSN(:,3) = z(segmentSSN(:,2));
%EOF
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------fitGMMs_func-------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function varargout = fitGMMs_func(forceFit,varargin)
% varargin -> {{SilFeat};{maxSilComp}}, {{NoiseFeat};{maxNoiseComp}}, {{SpeechFeat};{maxSpeechComp}}
% varargout (depends on varargin) -> SilGMM, NoiseGMM, SpeechGMM

options = statset('MaxIter',300);
varargout = cell(1,nargin-1);

for i=1:nargin-1
    Feat = varargin{i}{1};
    if ~isempty(Feat)
        maxComp = varargin{i}{2};
        if ~forceFit
            Comp = round(size(Feat,1)/(size(Feat,2)*4));
            if Comp<2;Comp=2;elseif Comp>maxComp; Comp=maxComp;end
        else 
            Comp = maxComp;
        end
        varargout{i} = fitgmdist(Feat,Comp,'CovarianceType','diagonal','Options',options);
    else
        varargout{i} = [];
    end
end

%EOF
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------HMMRecreate_func-------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function HMM = HMMRecreate_func(N_Speech,N_Silence,N_Noise)

N_Values = [N_Speech N_Silence N_Noise];
N_Values = N_Values(N_Values>0);
N_States = sum(N_Values);
string_begin = zeros(1,length(N_Values));
string_end = zeros(1,length(N_Values));
string_begin(1) = 1; string_end(1) = N_Values(1);

for i=2:length(N_Values)
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

end