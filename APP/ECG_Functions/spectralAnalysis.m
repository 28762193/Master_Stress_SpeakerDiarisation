function varargout = spectralAnalysis(hrv_stat,m_order,re_sampleRate,range_VLF,range_LF,range_HF)
%spectralAnalysis: The function takes the resampled-detrended (stationary) 
% hrv signal, model order of the parametric analysis (in this case AR 
% modelling with the Burg's method), and the re-sampling rate of the
% signal, as well as the frequency bands, and outputs the desired spectral features.
% The prefered Model Order is 16 for HRV data.


% Spectral Analysis with Autoregressive Modelling (Burg's Method):

[pxx,f] = pburg(hrv_stat,m_order,2^11,re_sampleRate);


% Absolute Power in s^2 for each frequency band: (Area under curve)
VLF = sum(pxx(f<=range_VLF))*(range_VLF/length(pxx(f<=range_VLF)));
LF = sum(pxx(f>range_VLF & f<=range_LF))*(range_LF-range_VLF)/length(pxx(f>range_VLF & f<=range_LF));
HF = sum(pxx(f>range_LF & f<=range_HF))*(range_HF-range_LF)/length(pxx(f>range_LF & f<=range_HF));   %sum(pxx(f<=range_HF)) - LF - VLF;
TP = sum(pxx)*((re_sampleRate/2)/length(pxx));

% Absolute Power in ms^2:
VLF = VLF*(1000^2);
LF = LF*(1000^2);
HF = HF*(1000^2);
TP = TP*(1000^2);
LFHF_ratio = LF/HF;

varargout{1} = pxx;
varargout{2} = f;
varargout{3} = VLF;
varargout{4} = LF;
varargout{5} = HF;
varargout{6} = TP;
varargout{7} = LFHF_ratio;
end