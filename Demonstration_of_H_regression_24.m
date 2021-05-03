% Deep Residual Learning Meets OFDM Channel Estimation
% SNR represents Es/N0, Es/N0 = Eb/N0 * log2(M)
% For SNR, the power of pilot is ignored when calculating the varience of
% noise and the pilots suffer from the noise on equal level with data
% Power is balanced since IFFT/FFT is applied
% InsertDCNull is true and the effect caused by DCNull is considered in the
% SNR_OFDM, which is adjusted to be SNR + 10 * log(Num_of_subcarriers_used
% / Num_of_FFT)
% h_channel is different for each frame and stored in H, or read from h_set

SNR_Range = 0:5:20;
Num_of_frame_each_SNR = 1000;

T_LS = zeros(length(SNR_Range), 1);
T_MMSE = zeros(length(SNR_Range), 1);
T_DNN = zeros(length(SNR_Range), 1);

MSE_LS_over_SNR = zeros(length(SNR_Range), 1);
MSE_MMSE_over_SNR = zeros(length(SNR_Range), 1);
MSE_DNN_over_SNR = zeros(length(SNR_Range), 1);

% Import Deep Neuron Network
load('ReEsNet24Pruning.mat');

Rgg_value = zeros(73, 73, 4);

for SNR = SNR_Range
    
    for i = 1 : 4
        Rgg_value(:, :, i) = Rgg_On_channel_estimation_in_OFDM_systems(10000, i);
    end

M = 4; % QPSK
k = log2(M);

Num_of_subcarriers = 72;
Num_of_FFT = Num_of_subcarriers + 1;
length_of_CP = 16;

Num_of_symbols = 10;
Num_of_pilot = 4;
Frame_size = Num_of_symbols + Num_of_pilot;

Pilot_interval = 4;
Pilot_starting_location = 1;

Pilot_location = [(1:12:Num_of_subcarriers)', (4:12:Num_of_subcarriers)', (7:12:Num_of_subcarriers)', (10:12:Num_of_subcarriers)'];
Pilot_value_user = 1 + 1j;

length_of_symbol = Num_of_FFT + length_of_CP;

Num_of_QPSK_symbols = Num_of_subcarriers * Num_of_symbols * Num_of_frame_each_SNR;
Num_of_bits = Num_of_QPSK_symbols * k;

LS_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
MMSE_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
DNN_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);

t_LS = zeros(Num_of_frame_each_SNR, 1);
t_MMSE = zeros(Num_of_frame_each_SNR, 1);
t_DNN = zeros(Num_of_frame_each_SNR, 1);

for Frame = 1:Num_of_frame_each_SNR

% Data generation
N = Num_of_subcarriers * Num_of_symbols;
data = randi([0 1], N, k);
Data = reshape(data, [], 1);
dataSym = bi2de(data);

% QPSK modulator
QPSK_symbol = QPSK_Modualtor(dataSym);
QPSK_signal = reshape(QPSK_symbol, Num_of_subcarriers, Num_of_symbols);

% Pilot inserted
[data_in_IFFT, data_location] = Pilot_Insert(Pilot_value_user, Pilot_starting_location, Pilot_interval, Pilot_location, Frame_size, Num_of_FFT, QPSK_signal);
[data_for_channel, ~] = Pilot_Insert(sqrt(2), Pilot_starting_location, Pilot_interval, kron((1 : Num_of_subcarriers)', ones(1, Num_of_pilot)), Frame_size, Num_of_FFT, (ones(Num_of_subcarriers, Num_of_symbols) * sqrt(2)));
data_for_channel(1, :) = sqrt(2);

%Noise_Variance = 2 / (10 ^ (SNR / 10));

%Nvariance = sqrt(Noise_Variance / 2);
%n = Nvariance * (randn(size(data_in_IFFT_REF, 1), size(data_in_IFFT_REF, 2)) + 1j * randn(size(data_in_IFFT_REF, 1), size(data_in_IFFT_REF, 2))); % Noise generation
%Noise_variance = mean(mean(abs(n).^2));
%data_in_IFFT = data_in_IFFT_REF + n;

% OFDM Transmitter
[Transmitted_signal, MMSE_signal] = OFDM_Transmitter(data_in_IFFT, Num_of_FFT, length_of_CP);
[Transmitted_signal_for_channel, ~] = OFDM_Transmitter(data_for_channel, Num_of_FFT, length_of_CP);

% Channel

% AWGN Channel
SNR_OFDM = SNR + 10 * log10((Num_of_subcarriers / Num_of_FFT));
%awgnChan = comm.AWGNChannel('NoiseMethod', 'Signal to noise ratio (SNR)');
%awgnChan.SNR = SNR_OFDM;

% Multipath Rayleigh Fading Channel
Multitap_h = [(randn + 1j * randn);...
    (randn + 1j * randn) / 2;...
    (randn + 1j * randn) / 4;...
    (randn + 1j * randn) / 8;...
    (randn + 1j * randn) / 16];

% linear convolution
Multitap_Channel_Signal_user = conv(Transmitted_signal, Multitap_h);
Multitap_Channel_Signal_user = Multitap_Channel_Signal_user(1 : length(Transmitted_signal));

SignalPower = mean(abs(Multitap_Channel_Signal_user) .^ 2);
Noise_Variance = SignalPower / (10 ^ (SNR_OFDM / 10));

Nvariance = sqrt(Noise_Variance / 2);
n = Nvariance * (randn(length(Transmitted_signal), 1) + 1j * randn(length(Transmitted_signal), 1)); % Noise generation

Multitap_Channel_Signal = Multitap_Channel_Signal_user + n;
Noise_power_time = mean(abs(Multitap_Channel_Signal) .^ 2 - abs(Multitap_Channel_Signal_user) .^ 2);

Multitap_Channel_Signal_user_for_channel = conv(Transmitted_signal_for_channel, Multitap_h);
Multitap_Channel_Signal_user_for_channel = Multitap_Channel_Signal_user_for_channel(1 : length(Transmitted_signal_for_channel));

SignalPower_MMSE = mean(abs(Multitap_Channel_Signal_user_for_channel) .^ 2);
Noise_Variance_MMSE = SignalPower_MMSE / (10 ^ (SNR_OFDM / 10));

Nvariance_MMSE = sqrt(Noise_Variance_MMSE / 2);
n_MMSE = Nvariance_MMSE * (randn(length(Transmitted_signal_for_channel), 1) + 1j * randn(length(Transmitted_signal_for_channel), 1)); % Noise generation

Multitap_Channel_Signal_for_MMSE = Multitap_Channel_Signal_user_for_channel + n_MMSE;

% OFDM Receiver
[Unrecovered_signal, RS_User] = OFDM_Receiver(Multitap_Channel_Signal, Num_of_FFT, length_of_CP, length_of_symbol, Multitap_Channel_Signal_user);
[RS_MMSE, RS] = OFDM_Receiver(Multitap_Channel_Signal_for_MMSE, Num_of_FFT, length_of_CP, length_of_symbol, Multitap_Channel_Signal_user_for_channel);
Pilot_location_symbols = Pilot_starting_location : Pilot_interval : Frame_size;

[Received_pilot, ~] = Pilot_extract(RS_User(2:end, :), Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);
H_Ref = Received_pilot ./ Pilot_value_user;

% Channel estimation

% Perfect knowledge on Channel

% LS
[Received_pilot_LS, ~] = Pilot_extract(Unrecovered_signal(2:end, :), Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);

tStart_LS = tic;

profile on

H_LS = LS(Received_pilot_LS, Pilot_value_user);

profile off

t_LS(Frame, 1) = toc(tStart_LS);

MSE_LS_frame = mean(abs(H_LS - H_Ref).^2, 'all');

% MMSE

% linear MMSE
%H_MMSE = LMMSE(Pilot_location, Multitap_h, Num_of_FFT, SNR_OFDM, H_LS);

% MMSE MIMO-OFDM
%H_MMSE_h = MMSE_Channel_Tap(Received_pilot, Pilot_value_user, Num_of_FFT, 2, SNR, Multitap_h');

%for i = 1:size(Pilot_location, 2)
%    H_MMSE = H_MMSE_h(Pilot_location(:, i), :);
%end

%H_MMSE = zeros(size(Received_pilot_LS, 1), Num_of_pilot);

%for i = 1 : Num_of_pilot
%    H_MMSE(:, i) = MMSE_on_channel_estimation(Num_of_subcarriers, Rgg_value, Received_pilot_LS(:, i), ones(size(Received_pilot_LS, 1), 1) * Pilot_value_user, Noise_power);
%end

% MMSE on channel estimation
H_MMSE = zeros(size(Received_pilot_LS, 1), Num_of_pilot);
Pilot_Frame = Pilot_starting_location : Pilot_interval : Frame_size;

tStart_MMSE = tic;

for i = Pilot_Frame
    %[H, g] = MMSE_on_channel_estimation(Num_of_FFT, Rgg_value(:, :, i == Pilot_Frame), RS_MMSE(:, i), data_for_channel(:, i), Noise_Variance_MMSE);
    [H, g] = MMSE_on_channel_estimation(Num_of_FFT, Rgg_value(:, :, i == Pilot_Frame), Unrecovered_signal(:, i), data_in_IFFT(:, i), Noise_Variance);
    H = H(2:end, :);
    H_MMSE(:, i == Pilot_Frame) = H(Pilot_location(:, i == Pilot_Frame), :);
end

t_MMSE(Frame, 1) = toc(tStart_MMSE);

MSE_MMSE_frame = mean(abs(H_MMSE - H_Ref).^2, 'all');

% Deep learning
Res_feature_signal(:, :, 1) = real(Received_pilot_LS / Pilot_value_user);
Res_feature_signal(:, :, 2) = imag(Received_pilot_LS / Pilot_value_user);

tStart_DNN = tic;

H_DNN_feature = predict(DNN_Trained, Res_feature_signal);

t_DNN(Frame, 1) = toc(tStart_DNN);

H_DNN = H_DNN_feature(:, :, 1) + 1j * H_DNN_feature(:, :, 2);
MSE_DNN_frame = mean(abs(H_DNN - (RS(2:end, :) / sqrt(2))).^2, 'all');

% LS MSE calculation in each frame
LS_MSE_in_frame(Frame, 1) = MSE_LS_frame;

% MMSE MSE calculation in each frame
MMSE_MSE_in_frame(Frame, 1) = MSE_MMSE_frame;

% DNN MSE calculation in each frame
DNN_MSE_in_frame(Frame, 1) = MSE_DNN_frame;

end

% BER calculation
MSE_LS_over_SNR(SNR_Range == SNR, 1) = sum(LS_MSE_in_frame, 1) / Num_of_frame_each_SNR;

MSE_MMSE_over_SNR(SNR_Range == SNR, 1) = sum(MMSE_MSE_in_frame, 1) / Num_of_frame_each_SNR;

MSE_DNN_over_SNR(SNR_Range == SNR, 1) = sum(DNN_MSE_in_frame, 1) / Num_of_frame_each_SNR;

% Complexity calculation
T_LS(SNR_Range == SNR, 1) = sum(t_LS, 1) / Num_of_frame_each_SNR;
T_MMSE(SNR_Range == SNR, 1) = sum(t_MMSE, 1) / Num_of_frame_each_SNR;
T_DNN(SNR_Range == SNR, 1) = sum(t_DNN, 1) / Num_of_frame_each_SNR;

end
