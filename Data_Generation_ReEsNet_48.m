% Channel Regression
% Data generation

function [Xtraining_Array_RSRP, Ytraining_regression_double_RSRP, Xvalidation_RSRP, Yvalidation_regression_double_RSRP] = Data_Generation_ReEsNet_48(Training_set_ratio, SNR_Range, Num_of_frame_each_SNR)

Num_of_subcarriers = 72; %126
Num_of_FFT = Num_of_subcarriers + 1;
length_of_CP = 16;

Num_of_symbols = 10;
Num_of_pilot = 4;
Frame_size = Num_of_symbols + Num_of_pilot;

Pilot_interval = 4;
Pilot_starting_location = 1;
Pilot_location = [(1:6:Num_of_subcarriers)', (3:6:Num_of_subcarriers)', (4:6:Num_of_subcarriers)', (6:6:Num_of_subcarriers)'];
Pilot_value_user = 1 + 1j;

length_of_symbol = Num_of_FFT + length_of_CP;

Xtraining_Array_RSRP = zeros(size(Pilot_location, 1), size(Pilot_location, 2), 2, Training_set_ratio * Num_of_frame_each_SNR * size(SNR_Range, 2));
Ytraining_regression_double_RSRP = zeros(Num_of_subcarriers, Frame_size, 2, Training_set_ratio * Num_of_frame_each_SNR * size(SNR_Range, 2));
Xvalidation_RSRP = zeros(size(Pilot_location, 1), size(Pilot_location, 2), 2, Num_of_frame_each_SNR * size(SNR_Range, 2) - Training_set_ratio * Num_of_frame_each_SNR * size(SNR_Range, 2));
Yvalidation_regression_double_RSRP = zeros(Num_of_subcarriers, Frame_size, 2, Num_of_frame_each_SNR * size(SNR_Range, 2) - Training_set_ratio * Num_of_frame_each_SNR * size(SNR_Range, 2));

for SNR = SNR_Range
        
for Frame = 1 : Num_of_frame_each_SNR

%% Data generation

QPSK_signal = ones(Num_of_subcarriers, Num_of_symbols);

% Pilot inserted
[data_in_IFFT, data_location] = Pilot_Insert(Pilot_value_user, Pilot_starting_location, Pilot_interval, Pilot_location, Frame_size, Num_of_FFT, QPSK_signal);
[data_for_channel, ~] = Pilot_Insert(1, Pilot_starting_location, Pilot_interval, kron((1 : Num_of_subcarriers)', ones(1, Num_of_pilot)), Frame_size, Num_of_FFT, QPSK_signal);
data_for_channel(1, :) = 1;

% OFDM Transmitter
[Transmitted_signal, ~] = OFDM_Transmitter(data_in_IFFT, Num_of_FFT, length_of_CP);
[Transmitted_signal_for_channel, ~] = OFDM_Transmitter(data_for_channel, Num_of_FFT, length_of_CP);

%% Channel

% AWGN Channel
SNR_OFDM = SNR + 10 * log10((Num_of_subcarriers / Num_of_FFT));
awgnChan = comm.AWGNChannel('NoiseMethod', 'Signal to noise ratio (SNR)');
awgnChan.SNR = SNR_OFDM;

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

Multitap_Channel_Signal_user_for_channel = conv(Transmitted_signal_for_channel, Multitap_h);
Multitap_Channel_Signal_user_for_channel = Multitap_Channel_Signal_user_for_channel(1 : length(Transmitted_signal_for_channel));

%% OFDM Receiver
[Received_signal, H_Ref] = OFDM_Receiver(Multitap_Channel_Signal, Num_of_FFT, length_of_CP, length_of_symbol, Multitap_Channel_Signal_user_for_channel);

Pilot_location_symbols = Pilot_starting_location : Pilot_interval : Frame_size;
[Received_pilot, ~] = Pilot_extract(Received_signal(2:end, :), Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);
H_LS = Received_pilot / Pilot_value_user;

if Frame <= fix(Training_set_ratio * Num_of_frame_each_SNR)
    Training_index = Frame + fix(Training_set_ratio * Num_of_frame_each_SNR) * (find(SNR_Range == SNR) - 1);
    Xtraining_Array_RSRP(:, :, 1, Training_index) = real(H_LS);
    Xtraining_Array_RSRP(:, :, 2, Training_index) = imag(H_LS);
    Ytraining_regression_double_RSRP(:, :, 1, Training_index) = real(H_Ref(2:end, :));
    Ytraining_regression_double_RSRP(:, :, 2, Training_index) = imag(H_Ref(2:end, :));
else
    Validation_index = Frame - Training_set_ratio * Num_of_frame_each_SNR + (find(SNR_Range == SNR) - 1) * (Num_of_frame_each_SNR - Training_set_ratio * Num_of_frame_each_SNR);
    Xvalidation_RSRP(:, :, 1, Validation_index) = real(H_LS);
    Xvalidation_RSRP(:, :, 2, Validation_index) = imag(H_LS);
    Yvalidation_regression_double_RSRP(:, :, 1, Validation_index) = real(H_Ref(2:end, :));
    Yvalidation_regression_double_RSRP(:, :, 2, Validation_index) = imag(H_Ref(2:end, :));
end

end

end

end
