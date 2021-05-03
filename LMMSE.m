function H_MMSE = LMMSE(Pilot_location, h, Num_of_FFT, SNR, H_LS)

H_MMSE = zeros(size(Pilot_location, 1), size(Pilot_location, 2));
H = fft(h, Num_of_FFT);
H = H(2:end, :);
for i = 1:size(Pilot_location, 2)
H_pilot = H(Pilot_location(:, i));
Rhh = H_pilot * H_pilot';
H_MMSE(:, i) = Rhh * pinv(Rhh + (1 / 10^(SNR / 10)) * eye(length(Pilot_location(:, i)))) * H_LS(:, i);
end