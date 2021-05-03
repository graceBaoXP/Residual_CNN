function H_MMSE = MMSE_Channel_Tap(Received_Pilot, Pilot_Value, Nfft, Nps, SNR, h)

% 2020.06.09

SNR_HEX = 10^(SNR / 10);
H_LS = Received_Pilot ./ Pilot_Value;

Np = (Nfft - 1) / Nps;

K = 0: length(h) - 1;
hh = h * h';
tmp = h .* conj(h) .* K;
r = sum(tmp) / hh;
r2 = tmp * K .'/hh;

tau_rms = sqrt(r2 - r^2);
df = 1/Nfft;
j2pi_tau_df = 1j * 2 * pi * tau_rms * df;
K1 = repmat([0 : Nfft - 1].', 1, Np);
K2 = repmat([0 : Np - 1], Nfft, 1);
rf = 1./(1 + j2pi_tau_df * (K1 - K2 * Nps));
K3 = repmat([0 : Np - 1].', 1, Np);
K4 = repmat([0 : Np - 1], Np, 1);
rf2 = 1./(1 + j2pi_tau_df * Nps * (K3 - K4));
Rhp = rf;
Rpp = rf2 + (eye(length(H_LS)) / SNR_HEX);
H_MMSE = Rhp * pinv(Rpp) * H_LS;
