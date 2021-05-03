figure;
semilogy(SNR_Range,MSE_LS_over_SNR,'b');
hold on
semilogy(SNR_Range,MSE_MMSE_over_SNR,'r');
hold on
semilogy(SNR_Range,MSE_DNN_over_SNR,'g');

legend('LS', ...
    'MMSE', ...
    'ReEsNet');
xlabel('SNR in dB');
ylabel('MSE');
title('MSE 48 pilot');
grid on;
hold off;
