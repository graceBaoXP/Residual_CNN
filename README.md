# Residual_CNN
Repeat the results of the paper called 'Deep Residual Learning Meets OFDM Channel Estimation' using MATLAB R2020b

It is defined as ReEsNet. I think they did not release the code so I am not sure what I did is hundred percent correct, so just have fun.

It is a simple demo for implementing residual Neural Network in MATLAB.

It is a version of my part code indeed but long long ago, so if there is any errors let me know. It works, at least I can send you a copy of trained DAG NN. If you have any question let me know.

%% Pruning

Go https://github.com/dianixn/Pruning to see the details.

Pruning.m is used to prune the Neural Network. DAG is a read only structure, that is the reason why we need a module to do pruning.

CDF_Layerweights.m is used to have a view on the CDF of weights and output the location of the weights prepared to be pruned for Pruning.m to do pruning.

%% MMSE

MMSE_Channel_Tap.m is based on the assumption that jakes model.

LMMSE.m is linear MMSE, used profile to view some information so I made it as a module.

%% Training

Run ResNN_pilot_regression.m to train the residual CNN.

ResImNet.m is the untrained DAG network which deploys interpolation to resize the output.

Residual_transposed.m the untrained DAG network which deploys the transpose convolution to resize the output.

%% Test

Use Demonstration_of_H_regression.m to test.

%% Channel

Simple Rayleigh channel and Doppler shift Rayleigh channel.

WINNER2 Channel is not deployed, and you can download the WINNER2 packet so you can use the WINNER2 channel. Go https://github.com/dianixn/Win2_Channel for simple implementation of WINNER2 channel, but I suggest you to have a check on MATLAB offical page :)

%% Result

I know the MSE performance is worse than the results of the paper called 'Deep Residual Learning Meets OFDM Channel Estimation', but we should have a common knowledge on LS so my LS and MMSE time domain estimation resluts are consistent with the paper called 'On channel estimation in OFDM systems', which is strong enough to be a reference. You can also compare the result of ChannelNet from the original paper.

%% By the way is there anyone know how to derive MMSE mathmatiscally? I would appreciate it if you can highlight me some course slides or book related.
%% By the way is there anyone know how to derivate matrix, not that deep learning type, say if one m by 1 matrix derivated by n by 1 matrix, the result should be m by n, is that how we did in MMSE derivation? Help :(
