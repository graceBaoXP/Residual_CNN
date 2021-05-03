# Residual_CNN
Repeat the results of the paper called 'Deep Residual Learning Meets OFDM Channel Estimation' using MATLAB.

It is defined by ReEsNet.

I think they did not release the code so I was not sure whats I did is hundred percent correct, so just have fun.

It is a simple demo for implementing residual Neural Network in MATLAB.

It is a version of my part code indeed but long long ago, so if there is any errors let me know. It works, at least I can send you a copy of trained DAG NN.

%% Pruning

Go https://github.com/dianixn/Pruning to see the details. By the way, my name is Dianxin, that's a typo.

Pruning.m is used to prune the Neural Network. DAG is a read only structure, that is the reason why we need a module to do pruning.

CDF_Layerweights.m is used to have a view on the CDF of weights and output the location of the weights prepared to be pruned for Pruning.m to do pruning.

%% MMSE

MMSE_Channel_Tap.m is based on the assumption that jakes model.

LMMSE.m is linear MMSE, used profile to view some information so I made it as a module.

%% Training

Run ResNN_pilot_regression.m to train the residual CNN.

%% Channel

You can see that there are a simple Rayleigh channel and doppler shift rayleigh channel, so depend on your preference to train the model.

%% Result

I know the MSE performance is worse than the results of the paper called 'Deep Residual Learning Meets OFDM Channel Estimation', but we should have a common knowledge on LS so my LS and MMSE time domain estimation resluts are consistent with the paper called 'On channel estimation in OFDM systems', which is strong enough to be a reference.
