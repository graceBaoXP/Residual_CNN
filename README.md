# Residual_CNN
Repeat the results of the paper called 'Deep Residual Learning Meets OFDM Channel Estimation' using MATLAB.

I think they did not release the code so I was not sure whats I did is hundred percent correct, so just have fun.

It is a simple demo for implementing residual Neural Network in MATLAB.

It is a version of my code indeed but long long ago, so if there is any errors let me know. From my testing it works, at least I can send you a copy of trained DAG NN.

Pruning.m is used to prune the Neural Network. DAG is a read only structure, that is the reason why we need a module to do pruning.

CDF_Layerweights.m is used to have a view on the CDF of weights and output the location of the weights prepared to be pruned for Pruning.m to do pruning.

I know the MSE performance is worse than the results of the paper called 'Deep Residual Learning Meets OFDM Channel Estimation', but we should have a common knowledge on LS so my LS and MMSE time domain estimation resluts are consistent with the paper called 'On channel estimation in OFDM systems', which is strong enough to be a reference.
