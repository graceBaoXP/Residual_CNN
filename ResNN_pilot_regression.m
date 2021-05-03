Data_Samples = 25000;
Training_Set_Rate = 0.995;
ValidationFrequency = 300;
SNR = 0:5:20;

[XTrain_RSRP, YTrain_RSRP, XValidation_RSRP, YValidation_RSRP] = Data_Generation_ReEsNet_48(Training_Set_Rate, SNR, Data_Samples);

Input_Layer_Size = size(XTrain_RSRP, [1, 2, 3]);

lgraph = layerGraph();

tempLayers = [
    imageInputLayer(Input_Layer_Size,"Name","imageinput")
    convolution2dLayer([3 3],16,"Name","conv_1","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],16,"Name","conv_2","Padding","same")
    reluLayer("Name","relu_1")
    convolution2dLayer([3 3],16,"Name","conv_3","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],16,"Name","conv_4","Padding","same")
    reluLayer("Name","relu_2")
    convolution2dLayer([3 3],16,"Name","conv_5","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],16,"Name","conv_6","Padding","same")
    reluLayer("Name","relu_3")
    convolution2dLayer([3 3],16,"Name","conv_7","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],16,"Name","conv_8","Padding","same")
    reluLayer("Name","relu_4")
    convolution2dLayer([3 3],16,"Name","conv_9","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_4")
    convolution2dLayer([3 3],16,"Name","conv_10","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_5")
    resize2dLayer("Name","resize-output-size","GeometricTransformMode","half-pixel","Method","bilinear","NearestRoundingMode","round","OutputSize",[72 14])
    convolution2dLayer([3 3],2,"Name","conv_11","Padding","same")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

clear tempLayers;

lgraph = connectLayers(lgraph,"conv_1","conv_2");
lgraph = connectLayers(lgraph,"conv_1","addition_1/in1");
lgraph = connectLayers(lgraph,"conv_1","addition_5/in2");
lgraph = connectLayers(lgraph,"conv_3","addition_1/in2");
lgraph = connectLayers(lgraph,"addition_1","conv_4");
lgraph = connectLayers(lgraph,"addition_1","addition_2/in1");
lgraph = connectLayers(lgraph,"conv_5","addition_2/in2");
lgraph = connectLayers(lgraph,"addition_2","conv_6");
lgraph = connectLayers(lgraph,"addition_2","addition_3/in1");
lgraph = connectLayers(lgraph,"conv_7","addition_3/in2");
lgraph = connectLayers(lgraph,"addition_3","conv_8");
lgraph = connectLayers(lgraph,"addition_3","addition_4/in1");
lgraph = connectLayers(lgraph,"conv_9","addition_4/in2");
lgraph = connectLayers(lgraph,"conv_10","addition_5/in1");

% Option settings
Options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'MiniBatchSize',128, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',10, ...
    'ValidationData',{XValidation_RSRP,YValidation_RSRP}, ...
    'ValidationFrequency',ValidationFrequency, ...
    'Shuffle','every-epoch', ...
    'Verbose',1, ...
    'L2Regularization',0.00000000002, ...
    'ExecutionEnvironment','auto', ...%'parallel'
    'Plots','training-progress');

% Train Network
[DNN_Trained, info] = trainNetwork(XTrain_RSRP, YTrain_RSRP, lgraph, Options);

% Network Pruning
%CDF_Layerweights

[XTest, YTest, ~, ~] = Data_Generation_ReEsNet_24(1, 10:5:20, 1000);

Ypred = predict(DNN_Trained, XTest);

MSE = sum(abs(Ypred - YTest), 'all') / size(Ypred, 4);
