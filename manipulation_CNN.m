clc;
clear all;
close all;

%% Step 1: Load training and test images

trainFolder = 'D:\DSP Project\Binary_CT_images_1\train'; 
testFolder  = 'D:\DSP Project\Binary_CT_images_1\test'; 

imdsTrain = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsTest = imageDatastore(testFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Check the number of images in each class
countEachLabel(imdsTrain)
countEachLabel(imdsTest)

%% Step 2: Resize and optionally augment images

inputSize = [64 64];  % You can also use [128 128] or [224 224]

% Data augmentation settings (optional but helpful for generalization)
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-25 25], ...
    'RandXReflection', true, ...
    'RandYReflection', false, ...
    'RandXScale', [0.8 1.2], ...
    'RandYScale', [0.8 1.2],...
    'RandXTranslation', [-8 8], ...
    'RandYTranslation', [-8 8]);

% Prepare augmented image datastores
%augImdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    %'DataAugmentation', imageAugmenter);

%augImdsTest = augmentedImageDatastore(inputSize, imdsTest);  % no augmentation
augImdsTrain = augmentedImageDatastore([64 64 3], imdsTrain, ...
    'DataAugmentation', imageAugmenter, ...
    'ColorPreprocessing', 'gray2rgb');

augImdsTest = augmentedImageDatastore([64 64 3], imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

% ✅ Check image size to detect color mode (grayscale or RGB)
%img = readimage(imdsTrain, 1);
%disp(size(img));

%% Step 3: Define CNN layers

layers = [
    imageInputLayer([64 64 3], 'Name', 'input')

    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv_1')
    batchNormalizationLayer('Name', 'bn_1')
    reluLayer('Name', 'relu_1')

    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv_1b')  % NEW
    batchNormalizationLayer('Name', 'bn_1b')
    reluLayer('Name', 'relu_1b')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool_1')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_2')
    batchNormalizationLayer('Name', 'bn_2')
    reluLayer('Name', 'relu_2')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_2b')  % NEW
    batchNormalizationLayer('Name', 'bn_2b')
    reluLayer('Name', 'relu_2b')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool_2')

    dropoutLayer(0.3, 'Name', 'dropout')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv_3')
    batchNormalizationLayer('Name', 'bn_3')
    reluLayer('Name', 'relu_3')
    averagePooling2dLayer(2, 'Stride', 2, 'Name', 'avg_pool')

    fullyConnectedLayer(2, 'Name', 'fc')  % 2 classes
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

%% Step 4: Set training options

 options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 1e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augImdsTest, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment','auto');

% === Train the CNN ===   
net = trainNetwork(augImdsTrain, layers, options);

% === Save trained model ===
save('trainedCNN.mat', 'net');
disp('✅ Model saved as trainedCNN.mat');   
