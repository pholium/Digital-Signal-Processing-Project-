clc;
clear;
close all;

% === Load trained model ===
load('trainedCNN.mat', 'net');

% === Recreate the test set ===
testFolder = 'D:\DSP Project\Binary_CT_images_1\test';

imdsTest = imageDatastore(testFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

%augImdsTest = augmentedImageDatastore([64 64], imdsTest);
augImdsTest = augmentedImageDatastore([64 64 3], imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

% === Predict on test set ===
YPred = classify(net, augImdsTest);
YTrue = imdsTest.Labels;

% === Confusion Matrix ===
figure;
confusionchart(YTrue, YPred);
title('Confusion Matrix - Test Set');

% === Metric Calculation ===
cm = confusionmat(YTrue, YPred);
labels = categories(YTrue);

if strcmp(labels{1}, 'Cancerous') && strcmp(labels{2}, 'Non-cancerous')
    TP = cm(1,1); FN = cm(1,2);
    FP = cm(2,1); TN = cm(2,2);
else
    TP = cm(2,2); FN = cm(2,1);
    FP = cm(1,2); TN = cm(1,1);
end

accuracy = (TP + TN) / sum(cm(:));
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1 = 2 * (precision * recall) / (precision + recall);

% === Display Metrics ===
fprintf('📊 Model Evaluation Metrics:\n');
fprintf('-----------------------------\n');
fprintf('✅ Accuracy     : %.2f%%\n', accuracy * 100);
fprintf('🎯 Precision    : %.2f%%\n', precision * 100);
fprintf('🧠 Recall       : %.2f%%\n', recall * 100);
fprintf('📎 F1 Score     : %.2f%%\n', f1 * 100);

%% === Optional: Manually test a single image ===
[filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp','Image Files'}, 'Select an image to test');
if isequal(filename,0)
    disp('❌ No file selected.');
else
    % Read and resize the selected image
    imgPath = fullfile(pathname, filename);
    img = imread(imgPath);
    img = imresize(img, [64 64]);

    % Convert grayscale to RGB if needed
    if size(img,3) == 1
        img = cat(3, img, img, img);
    end

    % Classify using loaded model
    predictedLabel = classify(net, img);

    % Display result
    figure; imshow(img);
    title(['Predicted: ', char(predictedLabel)], 'FontSize', 14, 'Color', 'g');
    disp(['✅ Predicted Class: ', char(predictedLabel)]);
end
