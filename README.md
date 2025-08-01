# Digital-Signal-Processing-Project-
Lung Nodule Segmentation and Detection Using Adaptive Thresholding and Watershed Transform and Cancerous CT Classification 

This repository contains MATLAB codes and resources for an automated pipeline to segment, detect, and classify lung nodules from CT scan images. The project leverages a combination of classical image processing techniques and deep learning (CNN) to facilitate early lung cancer detection.

Project Overview

Segmentation & Detection:
The project implements a robust workflow for lung and nodule segmentation from thoracic CT scans using adaptive thresholding, morphological operations, and marker-based watershed transform. Detected nodules are further measured and analyzed for clinical relevance.

Classification:
Segmented nodules are classified as benign or malignant using a Convolutional Neural Network (CNN) trained on labeled CT scan patches.

Datasets
1. LUNA16 Dataset (Segmentation and Detection)
We use the publicly available LUNA16 dataset for segmentation and detection tasks.

LUNA16 Dataset Link:
https://luna16.grand-challenge.org/Data/

2. Kaggle IQOTHNCCD Lung Cancer Dataset (Classification)
For the classification (benign/malignant) task, we use the IQOTHNCCD - Lung Cancer Dataset from Kaggle, which contains annotated CT images for deep learning experiments.

Kaggle Dataset Link:
https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset
