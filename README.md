# Cats vs Dogs SVM Classification

## Project Overview
This project is part of my Machine Learning internship at Skillcraft Tech Ltd.  
The goal is to implement image classification of cats and dogs using a Support Vector Machine (SVM), including data preprocessing, training, and evaluation.  

The project demonstrates practical understanding of machine learning pipelines, feature engineering, and model evaluation**.

---

## Challenge
To keep the workflow agile and practical, I set the following challenge:  

> Implement a complete SVM pipeline for cat and dog images within a **short, focused timeframe**, including:
- Data preparation  
- Feature engineering (resize & flatten)  
- Model training & hyperparameter selection  
- Evaluation (accuracy, confusion matrix)  

---

## Dataset
The project uses the **[Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)**:  
- Contains 37 pet categories (cats & dogs)  
- For binary classification, all cats are labeled as “0” and all dogs as “1”

## Dataset
Please download the Kaggle Dogs vs Cats dataset: https://www.kaggle.com/c/dogs-vs-cats/data
Place the images in `data/images/` before running the scripts.

## Results
The SVM model achieves an **accuracy of ~91% on the Kaggle Dogs vs Cats dataset (binary classification: cats vs dogs).  
The confusion matrix and classification report are generated locally when running the scripts.
