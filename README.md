[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/CDOoMPNW)
# AI Statistics Lab: Robust Linear Regression with Outliers

## Topic
Robust regression models for handling outliers.

This lab is inspired by the NVIDIA Technical Blog post:

Dealing with Outliers Using Three Robust Linear Regression Models

Blog link:
https://developer.nvidia.com/blog/dealing-with-outliers-using-three-robust-linear-regression-models/

The blog compares:
- Ordinary Linear Regression
- Huber Regression
- RANSAC Regression
- Theil-Sen Regression

The key idea is that ordinary linear regression can be strongly influenced by outliers, while robust regression methods reduce the effect of extreme observations.

---

## Learning Objectives

By the end of this lab, students should be able to:

1. Generate a synthetic regression dataset.
2. Add artificial outliers to the dataset.
3. Fit ordinary linear regression.
4. Fit robust regression models:
   - Huber Regression
   - RANSAC Regression
   - Theil-Sen Regression
5. Compare model coefficients.
6. Evaluate which model is more robust to outliers.
7. Visualize fitted regression lines.
8. Visualize RANSAC inliers and outliers.

---

## Background

Linear regression fits a line of the form:

    y = beta*x + b

where:

    beta = slope / coefficient
    b    = intercept

When outliers are present, ordinary linear regression may produce a biased coefficient.

Robust regression methods attempt to reduce the effect of outliers.

In this lab, you will create a synthetic dataset where the true coefficient is known. Then you will add outliers and compare how close each model coefficient is to the true coefficient.

---

## Question 1: Dataset Generation, Outlier Injection, and Data Visualization

You must implement:

1. generate_clean_data()
2. add_outliers()
3. plot_dataset_with_outliers()

The clean dataset should contain:
- 500 samples
- 1 feature
- 1 informative feature
- noise = 20
- random_state = 42

Then add 25 outliers to the first 25 observations.

Use this outlier pattern:

    X[:25] = 10 + 0.75 * random_normal_values
    y[:25] = -15 + 20 * random_normal_values

The function plot_dataset_with_outliers() should:
- plot normal data points
- plot the first 25 artificial outliers using a different marker or color
- label axes
- add a title
- return a matplotlib Figure object

---

## Question 2: Robust Regression and Model Visualization

You must implement:

1. fit_linear_regression()
2. fit_huber_regression()
3. fit_ransac_regression()
4. fit_theilsen_regression()
5. coefficient_errors()
6. best_robust_model()
7. ransac_outlier_summary()
8. plot_regression_fits()
9. plot_ransac_inliers_outliers()

The models should return their fitted slope coefficient.

The function coefficient_errors() should compute:

    absolute_error = abs(estimated_coefficient - true_coefficient)

The function best_robust_model() should return the robust model with the smallest coefficient error among:

    huber_regression
    ransac_regression
    theilsen_regression

Do not include ordinary linear_regression when selecting the best robust model.

The function ransac_outlier_summary() should return:

    total_outliers_detected, added_outliers_detected

where:
- total_outliers_detected = total number of observations classified as outliers by RANSAC
- added_outliers_detected = how many of the first 25 artificial outliers were detected

The function plot_regression_fits() should:
- plot the dataset
- plot fitted lines for Linear Regression, Huber, RANSAC, and Theil-Sen
- add legend, title, x-label, and y-label
- return a matplotlib Figure object

The function plot_ransac_inliers_outliers() should:
- fit RANSAC
- plot inliers and outliers separately
- add legend, title, x-label, and y-label
- return a matplotlib Figure object

---

## Files

Students must edit only:

    AI_stats_lab.py

Do not edit:

    test_AIstats_lab.py

---

## Run Locally

Install dependencies:

    pip install numpy scikit-learn matplotlib pytest

Run tests:

    pytest test_AIstats_lab.py

---

## Expected Concepts

Students should observe that:
- Linear regression is heavily distorted by artificial outliers.
- Robust models usually recover a coefficient closer to the true coefficient.
- RANSAC explicitly identifies inliers and outliers.
- Visualization helps explain the effect of outliers on model fitting.

