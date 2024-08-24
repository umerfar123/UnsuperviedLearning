# um_UnsuperviedLearning

## Introduction

Unsupervised learning in artificial intelligence is a type of machine learning that learns from data without human supervision. Unlike supervised learning, unsupervised machine learning models are given unlabeled data and allowed to discover patterns and insights without any explicit guidance or instruction. The main two algorithms that we will use in this project is KMeans and DBSCAN(Density Based Spatial Clustering With Noise).

This project can be used for learning these algorithms, by understanding how algorithms works when certain parameters are changed.When it comes to kmeans you can change the numbers of clusters parameters and see what happens to the clustered data. Similarly when DBSCAN parameters such as eps and min_samples can also be adjusted.

## Steps
1. Install libraries mentioned in requirements.txt
2. Run the main.py file using streamlit
3. Adjust the parameters to see the performance of model

## Note
+ While adjusting eps in DBSCAN if the uniques cluster printed have -1 in it ,that means some points have considered as noise and noise points also get plotted in scatterplot and may feel like inaccurate,try setting the eps such that there is no -1 in no.of unique clusters.

Also if you would like to contribute to the project by adding extra algorihtms please feel free to do it.........
