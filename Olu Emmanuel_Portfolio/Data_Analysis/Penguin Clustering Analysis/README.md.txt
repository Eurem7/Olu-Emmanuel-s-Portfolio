Penguin Clustering Analysis
Project Overview
This project applies unsupervised machine learning techniques to analyze and segment penguin populations based on their physical characteristics. Using K-means clustering, the analysis identifies natural groupings within the penguin dataset and explores the morphological features that distinguish different segments.
Objective
To discover meaningful patterns in penguin physical measurements and identify distinct groups based on features such as culmen length, culmen depth, flipper length, and body mass.
Dataset
The analysis uses the penguins.csv dataset containing measurements of penguin specimens, including:

Culmen length (mm)
Culmen depth (mm)
Flipper length (mm)
Body mass (g)
Categorical variables (species, island, sex)

Methodology
1. Data Preprocessing

Converted categorical variables to dummy/indicator variables using one-hot encoding
Standardized all features using StandardScaler to ensure equal weighting in clustering

2. Optimal Cluster Selection

Implemented the elbow method by testing K-means with k ranging from 1 to 9
Calculated inertia (within-cluster sum of squares) for each k value
Visualized results to identify the optimal number of clusters

3. Clustering Analysis

Applied K-means clustering with k=3 clusters
Assigned cluster labels to each penguin observation
Visualized clusters using culmen length vs. culmen depth scatter plot

4. Cluster Profiling

Calculated mean values for all numeric features within each cluster
Generated statistical summaries to understand the characteristics of each segment

Technologies Used

Python: Primary programming language
Pandas: Data manipulation and analysis
Scikit-learn: Machine learning algorithms (KMeans, StandardScaler)
Matplotlib: Data visualization

Key Findings
The analysis identified 3 distinct penguin clusters based on physical measurements, with each cluster showing unique characteristics in terms of body dimensions and morphology.
Visualizations

Elbow Plot: Shows the relationship between number of clusters and inertia to determine optimal k
Cluster Scatter Plot: Displays penguin segments based on culmen dimensions with color-coded clusters