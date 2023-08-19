# Iris Flower Classification using Machine Learning

![iris](https://github.com/ishikawa-yui/Iris_flower_dataset_ML/assets/71602299/674ce41f-3840-4288-9e54-e6f3a8cbc8c9)


## Table of Contents
- [About the Dataset](#about-the-dataset)
  - [Context](#context)
  - [Content](#content)
  - [Acknowledgements](#acknowledgements)
- [Libraries Used](#libraries-used)
- [Visualizations](#visualizations)
- [Data Split](#data-split)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)

## About the Dataset

### Context
The Iris flower dataset, also known as Anderson's Iris data set, was introduced by the British statistician and biologist Ronald Fisher in his 1936 paper. Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. The dataset consists of 50 samples from each of three species of Iris: Iris Setosa, Iris Virginica, and Iris Versicolor. Four features were measured from each sample: the length and width of the sepals and petals, in centimeters.

### Content
The dataset contains the following attributes:
- Petal Length
- Petal Width
- Sepal Length
- Sepal Width
- Class (Species)

### Acknowledgements
This dataset is free and publicly available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).

## Libraries Used

The project harnesses the power of several Python libraries:
- **Seaborn**: For data visualization and exploration.
- **NumPy**: For numerical operations on the data.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For creating plots and visualizations.

## Visualizations

The project offers insightful visualizations to understand the data and model performance:
1. **Correlation Matrix Heatmap**: This heatmap visually represents the correlation between different features in the dataset. It helps to identify relationships and dependencies among features.
2. **Pairplot**: Pairwise relationships between features are explored using scatter plots. This visualization aids in recognizing patterns and clusters.
3. **KDE Plot**: A Kernel Density Estimation (KDE) plot of "sepal_length" versus "sepal_width" showcases the distribution of these two features.

## Data Split

The dataset is divided into training and testing sets using a split ratio of 0.3. This means that 70% of the data is used for training the machine learning model, while the remaining 30% is reserved for testing and evaluating the model's performance.

## Model Training

The machine learning model employed for this project is the **Support Vector Machine (SVM) Classifier**. To fine-tune the SVM model, a **GridSearch** is performed. The grid search explores various parameter combinations to optimize the model's performance. The parameters explored are:
- 'C': [0.1, 1, 10, 100]
- 'gamma': [1, 0.1, 0.01, 0.001]

## Evaluation

The model's effectiveness is evaluated using two primary metrics:
- **Confusion Matrix**: This matrix provides insights into the classification results, including true positives, true negatives, false positives, and false negatives.
- **Classification Report**: The report offers a more detailed evaluation, including metrics like precision, recall, F1-score, and support for each class.

## Conclusion

This project exemplifies the complete process of creating a machine learning model for Iris flower classification. By leveraging the Support Vector Machine algorithm and a rich array of data visualization tools, the model is trained and evaluated. The provided visualizations and evaluation metrics offer a comprehensive understanding of the model's performance and its ability to categorize different species of Iris flowers based on their unique features.

Feel free to explore the accompanying Jupyter Python notebook to dive deep into the code, analysis, and methodologies employed in this project.
