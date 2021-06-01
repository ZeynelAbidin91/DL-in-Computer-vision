# Data Analysis Checklist

## EDA Exploratory Data Analysis

(EDA). It is a classical and under-utilized approach that helps you quickly build a relationship with the new data. 

It is always better to explore each data set using multiple exploratory techniques and compare the results. This step aims to understand the dataset, identify the missing values & outliers if any using visual and quantitative methods to get a sense of the story it tells. It suggests the next logical steps, questions, or areas of research for your project.

Exploratory data analysis (EDA) is used by data scientists to analyze and investigate datasets and summarize their main characteristics, often employing data visualization methods. It helps determine how best to manipulate data sources to get the answers you need, making it easier for data scientists to discover patterns, spot anomalies, test a hypothesis, or check assumptions.

> EDA is primarily used to see what data can reveal beyond the formal modeling or hypothesis testing task and provides a provides a better understanding of data set variables and the relationships between them.

### **Steps in Data Exploration and Preprocessing:**

- Identification of variables and data types

    The very first step in exploratory data analysis is to identify the type of variables in the dataset. Variables are of two types — Numerical and Categorical. They can be further classified as follows:

    ![https://miro.medium.com/max/1050/1*nAsTC_qYpOSftm-9c9HXMg.png](https://miro.medium.com/max/1050/1*nAsTC_qYpOSftm-9c9HXMg.png)

    `df.dtypes, df.shape, df.describe()`

- **Analyzing the basic metrics**
- **Non-Graphical Univariate Analysis**

    To get the count of unique values:

    The value_counts() method in Pandas returns a series containing the counts of all the unique values in a column. The output will be in descending order so that the first element is the most frequently-occurring element.

    `df['column_name'].value_counts()`

    To get the list & number of unique values:

    The nunique() function in Pandas returns a series with several distinct observations in a column.

    `df['column_name'].nunique()`

    Similarly, the unique() function of pandas returns the list of unique values in the dataset.

    `df['column_name'].unique()`

- **Graphical Univariate Analysis**

    Histogram:

    Histograms are one of the most common graphs used to display numeric data. Histograms two important things we can learn from a histogram:

    1. distribution of the data — Whether the data is normally distributed or if it’s skewed (to the left or right)
    2. To identify outliers — Extremely low or high values that do not fall near any other data points.

    `train['ltv'].hist(bins=25)`

    Box Plots:

    A Box Plot is the visual representation of the statistical summary of a given data set.

    The Summary includes:

    - Minimum
    - First Quartile
    - Median (Second Quartile)
    - Third Quartile
    - Maximum

    ![https://miro.medium.com/max/723/1*8Hm2XKlmwhX7mlY0NcMQvQ.png](https://miro.medium.com/max/723/1*8Hm2XKlmwhX7mlY0NcMQvQ.png)

    `train.boxplot(column='disbursed_amount')`

    `sns.boxplot(x=train['asset_cost'])`

    Count Plots:

    A count plot can be thought of as a histogram across a categorical, instead of numeric, variable. It is used to find the frequency of each category.

    `sns.countplot(train.loan_default)`

- **Bivariate Analysis**
- Variable transformations
- Missing value treatment
- Outlier treatment
- Correlation Analysis
- Dimensionality Reduction

Data Type Conversion using to_datetime() and astype() methods:

Pandas astype() method is used to change the data type of a column. to_datetime() method is used to change, particularly to DateTime type. When the data frame is imported from a CSV file, the data type of the columns is set automatically, which many times is not what it actually should have. For example, in the above dataset, Date.of.Birth and DisbursalDate are both set as object type, but they should be DateTime.

`train['Date.of.Birth']= pd.to_datetime(train['Date.of.Birth'])`

`train['ltv'] = train['ltv'].astype('int64')`