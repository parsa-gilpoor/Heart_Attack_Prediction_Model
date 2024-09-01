**English Description**

<span style="color:yellow; font-size:18px;">The Heart Disease Dataset is a widely used dataset for predicting the likelihood of heart disease based on various medical attributes. It typically includes features such as age, sex, blood pressure, cholesterol levels, and other health indicators. This dataset is valuable for building and evaluating machine learning models in the context of healthcare, particularly for risk assessment and diagnostic purposes. The goal is to classify whether a patient has heart disease or not, based on these attributes.</span>

In this code, we use several libraries and commands for data preparation and analysis:

1. **<span style="color:blue; font-weight:bold;">`import numpy as np`</span>**:
   - <span style="color:green;">The <strong>NumPy</strong> library is used for numerical and matrix operations in Python.</span> It allows for efficient manipulation of arrays and matrices and performing complex mathematical computations.

2. **<span style="color:blue; font-weight:bold;">`import pandas as pd`</span>**:
   - <span style="color:green;">The <strong>Pandas</strong> library is used for data analysis and working with structured data.</span> It provides powerful tools for reading, writing, and processing tabular data.

3. **<span style="color:blue; font-weight:bold;">`import matplotlib.pyplot as plt`</span>**:
   - <span style="color:green;">The <strong>Matplotlib</strong> library is used for creating visualizations and plots.</span> Specifically, <strong>pyplot</strong> is used for generating graphs and figures.

4. **<span style="color:blue; font-weight:bold;">`from sklearn import preprocessing`</span>**:
   - <span style="color:green;">The <strong>preprocessing</strong> module from <strong>Scikit-learn</strong> is used for preparing and transforming data, such as normalization and standardization.</span> This can improve the performance of machine learning models.

5. **<span style="color:blue; font-weight:bold;">`%matplotlib inline`</span>**:
   - <span style="color:green;">This magic command in Jupyter Notebook ensures that plots are displayed directly within the notebook, eliminating the need for <strong>plt.show()</strong>.</span> It allows for interactive and inline visualization of plots.

By using these libraries and commands, we can prepare data, perform analyses, and visualize the results within the notebook.

**<span style="color:blue; font-weight:bold;">`mdi=pd.read_csv("heart_disease_dataset.csv")`</span>**:
- <span style="color:green;">This line of code reads the CSV file named <strong>"heart_disease_dataset.csv"</strong> into a DataFrame called <strong>mdi</strong> using the <strong>pandas</strong> library.</span>

**<span style="color:blue; font-weight:bold;">`mdi.head(10)`</span>**:
- <span style="color:green;">This line displays the first 10 rows of the DataFrame <strong>mdi</strong>.</span> It helps us quickly inspect the contents of the dataset and understand its structure.
**<span style="color:blue; font-weight:bold;">`mdi=pd.read_csv("heart_disease_dataset.csv")`</span>**:

**<span style="color:blue; font-weight:bold;">`mdi_cleaned = mdi.dropna()`</span>**:
- <span style="color:green;">This line of code removes any rows with missing values from the <strong>mdi</strong> DataFrame and creates a new DataFrame called <strong>mdi_cleaned</strong>.</span>

**<span style="color:blue; font-weight:bold;">`print(mdi_cleaned)`</span>**:
- <span style="color:green;">This line prints the contents of the <strong>mdi_cleaned</strong> DataFrame.</span> It allows us to view the dataset after removing rows with missing values.
**<span style="color:blue; font-weight:bold;">`mdi_cleaned.describe()`</span>**:
- <span style="color:green;">This line of code generates descriptive statistics of the <strong>mdi_cleaned</strong> DataFrame.</span> It provides summary statistics such as count, mean, standard deviation, minimum, and maximum values for each numerical column in the dataset. This helps to understand the distribution and general characteristics of the data.
  
**<span style="color:blue; font-weight:bold;">`mdi_cleaned['Heart_Rate'].value_counts()`</span>**:
- <span style="color:green;">This line of code counts the occurrences of each unique value in the <strong>'Heart_Rate'</strong> column of the <strong>mdi_cleaned</strong> DataFrame.</span> It provides a frequency distribution of heart rate values in the dataset, helping to understand how often each heart rate occurs.
**<span style="color:blue; font-weight:bold;">`mdi_cleaned.hist(column='Heart_Rate', bins=50)`</span>**:
- <span style="color:green;">This line of code generates a histogram of the <strong>'Heart_Rate'</strong> column in the <strong>mdi_cleaned</strong> DataFrame with 50 bins.</span> The histogram visually represents the distribution of heart rate values, showing how frequently each range of values occurs. The number of bins (50) determines the granularity of the histogram.

**<span style="color:blue; font-weight:bold;">`mdi_cleaned.columns`</span>**:
- <span style="color:green;">This line of code returns the list of column names in the <strong>mdi_cleaned</strong> DataFrame.</span> It helps to quickly view the names of all the columns in the dataset, allowing us to understand the structure and available data fields.

**<span style="color:blue; font-weight:bold;">`from sklearn.preprocessing import LabelEncoder`</span>**:
- <span style="color:green;">This line imports the <strong>LabelEncoder</strong> class from the <strong>sklearn.preprocessing</strong> module.</span> LabelEncoder is used to convert categorical labels into numeric values, making them suitable for machine learning models.

**<span style="color:blue; font-weight:bold;">`mdi_cleaned = mdi.copy()`</span>**:
- <span style="color:green;">This line creates a copy of the original DataFrame <strong>mdi</strong> and stores it in <strong>mdi_cleaned</strong>.</span> This is done to preserve the original data while making modifications.

**<span style="color:blue; font-weight:bold;">`binary_features = ['Gender', 'Smoking', 'Family_History', 'Diabetes', 'Obesity', 'Exercise Induced Angina']`</span>**:
- <span style="color:green;">This line defines a list of binary categorical features that will be encoded using <strong>LabelEncoder</strong>.</span> These features have only two possible values (e.g., Yes/No).

**<span style="color:blue; font-weight:bold;">`for feature in binary_features:`</span>**:
- <span style="color:green;">This loop iterates through each feature in the <strong>binary_features</strong> list and encodes it using <strong>LabelEncoder</strong>.</span> The encoded values are then stored back in the <strong>mdi_cleaned</strong> DataFrame.

**<span style="color:blue; font-weight:bold;">`categorical_features = ['Alcohol_Intake', 'Chest_Pain Type']`</span>**:
- <span style="color:green;">This line defines a list of categorical features with more than two categories.</span> These features will be one-hot encoded.

**<span style="color:blue; font-weight:bold;">`mdi_cleaned = pd.get_dummies(mdi_cleaned, columns=categorical_features)`</span>**:
- <span style="color:green;">This line applies one-hot encoding to the features in the <strong>categorical_features</strong> list.</span> The result is a DataFrame where each category is represented by a separate binary column.

**<span style="color:blue; font-weight:bold;">`X = mdi_cleaned[['Age', 'Gender', 'Cholesterol', 'Blood_Pressure', 'Heart_Rate', 'Smoking', 'Exercise_Hours', 'Family_History', 'Diabetes', 'Obesity', 'Stress_Level', 'Blood_Sugar', 'Exercise Induced Angina', 'Heart Disease']].values.astype

**<span style="color:blue; font-weight:bold;">`y = mdi_cleaned['Heart Disease'].values`</span>**:
- <span style="color:green;">This line of code extracts the values from the <strong>'Heart Disease'</strong> column of the <strong>mdi_cleaned</strong> DataFrame and stores them in the variable <strong>y</strong>.</span> This variable <strong>y</strong> will typically be used as the target variable in a machine learning model, representing whether each patient has heart disease or not.

**<span style="color:blue; font-weight:bold;">`y[0:5]`</span>**:
- <span style="color:green;">This line retrieves the first 5 elements of the <strong>y</strong> array.</span> This is useful for quickly inspecting the initial values of the target variable to ensure they were correctly extracted.

**<span style="color:blue; font-weight:bold;">`X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))`</span>**:
- <span style="color:green;">This line of code standardizes the feature matrix <strong>X</strong> by scaling each feature to have a mean of 0 and a standard deviation of 1 using <strong>StandardScaler</strong> from <strong>sklearn.preprocessing</strong>.</span> The `.fit(X)` method calculates the mean and standard deviation for scaling, while `.transform(X)` applies the scaling transformation. This step ensures that all features contribute equally to the model and helps in improving the performance of machine learning algorithms.

**<span style="color:blue; font-weight:bold;">`X[0:5]`</span>**:
**<span style="color:blue; font-weight:bold;">`X_df = pd.DataFrame(X)`</span>**:
- <span style="color:green;">This line of code converts the standardized feature matrix <strong>X</strong> into a DataFrame object and stores it in <strong>X_df</strong>.</span> This allows for easier manipulation and analysis of the data using the powerful tools provided by the pandas library.

**<span style="color:blue; font-weight:bold;">`X_df.info()`</span>**:
- <span style="color:green;">This line provides a concise summary of the DataFrame <strong>X_df</strong>, including the number of entries, columns, non-null values, and data types.</span> This is useful for quickly understanding the structure of the DataFrame and ensuring that the data is in the expected format.
**<span style="color:blue; font-weight:bold;">`from sklearn.model_selection import train_test_split`</span>**:
- <span style="color:green;">This line imports the <strong>train_test_split</strong> function from the <strong>sklearn.model_selection</strong> module.</span> This function is used to split the dataset into training and testing sets, allowing for the evaluation of the model's performance on unseen data.

**<span style="color:blue; font-weight:bold;">`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)`</span>**:
- <span style="color:green;">This line splits the feature matrix <strong>X</strong> and the target variable <strong>y</strong> into training and testing sets.</span> The parameter <strong>test_size=0.2</strong> specifies that 20% of the data should be used for testing, while the remaining 80% is used for training. The <strong>random_state=4</strong> parameter ensures that the split is reproducible, meaning that the same split will occur every time this code is run.

**<span style="color:blue; font-weight:bold;">`print('Train set:', X_train.shape, y_train.shape)`</span>**:
- <span style="color:green;">This line prints the shape (dimensions) of the training set features (<strong>X_train</strong>) and labels (<strong>y_train</strong>).</span> It helps to verify that the data has been split correctly and the training set has the expected number of samples.

**<span style="color:blue; font-weight:bold;">`print('Test set:', X_test.shape, y_test.shape)`</span>**:
- <span style="color:green;">This line prints the shape (dimensions) of the testing set features (<strong>X_test</strong>) and labels (<strong>y_test</strong>).</span> It ensures that the testing set has been created correctly and contains the correct number of samples.

**<span style="color:blue; font-weight:bold;">`from sklearn.neighbors import KNeighborsClassifier`</span>**:
- <span style="color:green;">This line imports the <strong>KNeighborsClassifier</strong> class from the <strong>sklearn.neighbors</strong> module.</span> The KNeighborsClassifier is a machine learning model used for classification tasks. It is based on the k-nearest neighbors algorithm, which classifies data points based on the classes of their nearest neighbors in the feature space. This classifier is simple, easy to understand, and often effective for various types of classification problems.

**<span style="color:blue; font-weight:bold;">`k = 5`</span>**:
- <span style="color:green;">This line sets the value of <strong>k</strong> to 5, which specifies the number of nearest neighbors that the <strong>KNeighborsClassifier</strong> will consider when making predictions.</span> In the k-nearest neighbors algorithm, the value of <strong>k</strong> is a critical hyperparameter that determines how many neighbors influence the classification of a given data point.

**<span style="color:blue; font-weight:bold;">`neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)`</span>**:
- <span style="color:green;">This line creates an instance of the <strong>KNeighborsClassifier</strong> with <strong>k</strong> set to 5 and trains the model using the training data <strong>X_train</strong> and <strong>y_train</strong>.</span> The <strong>.fit()</strong> method fits the classifier to the training data, allowing it to learn from the features and corresponding labels.

**<span style="color:blue; font-weight:bold;">`neigh`</span>**:
- <span style="color:green;">This line displays the trained <strong>KNeighborsClassifier</strong> model stored in the variable <strong>neigh</strong>.</span> This is useful for confirming that the model has been successfully created and trained.
**<span style="color:blue; font-weight:bold;">`yhat = neigh.predict(X_test)`</span>**:
- <span style="color:green;">This line uses the trained <strong>KNeighborsClassifier</strong> model stored in <strong>neigh</strong> to make predictions on the test set <strong>X_test</strong>.</span> The <strong>.predict()</strong> method applies the model to the test data to predict the class labels. The resulting <strong>yhat</strong> variable contains these predicted labels.

**<span style="color:blue; font-weight:bold;">`yhat[0:5]`</span>**:
- <span style="color:green;">This line retrieves the first 5 predicted labels from the <strong>yhat</strong> array.</span> It allows you to inspect the initial predictions made by the model and compare them with the actual labels to evaluate performance.
**<span style="color:blue; font-weight:bold;">`from sklearn import metrics`</span>**:
- <span style="color:green;">This line imports the <strong>metrics</strong> module from <strong>sklearn</strong>.</span> The <strong>metrics</strong> module provides functions to evaluate the performance of machine learning models, including accuracy, precision, recall, and more.

**<span style="color:blue; font-weight:bold;">`print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))`</span>**:
- <span style="color:green;">This line calculates and prints the accuracy of the <strong>KNeighborsClassifier</strong> model on the training set.</span> The <strong>accuracy_score</strong> function compares the true labels <strong>y_train</strong> with the predicted labels from the model on the training data <strong>neigh.predict(X_train)</strong>. This metric shows how well the model performs on the data it was trained on.

**<span style="color:blue; font-weight:bold;">`print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))`</span>**:
- <span style="color:green;">This line calculates and prints the accuracy of the model on the test set.</span> The <strong>accuracy_score</strong> function compares the true labels <strong>y_test</strong> with the predicted labels <strong>yhat</strong> obtained from the model. This metric shows how well the model generalizes to new, unseen data.
**<span style="color:blue; font-weight:bold;">`Ks = 10`</span>**:
- <span style="color:green;">This line sets the maximum number of neighbors <strong>Ks</strong> to 10.</span> The variable <strong>Ks</strong> determines the range of <strong>k</strong> values to test in the loop to find the optimal number of neighbors for the k-nearest neighbors algorithm.

**<span style="color:blue; font-weight:bold;">`mean_acc = np.zeros((Ks-1))`</span>**:
- <span style="color:green;">This line initializes an array <strong>mean_acc</strong> with zeros to store the mean accuracy scores for each <strong>k</strong> value.</span> The size of this array is <strong>Ks-1</strong> because we are testing values of <strong>k</strong> from 1 to <strong>Ks-1</strong>.

**<span style="color:blue; font-weight:bold;">`std_acc = np.zeros((Ks-1))`</span>**:
- <span style="color:green;">This line initializes an array <strong>std_acc</strong> with zeros to store the standard deviation of accuracy scores for each <strong>k</strong> value.</span> This helps in understanding the variability of accuracy for each <strong>k</strong> value.

**<span style="color:blue; font-weight:bold;">`for n in range(1, Ks):`</span>**:
- <span style="color:green;">This line starts a loop to iterate over <strong>k</strong> values from 1 to <strong>Ks-1</strong>.</span> Each iteration evaluates a different number of neighbors for the k-nearest neighbors classifier.

**<span style="color:blue; font-weight:bold;">`neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train, y_train)`</span>**:
- <span style="color:green;">This line creates a <strong>KNeighborsClassifier</strong> instance with <strong>n</strong> neighbors and trains it using the training data <strong>X_train</strong> and <strong>y_train</strong>.</span> The model is fitted to the training data for each <strong>k</strong> value.

**<span style="color:blue; font-weight:bold;">`yhat = neigh.predict(X_test)`</span>**:
- <span style="color:green;">This line uses the trained model to predict the labels for the test set <strong>X_test</strong>.</span> The predictions are stored in <strong>yhat</strong>.

**<span style="color:blue; font-weight:bold;">`mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)`</span>**:
- <span style="color:green;">This line calculates the accuracy score of the model on the test set and stores it in the <strong>mean_acc</strong> array for the current <strong>k</strong> value.</span> It helps in tracking how the accuracy changes with different values of <strong>k</strong>.

**<span style="color:blue; font-weight:bold;">`std_acc[n-1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])`</span>**:
- <span style="color:green;">This line calculates the standard deviation of the accuracy scores and stores it in the <strong>std_acc</strong> array for the current <strong>k</strong> value.</span> The standard deviation is normalized by the square root of the number of predictions to provide a measure of accuracy variability.
  
**<span style="color:blue; font-weight:bold;">`mean_acc`</span>**:
- <span style="color:green;">This line outputs the <strong>mean_acc</strong> array, which contains the mean accuracy scores for each <strong>k</strong> value tested.</span> It provides an overview of how well different <strong>k</strong> values perform.

**<span style="color:blue; font-weight:bold;">`plt.plot(range(1, Ks), mean_acc, 'g')`</span>**:
- <span style="color:green;">This line plots the mean accuracy scores (<strong>mean_acc</strong>) as a green line.</span> The x-axis represents the range of <strong>k</strong> values from 1 to <strong>Ks-1</strong>, and the y-axis represents the accuracy. The green line shows how the accuracy varies with different numbers of neighbors.

**<span style="color:blue; font-weight:bold;">`plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)`</span>**:
- <span style="color:green;">This line fills the area between the mean accuracy ± 1 standard deviation with a translucent shading.</span> This shading represents the range within one standard deviation of the mean accuracy, giving a visual sense of the variability.

**<span style="color:blue; font-weight:bold;">`plt.fill_between(range(1, Ks), mean_acc - 3 * std_acc, mean_acc + 3 * std_acc, alpha=0.10, color="green")`</span>**:
- <span style="color:green;">This line fills the area between the mean accuracy ± 3 standard deviations with a green translucent shading.</span> This shading represents the range within three standard deviations of the mean accuracy, highlighting the broader variability range.

**<span style="color:blue; font-weight:bold;">`plt.legend(('Accuracy ', '+/- 1xstd', '+/- 3xstd'))`</span>**:
- <span style="color:green;">This line adds a legend to the plot with labels for the accuracy, ±1 standard deviation, and ±3 standard deviations.</span> The legend helps identify what each shaded region and line represents.

**<span style="color:blue; font-weight:bold;">`plt.ylabel('Accuracy ')`</span>**:
- <span style="color:green;">This line sets the label for the y-axis to 'Accuracy'.</span> This label indicates that the y-axis represents the accuracy of the model.

**<span style="color:blue; font-weight:bold;">`plt.xlabel('Number of Neighbors (K)')`</span>**:
- <span style="color:green;">This line sets the label for the x-axis to 'Number of Neighbors (K)'.</span> This label indicates that the x-axis represents the number of neighbors used in the k-nearest neighbors algorithm.

**<span style="color:blue; font-weight:bold;">`plt.tight_layout()`</span>**:
- <span style="color:green;">This line adjusts the layout of the plot to ensure that everything fits within the figure area without overlapping.</span> It helps in making sure that the plot elements are well-organized and readable.

**<span style="color:blue; font-weight:bold;">`plt.show()`</span>**:
- <span style="color:green;">This line displays the plot.</span> It renders the plot in the output cell of the Jupyter Notebook, allowing you to visualize the accuracy versus the number of neighbors and the associated variability.

**<span style="color:blue; font-weight:bold;">`print("The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)`</span>**:
- <span style="color:green;">This line prints the best accuracy score and the corresponding value of <strong>k</strong> that achieved this accuracy.</span> 

**<span style="color:blue; font-weight:bold;">`mean_acc.max()`</span>**:
- <span style="color:green;">This part returns the maximum value in the <strong>mean_acc</strong> array, which represents the highest accuracy achieved.</span>

**<span style="color:blue; font-weight:bold;">`mean_acc.argmax()+1`</span>**:
- <span style="color:green;">This part finds the index of the maximum value in the <strong>mean_acc</strong> array and adds 1 to it to get the corresponding <strong>k</strong> value (since indexing starts from 0).</span>
**<span style="color:blue; font-weight:bold;">`from sklearn.neighbors import KNeighborsRegressor`</span>**:
- <span style="color:green;">This line imports the <strong>KNeighborsRegressor</strong> class from the <strong>sklearn.neighbors</strong> module.</span> This class is used for regression tasks with the k-nearest neighbors algorithm.

**<span style="color:blue; font-weight:bold;">`r2 = neigh.score(X_test, y_test)`</span>**:
- <span style="color:green;">This line calculates the R-squared score of the model on the test set.</span> It evaluates how well the model's predictions match the actual values in <strong>y_test</strong>. Note that this is more relevant for regression tasks. For classification tasks, accuracy is usually reported instead.

**<span style="color:blue; font-weight:bold;">`print("R-squared:", r2)`</span>**:
- <span style="color:green;">This line prints the R-squared score.</span> The R-squared score indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
<span style="color:red; font-weight:bold;">### Summary</span>

In this analysis, we have worked with a heart disease dataset to build a k-nearest neighbors (KNN) model for classification. Here’s a summary of the steps and findings:

1. **<span style="color:blue; font-weight:bold;">Data Loading and Initial Inspection</span>**:
   - We loaded the dataset using `pd.read_csv` and displayed the first few rows to understand its structure.

2. **<span style="color:blue; font-weight:bold;">Data Cleaning</span>**:
   - We handled missing values by removing any rows with `dropna`, ensuring that the dataset is complete and ready for analysis.

3. **<span style="color:blue; font-weight:bold;">Data Description</span>**:
   - We used `describe()` to get a statistical summary of the cleaned dataset, providing insights into the distribution and range of values.

4. **<span style="color:blue; font-weight:bold;">Feature Encoding</span>**:
   - We applied label encoding for binary categorical features and one-hot encoding for categorical features with more than two categories. This transformed categorical data into numerical values suitable for modeling.

5. **<span style="color:blue; font-weight:bold;">Feature Scaling</span>**:
   - We standardized the feature matrix using `StandardScaler` to ensure all features contribute equally to the model.

6. **<span style="color:blue; font-weight:bold;">Model Training and Evaluation</span>**:
   - We split the data into training and testing sets using `train_test_split`.
   - We trained a k-nearest neighbors classifier and evaluated its performance. We found the best accuracy by varying the number of neighbors and plotted the results.

7. **<span style="color:blue; font-weight:bold;">Accuracy and R-squared</span>**:
   - The accuracy of the model on the test set was assessed. We also calculated the R-squared score for the regression model to understand the variance explained by the model.

<span style="color:red; font-weight:bold;">### Conclusion</span>

Based on the analysis:

- **<span style="color:yellow; font-weight:bold;">Model Performance</span>**: The k-nearest neighbors classifier achieved varying accuracy depending on the number of neighbors. We identified the optimal number of neighbors for the best performance.
  
- **<span style="color:yellow; font-weight:bold;">Feature Importance</span>**: Proper encoding and scaling of features were crucial for achieving accurate predictions.

- **<span style="color:yellow; font-weight:bold;">Future Work</span>**: Further improvements could involve exploring other classifiers, tuning hyperparameters, and incorporating additional features for better predictions.

This comprehensive approach ensures a well-rounded evaluation of the k-nearest neighbors model and provides a foundation for further enhancements in predicting heart disease risk.

<span style="color:red; font-weight:bold;">### خلاصه</span>

در این تحلیل، با استفاده از دیتاست بیماری قلبی، مدل k-nearest neighbors (KNN) برای طبقه‌بندی ساخته‌ایم. در اینجا خلاصه‌ای از مراحل و یافته‌ها آورده شده است:

1. **<span style="color:blue; font-weight:bold;">بارگذاری و بررسی اولیه داده‌ها</span>**:
   - با استفاده از `pd.read_csv` دیتاست را بارگذاری کردیم و اولین چند ردیف را نمایش دادیم تا ساختار آن را درک کنیم.

2. **<span style="color:blue; font-weight:bold;">پاک‌سازی داده‌ها</span>**:
   - با استفاده از `dropna` به حذف ردیف‌های دارای مقادیر گمشده پرداختیم و اطمینان حاصل کردیم که دیتاست کامل و آماده برای تحلیل است.

3. **<span style="color:blue; font-weight:bold;">توصیف داده‌ها</span>**:
   - با استفاده از `describe()` خلاصه آماری دیتاست پاک‌سازی شده را به‌دست آوردیم و بینش‌هایی از توزیع و دامنه مقادیر ارائه دادیم.

4. **<span style="color:blue; font-weight:bold;">کدگذاری ویژگی‌ها</span>**:
   - برای ویژگی‌های دسته‌ای باینری از کدگذاری لیبل و برای ویژگی‌های دسته‌ای با بیش از دو دسته از کدگذاری one-hot استفاده کردیم. این امر داده‌های دسته‌ای را به مقادیر عددی مناسب برای مدل‌سازی تبدیل کرد.

5. **<span style="color:blue; font-weight:bold;">مقیاس‌بندی ویژگی‌ها</span>**:
   - با استفاده از `StandardScaler` ماتریس ویژگی‌ها را استاندارد کردیم تا اطمینان حاصل شود که همه ویژگی‌ها به‌طور مساوی به مدل کمک می‌کنند.

6. **<span style="color:blue; font-weight:bold;">آموزش و ارزیابی مدل</span>**:
   - داده‌ها را به مجموعه‌های آموزشی و آزمایشی با استفاده از `train_test_split` تقسیم کردیم.
   - مدل طبقه‌بندی k-nearest neighbors را آموزش دادیم و عملکرد آن را ارزیابی کردیم. با تغییر تعداد همسایه‌ها بهترین دقت را شناسایی کرده و نتایج را ترسیم کردیم.

7. **<span style="color:blue; font-weight:bold;">دقت و R-squared</span>**:
   - دقت مدل بر روی مجموعه تست ارزیابی شد. همچنین امتیاز R-squared را برای مدل رگرسیون محاسبه کردیم تا میزان واریانس توضیح داده شده توسط مدل را درک کنیم.

<span style="color:red; font-weight:bold;">### نتیجه‌گیری</span>

بر اساس تحلیل انجام شده:

- **<span style="color:yellow; font-weight:bold;">عملکرد مدل</span>**: مدل k-nearest neighbors با توجه به تعداد همسایه‌ها دقت‌های متفاوتی را به دست آورد. بهترین عملکرد را با شناسایی تعداد بهینه همسایه‌ها تعیین کردیم.

- **<span style="color:yellow; font-weight:bold;">اهمیت ویژگی‌ها</span>**: کدگذاری و مقیاس‌بندی صحیح ویژگی‌ها برای دستیابی به پیش‌بینی‌های دقیق ضروری بود.

- **<span style="color:yellow; font-weight:bold;">کارهای آینده</span>**: بهبودهای بیشتری می‌تواند شامل بررسی سایر طبقه‌بندها، تنظیم هایپرپارامترها و اضافه کردن ویژگی‌های بیشتر برای پیش‌بینی‌های بهتر باشد.

این رویکرد جامع اطمینان می‌دهد که ارزیابی مدل k-nearest neighbors به‌طور کامل انجام شده و پایه‌ای برای بهبودهای بیشتر در پیش‌بینی ریسک بیماری قلبی فراهم می‌آورد.











