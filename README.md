# PDAN8411# ReadMe  

# PDAN8411. POE

# Task 1 Description:    
Predict video game ratings using a linear regression model. Utilizing a dataset with features like launch year, 
genre, voter count, and directors, this app trains a model to forecast game ratings. Features are extracted, 
overfitting and underfitting are addressed, and predictions are made. Visualization tools aid in understanding 
data relationships. Libraries like Pandas, Scikit-learn, Matplotlib, and Seaborn are employed. Cross-validation 
ensures robustness.  

# Task 2 Description:    
Predict stroke occurrences using Naive Bayes and K-Nearest Neighbors classifiers. Utilizing a healthcare dataset with features such as age, 
hypertension, heart disease, and lifestyle factors, this project trains models to forecast stroke likelihood. Data preprocessing steps include 
handling missing values and encoding categorical variables. The models are evaluated using accuracy, confusion matrices, and classification reports. 
Visualization tools aid in comparing model performance and understanding the impact of different training/test splits. Libraries such as Pandas, 
Scikit-learn, Matplotlib, and Seaborn are employed to ensure comprehensive analysis and robust predictions.

# Task 3 Description:  
Classify text messages as spam or ham using a machine learning pipeline with text preprocessing, TF-IDF vectorization, and a Multinomial Naive Bayes classifier. 
The project includes downloading necessary NLTK data, loading a labeled dataset, and performing preprocessing steps such as stemming and removing stopwords. 
A pipeline is created for transforming the data and training the model, with hyperparameter tuning performed using GridSearchCV. 
The model's performance is evaluated using classification reports, confusion matrices, and accuracy scores. Visualizations, including a confusion matrix heatmap, 
help to illustrate the results. Libraries like Pandas, NLTK, Scikit-learn, Matplotlib, and Seaborn are used throughout the process to ensure effective text classification.

# How to Install and Run the Project:    
Visual Studio Code Project ~  
Step 1  
* Ensure that your computer meets the system requirements to run Visual Studio Code.  
* Check for available disk space to accommodate the application.  
* Verify the necessary system requirements for your operating system.  

Step 2  
* Navigate to the official Visual Studio Code website at https://code.visualstudio.com/.  
* Download the latest version of Visual Studio Code suitable for your operating system (Windows, macOS, or Linux).  

Step 3  
* Review and agree to the terms and conditions presented during the installation process.  

Step 4  
* Once installed, launch Visual Studio Code from your applications menu or desktop shortcut.  
* Familiarize yourself with the user interface and available features.  

Step 5  
* To start coding, create a new file or open an existing project folder by selecting "File" > "Open Folder" rom the menu.  
* Begin coding and exploring the various functionalities offered by Visual Studio Code.  

Step 6  
* For downloading a project code from a repository, navigate to the GitHub repository link provided by the project team.  
* Clone the repository using the Git extension integrated into Visual Studio Code or download the code as a zip file.  

Step 7  
* Once downloaded, open Visual Studio Code, click on "File" > "Open Folder," and navigate to the folder containing the project code.  
* You are now ready to use Visual Studio Code for your project development.  

# Task 1
## Key Features:  
## Model Training and Prediction:  
• Train a linear regression model to predict video game ratings based on features like launch year, genre, votes, and directors.  
• Evaluate the model's performance using metrics such as Mean Absolute Error, Mean Squared Error, and Root Mean Squared Error.  
• Implement K-fold cross-validation to ensure the model's robustness and generalizability.  

## Data Visualization:  
• Explore relationships between variables through scatter plots and regression graphs.  
• Visualize the distribution of numerical variables like launch year and voter count using histograms.  

## Data Preprocessing:  
• Handle missing values by dropping rows with NaN values in the 'year' column.  
• Convert categorical variables like genre and directors into dummy variables for model compatibility.  

## User-friendly Interface:  
• Simple and intuitive design for ease of use.  
• Interactive visualizations to aid in understanding data relationships.  

## Why Use This App?
• Predictive Analytics: Forecast video game ratings based on historical data, aiding in decision-making processes.  
• Educational Resource: Gain insights into the factors influencing video game ratings and learn about linear regression modeling.  
• Enhanced Data Understanding: Explore data patterns visually and understand the relationships between variables.  

# Error features:     
- No issues that I could find.

# Task 2
## Key Features:  
## Data Preprocessing: 
• Handle missing values by filling NaN values in the 'bmi' column with the mean of the column.
• Encode categorical variables ('gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status') using LabelEncoder for model compatibility.

## Model Training and Prediction:
• Train a Naive Bayes classifier to predict the likelihood of stroke based on features such as age, hypertension, heart disease, etc.
• Train a K-Nearest Neighbors (KNN) classifier for the same prediction task to compare performance.
Evaluate both models using accuracy, confusion matrix, and classification report metrics.

## Model Evaluation:
• Print the accuracy score for both Naive Bayes and K-Nearest Neighbors classifiers.
• Display confusion matrices and detailed classification reports for both models to understand their performance.

## Data Visualization:
• Visualize confusion matrices for both models using heatmaps to better understand the classification results.
• Compare the accuracy of both models using a bar chart to determine which model performs better.

## Impact of Training/Test Splits:
• Investigate the effect of different training/test split ratios (0.1, 0.2, 0.3, 0.4, 0.5) on the accuracy of both Naive Bayes and K-Nearest Neighbors classifiers.
• Plot accuracy vs. test size ratio to visualize the impact of different training/test splits on model performance.

## Data Visualization:
• Plot confusion matrices for both Naive Bayes and KNN models using heatmaps.
• Create a bar chart to compare the accuracy of Naive Bayes and KNN classifiers.
• Plot accuracy against different test size ratios to observe the impact on model performance.

## Why Use This App?
• Predictive Analytics: Predict the likelihood of a stroke based on various health-related features, aiding in early detection and prevention strategies.
• Comparative Analysis: Compare the performance of two different machine learning models (Naive Bayes and KNN) on the same dataset.
• Data Exploration: Visualize the impact of different training/test splits and understand the importance of data preprocessing steps like handling missing values and encoding categorical variables.

# Error features:     
- No issues that I could find.

# Task 3
## Key Features:  
### Text Classification and Prediction:
• Train a text classification model to identify spam messages using features like TF-IDF vectorization and Naive Bayes classification.
• Evaluate the model's performance using metrics such as Classification Report, Confusion Matrix, and Accuracy Score.
• Implement GridSearchCV for hyperparameter tuning to optimize the model's accuracy.

### Data Visualization:
• Explore text data through visualizations like confusion matrices.
• Visualize the distribution of predicted labels using heatmaps.

### Data Preprocessing:
• Handle missing values by dropping columns with NaN values.
• Preprocess text data by tokenizing, removing stopwords, stemming, and lemmatizing.
• Apply advanced tokenization to enhance text processing.

### Text Processing Techniques:
• Apply the term frequency–inverse document frequency (TF-IDF) method to text data for feature extraction.
• Apply pipelines in Python to chain multiple steps for efficient text processing and classification.

### User-friendly Interface:
• Simple and intuitive design for ease of use.
• Clear and detailed visualization of model performance metrics.

### Why Use This App?
• Spam Detection: Identify and filter out spam messages, improving communication efficiency.
• Educational Resource: Gain insights into text preprocessing, classification, and model evaluation techniques.
• Enhanced Data Understanding: Explore patterns in text data and understand the effectiveness of spam detection models.

### Error features:
I had a very hard time implementing 'Apply Latent Dirichlet Allocation to text data'. I had to taken it out because it was doing weird things. Apologies for that.
