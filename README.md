# Classification_on_Imbalanced Data 

## Table of contents
- [Project Overview](#project-overview)
- [Executive Summary](#executive-summary)
- [Goal](goal)
- [Data Structure](data-structure)
- [Tools](tools)
- [Analysis](#analysis)
- [Insights](insights)
- [Recommendations](recommendations)

### Project Overview
The project addresses a common challenge in machine learning: imbalanced datasets, where the number of observations in one class (the majority) significantly outweighs the other (the minority). Specifically, the project uses an Insurance Claims dataset to predict the frequency of claims. In this context, "no claim" (0) is the majority class, while "claim made" (1) is the minority. The project demonstrates a full end-to-end pipeline in Python, including Exploratory Data Analysis (EDA), data preprocessing, handling imbalance via oversampling, and model training using a Random Forest Classifier.

### Executive Summary
Traditional machine learning models often become biased toward the majority class, leading to high overall accuracy but poor predictive performance on the minority class—which is usually the more important class to identify (e.g., fraud detection or insurance claims).

1. The Data: A dataset of 58,592 entries related to insurance policies, including features like vehicle age, customer age, and region density.

2. The Approach: The project utilizes the Random Forest algorithm due to its robustness with categorical and numerical data. To fix the class imbalance, it employs the Oversampling technique to equalize the number of claim and no-claim instances (54,844 entries each).

3. The Outcome: The final model achieved a high performance, with a 99% F1-score and recall. This indicates that the model is exceptionally effective at identifying actual insurance claims without being biased by the majority of non-claimants.

### Goal
The primary objectives of this project are:

1. Technical Implementation: To demonstrate how to handle imbalanced data using Python libraries like pandas, seaborn, and scikit-learn.

2. Resampling Mastery: To apply oversampling strategies to the minority class to ensure the model learns the characteristics of the "claim" class effectively.

3. Feature Importance Identification: To determine which factors (such as subscription length and customer age) are the strongest predictors of whether an insurance claim will be filed.

4. Predictive Accuracy: To build a robust classification model that achieves high Recall and F1-Score, ensuring that the minority class is correctly predicted despite its initial scarcity in the data.

### Data structure and initial checks
[Dataset](https://docs.google.com/spreadsheets/d/1GK4tnY4_YfX8ccNhVtedEoGpsUUGOwzlysEVEQr_xpA/edit?gid=1748548740#gid=1748548740)

 - The initial checks of your transactions.csv dataset reveal the following:

| Features | Description | Data types |
| --------- | ---------- | ---------- |
| policy_id | A unique identifier assigned to each insurance policy. | object  | 
| subscription_length | The duration (in years) the customer has been associated with the insurance company. | float64 | 
|  vehicle_age  | The age of the vehicle being insured in years. | float64 | 
| customer_age | The age of the primary policyholder (ranging from 35 to 75 years in this dataset) | int64 | 
| region_code | A code representing the geographical area where the customer is located (e.g.,C8, C2). | object  | 
|  region_density | The population density of the customer's region. | int64  | 
| segment | The category of the vehicle based on size and price (e.g., A, B2, C1, C2). | object | 
| model | The specific code for the car model (e.g., M1, M4, M9). | object  | 
| fuel_type | The specific code for the car model (e.g., M1, M4, M9). | object  | 
| max_torque | Technical performance metrics of the engine (e.g., 250Nm. | object  | 
| max_power | Technical performance metrics of the engine (2750rpm). | object  | 
| engine_type | The specific engine model or series (e.g., 1.5 L U2 CRDi, F8D Petrol Engine). | object  |
| airbags | Total number of airbags present in the vehicle. | int64  |
| engine_type | The specific engine model or series (e.g., 1.5 L U2 CRDi, F8D Petrol Engine). | object  |
| is_esc | Indicates if the vehicle has Electronic Stability Control. | object  |
| is_adjustable_steering  | Boolean indicators (Yes/No) for features like fog lights, window wipers/washers/defoggers, speed alerts, and power steering. | object  |
| is_tpms | Indicates if the vehicle has a Tire Pressure Monitoring System. | object  |
| is_parking_sensors | Indicates the presence of parking assistance tech. |  object |
| is_parking_camera | Indicates the presence of parking assistance tech. |  object |
| rear_brakes_type |  | object |
| displacement | The engine's swept volume in cubic centimeters (cc). | object |
| cylinder | The number of cylinders in the engine. | int64  |
| transmission_type | Whether the vehicle is Manual or Automatic. | object  |
| steering_type | The type of steering system (Power, Electric, or Manual). | object |
| turning_radius | The minimum space required for the vehicle to make a U-turn. | float64 |
| length  | The physical dimensions (in mm) and weight (in kg) of the vehicle | int64 |
| width | The physical dimensions (in mm) and weight (in kg) of the vehicle | int64 |
| gross_weightint64 | The physical dimensions (in mm) and weight (in kg) of the vehicle | int64 |
| is_front_fog_lights | Boolean indicators (Yes/No) for features like fog lights, window wipers/washers/defoggers, speed alerts, and power steering. | object |
| is_rear_window_wiper | Boolean indicators (Yes/No) for features like fog lights, window wipers/washers/defoggers, speed alerts, and power steering. | object  |
| is_rear_window_washer  | Boolean indicators (Yes/No) for features like fog lights, window wipers/washers/defoggers, speed alerts, and power steering. | object |
| is_rear_window_defogger  | Boolean indicators (Yes/No) for features like fog lights, window wipers/washers/defoggers, speed alerts, and power steering. | object |
| is_brake_assist | Indicates if the car has a system that increases braking force during emergencies. | object  |
| is_power_door_locks | Boolean indicators (Yes/No) for features like fog lights, window wipers/washers/defoggers, speed alerts, and power steering. | object  |
| is_central_locking | Boolean indicators (Yes/No) for features like fog lights, window wipers/washers/defoggers, speed alerts, and power steering. | object |
| is_power_steering | Boolean indicators (Yes/No) for features like fog lights, window wipers/washers/defoggers, speed alerts, and power steering. | object |
| is_driver_seat_height_adjustable | Boolean indicators (Yes/No) for features like fog lights, window wipers/washers/defoggers, speed alerts, and power steering. | object |
| is_day_night_rear_view_mirror | Boolean indicators (Yes/No) for features like fog lights, window wipers/washers/defoggers, speed alerts, and power steering. | object |
| is_ecw | Boolean indicators (Yes/No) for features like fog lights, window wipers/washers/defoggers, speed alerts, and power steering. | object |
| is_speed_alert | Boolean indicators (Yes/No) for features like fog lights, window wipers/washers/defoggers, speed alerts, and power steering. | object |
| ncap_rating | The official safety rating assigned by the New Car Assessment Program (ranges from 0 to 5). | int64 |
| claim_status | This is the label we are trying to predict. 0 indicates no claim was made, and 1 indicates a claim was filed by the customer. | int64 | 

### Tools
- Python: Google Colab - Data Preparation and pre-processing, Exploratory Data Analysis, Descriptive Statistics, Data manipulation,Visualization, Handling Class Imbalance with Oversampling technique, Model developoment, Model training and evaluation
  
### Analysis
Python
Importing all the necessary libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
```
``` python
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
```
Settings to display maximum columns & rows in my dataframe
``` python
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```
Laoding the dataset
```python
df = pd.read_csv("Insurance claims data.csv")
print(df.head())
```
<img width="645" height="538" alt="image" src="https://github.com/user-attachments/assets/57c3f7c1-3ccf-4c4f-9e18-2fa417ac0570" />
<img width="642" height="544" alt="image" src="https://github.com/user-attachments/assets/9789bd3b-53fb-45a5-b302-88b95d4e1604" />
<img width="614" height="267" alt="image" src="https://github.com/user-attachments/assets/1f65ff60-5150-4118-af56-44610ab4c9a2" />

Shape of the data
```python
df.shape
```
information about the dataset
```python
df.info()
```
<img width="525" height="589" alt="image" src="https://github.com/user-attachments/assets/5f92b97c-65cc-4603-810b-ae9c87229430" /><img width="501" height="353" alt="image" src="https://github.com/user-attachments/assets/b4721407-bdde-41d7-9932-bedbe7cf1113" />

Check on missing values
``` python
df.isna().sum()/len(df)
```
we notice that our data doesn't any missing values.

Descriptive statistics
```python
df.describe()
```
<img width="1511" height="274" alt="image" src="https://github.com/user-attachments/assets/298b6713-f826-4c88-ad19-2708e4b71aca" />

**Insights**

- The mean for claim_status is approximately 0.064, indicating that only about 6.4% of the 58,592 customers actually made a claim.
- This confirms the dataset is highly imbalanced, with over 93% of the entries belonging to the "no-claim" category.
- The average customer age is 44.8 years, with the youngest customer being 35 and the oldest being 75.
- Policyholders have a mean subscription length of 6.1 years. Interestingly, the 75th percentile is at 10.4 years, but the maximum is 14 years, suggesting a segment of very long-term customers.
- Most vehicles are relatively new, with a mean age of 1.39 years. However, there is a significant outlier with a maximum vehicle age of 20 years.
- On average, vehicles are equipped with 3.14 airbags, though some entry-level models have as few as 1 and premium models have up to 6.
- The average ncap_rating is 1.76, with a median (50%) of 2.0, indicating that a large portion of the insured vehicles have lower safety ratings.
- There is a massive variance in region_density, ranging from as low as 290 to as high as 73,430. This suggests the data covers both sparse rural areas and highly dense urban centers.
- The average vehicle length is 3,850 mm and width is 1,672 mm, which typically characterizes compact or sub-compact cars.

**Exploratpry Data Analysis**

Now, let's dive into the distribution of our target variable which is 'Claim status'
```python
# plot the distribution of the target variable 'claim_status'
print(df['claim_status'].value_counts())

plt.figure(figsize=(8, 5))
sns.countplot(x='claim_status', data=df)
plt.title('Distribution of Claim Status')
plt.xlabel('Claim Status')
plt.ylabel('Count')
plt.show()
```
<img width="224" height="84" alt="image" src="https://github.com/user-attachments/assets/c47aa864-236d-4eb2-9a92-5b5cb0935fd0" />
<img width="713" height="468" alt="image" src="https://github.com/user-attachments/assets/3253875a-edee-49b2-9eeb-cdc64fbf97dc" />

Numerical Feature analysis
```python
numerical_columns = ['subscription_length', 'vehicle_age', 'customer_age']

# plotting distributions of numerical features
plt.figure(figsize=(15, 8))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(1, 3, i)
    sns.histplot(df[column], bins=30, kde=True,color='blue')
    plt.title(f'Distribution of {column}')

plt.tight_layout()
plt.show()
```
<img width="1490" height="790" alt="image" src="https://github.com/user-attachments/assets/2b4d777a-028c-40f2-a3ae-5db6422e4e38" />

Categorical Feature analysis
``` python
categorical_columns = ['region_code', 'segment', 'fuel_type']

# plotting distributions of categorical features
plt.figure(figsize=(15, 10))
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(3, 1, i)
    sns.countplot(y=column, data=df, order = df[column].value_counts().index,color='orange')
    plt.title(f'Distribution of {column}')
    plt.xlabel('Count')
    plt.ylabel(column)

plt.tight_layout()
plt.show()
```
<img width="1490" height="989" alt="image" src="https://github.com/user-attachments/assets/fc8663e2-a235-4d0d-8dc4-2c8f95a9b5ee" />

As we notice the classes in the target variable are imbalanced. Let's g ahead and fix it.
```python
# separate majority and minority classes
majority = df[df.claim_status == 0]
minority = df[df.claim_status == 1]

# oversample the minority class
minority_oversampled = resample(minority,
                                replace=True,
                                n_samples=len(majority),
                                random_state=42)

# combine majority class with oversampled minority class
oversampled_data = pd.concat([majority, minority_oversampled])

# check the distribution of undersampled and oversampled datasets
oversampled_distribution = oversampled_data.claim_status.value_counts()

oversampled_distribution
```
<img width="245" height="85" alt="image" src="https://github.com/user-attachments/assets/0565ff4f-ba75-4379-a2f6-3e7363fb6232" />

It is important to note that while oversampling was the primary method used in the source article to balance the classes, there are several other sophisticated strategies available to handle imbalanced data.

Alternative Resampling Techniques
Undersampling: This involves reducing the number of samples in the majority class to match the size of the minority class.

SMOTE (Synthetic Minority Over-sampling Technique): Instead of simply replicating existing minority samples, SMOTE creates synthetic "neighboring" examples to broaden the minority class's influence.

Hybrid Approaches (SMOTE + Tomek Links): This method combines oversampling with cleaning; SMOTE generates new samples while Tomek links identify and remove overlapping pairs from the majority class to clarify the decision boundary between classes.

Algorithm-Level Techniques
Rather than modifying the dataset itself, you can adjust how the model treats each sample:

class_weight Parameter: In Random Forest or Logistic Regression, setting this to "balanced" tells the model to automatically penalize mistakes on the minority class more heavily than mistakes on the majority class.

sample_weight Parameter: This allows for granular control by assigning a specific weight to each individual observation during the training phase.

Why These Matter
The goal of these techniques is to prevent the model from becoming biased toward the majority class, ensuring it doesn't ignore the minority class—which is often the primary focus in tasks like insurance claim prediction.

Now, lets for the distribution of some features.
```python
# plotting the distribution of 'customer_age', 'vehicle_age', and 'subscription_length' with respect to 'claim_status'
plt.figure(figsize=(15, 8))

# 'customer_age' distribution
plt.subplot(1, 3, 1)
sns.histplot(data=oversampled_data, x='customer_age', hue='claim_status', element='step', bins=30)
plt.title('Customer Age Distribution')

# 'vehicle_age' distribution
plt.subplot(1, 3, 2)
sns.histplot(data=oversampled_data, x='vehicle_age', hue='claim_status', element='step', bins=30)
plt.title('Vehicle Age Distribution')

# 'subscription_length' distribution
plt.subplot(1, 3, 3)
sns.histplot(data=oversampled_data, x='subscription_length', hue='claim_status', element='step', bins=30)
plt.title('Subscription Length Distribution')

plt.tight_layout()
plt.show()
```
<img width="1489" height="790" alt="image" src="https://github.com/user-attachments/assets/5bbb778b-7d19-4bf5-8dd7-60c5bd0605cc" />

To address the data imbalance, we will first apply Label Encoding to transform categorical features into a numerical format suitable for machine learning. We will then implement a Random Forest Classifier, a supervised learning model chosen for its inherent ability to handle complex interactions and provide a baseline for managing imbalanced classes. This model will also allow us to perform Feature Importance analysis to identify which specific variables—such as subscription length or customer age—exert the most influence on the likelihood of an insurance claim.

```python
# encode categorical variables
le = LabelEncoder()
encoded_data = df.apply(lambda col: le.fit_transform(col) if col.dtype == 'object' else col)

X = encoded_data.drop('claim_status', axis=1)
y = encoded_data['claim_status']

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)

feature_importance = rf_model.feature_importances_
features_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
features_df = features_df.sort_values(by='Importance', ascending=False)

print(features_df.head(10))
```
<img width="303" height="234" alt="image" src="https://github.com/user-attachments/assets/2bb039b9-2eca-4510-ab75-805521de1e8f" />

Let's train our model.
```python
# Model Training

oversampled_data = oversampled_data.drop('policy_id', axis=1)

# prepare the oversampled data
X_oversampled = oversampled_data.drop('claim_status', axis=1)
y_oversampled = oversampled_data['claim_status']

# encoding categorical columns
X_oversampled_encoded = X_oversampled.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'object' else col)

# splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_oversampled_encoded, y_oversampled, test_size=0.3, random_state=42)

# Random Forest model
rf_model_oversampled = RandomForestClassifier(random_state=42)
rf_model_oversampled.fit(X_train, y_train)

# predictions
y_pred = rf_model_oversampled.predict(X_test)

print(classification_report(y_test, y_pred))
```
<img width="434" height="175" alt="image" src="https://github.com/user-attachments/assets/8dec5e6c-ff23-4cd0-bd4b-0be7462cb6ab" />

Let's check for the predictions versus the actual data to confirm how well our model is perfroming by looking at our dataset and the distribution of classified & misclassified labels.
``` python
original_encoded = df.drop('policy_id', axis=1).copy()
encoders = {col: LabelEncoder().fit(X_oversampled[col]) for col in X_oversampled.select_dtypes(include=['object']).columns}

for col in original_encoded.select_dtypes(include=['object']).columns:
    if col in encoders:
        original_encoded[col] = encoders[col].transform(original_encoded[col])

original_encoded_predictions = rf_model_oversampled.predict(original_encoded.drop('claim_status', axis=1))

comparison_df = pd.DataFrame({
    'Actual': original_encoded['claim_status'],
    'Predicted': original_encoded_predictions
})

print(comparison_df.head(10))
```
<img width="183" height="233" alt="image" src="https://github.com/user-attachments/assets/755aa272-9efc-43ed-bd10-f8bca75ae6c6" />

```pyhton 
correctly_classified = (comparison_df['Actual'] == comparison_df['Predicted']).sum()
incorrectly_classified = (comparison_df['Actual'] != comparison_df['Predicted']).sum()

classification_counts = [correctly_classified, incorrectly_classified]
labels = ['Correctly Classified', 'Misclassified']

# create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(classification_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=["#61EC65", "#F00808"])
plt.title('Classification Accuracy')
plt.show()
```
<img width="660" height="656" alt="image" src="https://github.com/user-attachments/assets/6a070e8f-70b6-4699-8775-aa394633d0a1" />

Hence, we can conclude that our model using the oversampling technique with a random forest model works fine.

### Insights

1. Extreme Class Imbalance: Out of 58,592 customers, only approximately 6.4% have filed a claim. This disparity means a standard model would likely "ignore" the claim class to achieve high overall accuracy, failing the business's primary goal of identifying risky policies.

2. Predictive Power of Policy Lifecycle: subscription_length and customer_age emerged as top features. This suggests that the duration of the relationship with the insurer and the life stage of the customer are stronger indicators of claim probability than specific vehicle technical specs like engine_type.

3. Safety vs. Risk: Features like ncap_rating and airbags show that while safety is recorded, many vehicles in the dataset have lower safety ratings (average NCAP ~1.76). This indicates a potential segment of high-risk vehicles that the model must differentiate.

4. Geographic Density Impact: The wide range in region_density (290 to 73,430) indicates that urban vs. rural settings significantly influence claim likelihood, likely due to traffic volume and accident frequency in denser areas.

### Recommendations

**Data-Level Recommendations**

a. Move Beyond Random Oversampling: Your notebook uses basic oversampling (duplicating rows). While effective initially, this can lead to overfitting. Use SMOTE (Synthetic Minority Over-sampling Technique) to create "synthetic" claim examples rather than exact copies.

b. Feature Engineering: * Recommendation: Create a "Risk Index" feature that combines vehicle_age, ncap_rating, and region_density. This can help the Random Forest model see the interaction between these variables more clearly.

**Model & Evaluation Recommendations**

a. Change the Success Metric: Avoid using "Accuracy" as your primary KPI. A model could be 93% accurate by never predicting a single claim. Focus on Recall (to ensure you don't miss actual claims) and F1-Score. A high Recall is vital for an insurance company to prepare for payouts.

b. Threshold Tuning: The default classification threshold is 0.5. Experiment with lowering the threshold (e.g., to 0.3) to catch more potential claims, even if it increases false positives slightly.

**Algorithm-Level Recommendations**

a. Implement Class Weights: Instead of changing the data, use the class_weight='balanced' parameter inside your RandomForestClassifier. This tells the model to treat the 6% of claim cases as significantly more important than the 94% of non-claim cases during the learning process.

b. Clean Decision Boundaries: * Recommendation: Use Tomek Links after oversampling to remove overlapping points between classes. This "cleans" the border between claimants and non-claimants, making the model's decision-making more precise.

