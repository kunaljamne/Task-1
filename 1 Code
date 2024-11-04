# CodeTech-IT-Credit-card-Fraud-Detection-Project-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, accuracy_score

credit_card_data = pd.read_csv('/content/drive/MyDrive/Credit card fraud detection/creditcard.csv.zip')

#First 5 rows of the Dataset
credit_card_data.head()

credit_card_data.tail()

#Data Informations
credit_card_data.info()
#Checking the Number of Missing Values in each Column
credit_card_data.isnull().sum()

#In our Data is no any missing value but In other Data has Missing values then How we Takle  this values we need to do some more processing to Convert some missing values into Meaning full number
#Distribution of legit transaction & Fraudulent transactios -- value count() is a function that give this details
#in NOrmal Transaction thera are 284315 transaction(Data POints) that is show by (0). and 492 Transaction only for fraaud Transaction that is by (1).
#thats why this Data set is very Unbalanced. in that case we 2 target variables or 2 classes(0 and 1).
# MORE THAN 90 PERCENT OF DATA is only 1 perticular class(0)
credit_card_data['Class'].value_counts()

#So we can  not use this data to train our machine learning model beacause. if we train machine learning model of use this data.
#than it cannot recognized the Fraudulent Transaction beacause we have very less Data points(492) in particular case.
#that is why the process is comes to play and this is Unbalaced data set Sobefore that let do some more analysis of data set.

#Separating the data For Analysis
#This is Pandas series data type
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)

#get Statistica Measures of the data .
legit.Amount.describe ()

fraud.Amount.describe()
#Compare values for both Transactions
credit_card_data.groupby('Class').mean()

#The number of Frauduent Transaction ----> 492.
#we have to equal amount of data so we do Normal Transaction data points isreduced or select on 492 transaction
#in 284315 data points in Normal Transaction and going to randomly take 492 transaction in Normal Transaction and then we Joint the data Normal and Fraudulent
#once we do that so we have 492 is Normal Transaction and 492 is fraudulent transaction. and then it is very good dataet in that case because
#the data is NOrmally and Uniformly Distrbution of Both.if we do that the distribution is Even then we can better prediction in machine learning .

legit_sampling = legit.sample(n=492)

#Now we have to Concatanate of 2 Data Frames.
#if we added Axis = 0 then the data frames is added one by one .
#so we have a legitsample if we giive axis = 0 all the 492 values will be added below the legit sample
#if you mention axis = 1 then the Values is Added Column wise which we dont want so we want values added row wise so I mention axis to 0;
#Axis = 0 means ROWS ;
#Axis = 1 means COLUMNS ;
new_dataset = pd.concat([legit_sampling, fraud], axis=0)

new_dataset.head(5)
#so we can see the random values So this is Serial number is Above is 0,1...
#and now Samples are Picked randomly so this is are legit samples and we can also last 5 data points .
#in the last the class column is all the entries is 0 so it is Normal transactons data points.

#last 5 data points
new_dataset.tail()

#Now in this is last column on 1 so this is Fraudulent data points .
new_dataset['Class'].value_counts()
#this is Show 0 --- 492 Normal Transaction
#and 1 ----  492 Fraudulent Transaction..

new_dataset.groupby('Class').mean()

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)
# Train the model
model.fit(X_train, Y_train)

#Model Evaluation

#Accuracy Score

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

#Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test Data : ',test_data_accuracy)

# Make predictions
Y_pred = model.predict(X_test)

# Function to detect transaction
def detect_transaction(transaction):
    transaction_scaled = scaler.transform([transaction])  # Scale the transaction using the scaler
    prediction = model.predict(transaction_scaled)  # Use the model to predict
    return "\n  This is NORMAL TRANSACTION . " if prediction[0] == 0  else "\n  This is  FRADULENT TRANSACTION. "
    print("\n\n")


def user_input_transaction():
    print("Now we check if the transaction is Normal or Fraudulent for the given mean values of V1, V2, V3...V30:")

    features = []

    for i in range(30):  # Assuming 30 features
        while True:
            try:
                value = float(input(f"Feature {i+1}: "))
                features.append(value)
                break  # Exit the loop if input is valid
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
            except KeyboardInterrupt:
                print("\nInput interrupted. Exiting.")
                return  # Exit the function if interrupted

    # Assuming detect_transaction is defined elsewhere and returns a result
    result = detect_transaction(features)
    print(result)

# Call the user input function
user_input_transaction()


#output of the code is predict the Transaction is normal and Fradulent.

]
def user_input_transaction():
    print("Now we check if the transaction is Normal or Fraudulent for the given mean values of V1, V2, V3...V30:")

    features = []

    for i in range(30):  # Assuming 30 features
        while True:
            try:
                value = float(input(f"Feature {i+1}: "))
                features.append(value)

Now we check if the transaction is Normal or Fraudulent for the given mean values of V1, V2, V3...V30:
Feature 1: 1
Feature 2: .08
Feature 3: 0.5
Feature 4: 0.4
Feature 5: 0.2
Feature 6: 0.3
Feature 7: -32
Feature 8: -3
Feature 9: -2
Feature 10: 0.6
Feature 11: 0.7
Feature 12: 0.8
Feature 13: 0.9
Feature 14: 1.1
Feature 15: 1.2
Feature 16: 1.4
Feature 17: 1,
Invalid input. Please enter a numeric value.
Feature 17: 1.4
Feature 18: 1.5
Feature 19: -2
Feature 20: -3.44
Feature 21: -3.5
Feature 22: 0.6
Feature 23: -.4
Feature 24: -.3
Feature 25: -3.5
Feature 26: -.2.2
Invalid input. Please enter a numeric value.
Feature 26: -2.2
Feature 27: -3
Feature 28: 0.8
Feature 29: 0.7
Feature 30: 0.54

  This is  FRADULENT TRANSACTION.
