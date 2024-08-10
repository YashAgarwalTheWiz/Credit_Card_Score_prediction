import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model, scalers, and encoders
rf_model = joblib.load('rf_model.pkl')  # Load your trained RandomForest model
scaler = joblib.load('scaler.pkl')      # Load your StandardScaler
credit_score_encoder = joblib.load('credit_score_encoder.pkl')  # Load your LabelEncoder for Credit Score
credit_mix_encoder = joblib.load('credit_mix_encoder.pkl')  # Load your LabelEncoder for Credit Mix
credit_history_age_encoder = joblib.load('credit_history_age_encoder.pkl')  # Load your LabelEncoder for Credit History Age Category
feature_columns = joblib.load('feature_columns.pkl')  # Load your feature columns

# Define loan types
loan_types = [
    'Auto Loan', 'Credit-Builder Loan', 'Debt Consolidation Loan', 'Home Equity Loan',
    'Mortgage Loan', 'Not Specified', 'Payday Loan', 'Personal Loan', 'Student Loan'
]

# Streamlit app
st.title('Credit Score Prediction App')

# User input fields with explanations
st.write("Please fill in the following details:")

monthly_inhand_salary = st.number_input('Monthly Inhand Salary', min_value=0,
                                        help="The amount of money you receive each month after taxes and deductions.")
annual_income = st.number_input('Annual Income', min_value=0.0,
                                help="Your total income for the year before taxes.")
num_bank_accounts = st.number_input('Number of Bank Accounts', min_value=0,
                                    help="The total number of bank accounts you have.")
num_credit_card = st.number_input('Number of Credit Cards', min_value=0,
                                  help="The total number of credit cards issued in your name.")
interest_rate = st.number_input('Interest Rate', min_value=0.0,
                                 help="The percentage of interest charged on your loans or credit balances.")
num_of_loan = st.number_input('Number of Loans', min_value=0,
                              help="The total number of active loans you have.")
type_of_loan = st.multiselect('Type of Loan', options=loan_types,
                              help="The kinds of loans you have. You can select multiple types.")
delay_from_due_date = st.number_input('Delay from Due Date', min_value=0,
                                      help="The number of days you are late in making payments on your loans or credit accounts.")
num_of_delayed_payment = st.number_input('Number of Delayed Payments', min_value=0,
                                         help="The total number of times you have missed or delayed payments.")
changed_credit_limit = st.number_input('Changed Credit Limit', min_value=0.0,
                                       help="The amount by which your credit limit has been increased or decreased.")
num_credit_inquiries = st.number_input('Number of Credit Inquiries', min_value=0,
                                       help="The number of times your credit report has been checked by lenders or other entities.")
credit_mix = st.selectbox('Credit Mix', options=['Good', 'Standard', 'Bad'],
                          help="The variety of credit accounts you have. Options are 'Good', 'Standard', or 'Bad'.")
outstanding_debt = st.number_input('Outstanding Debt', min_value=0.0,
                                   help="The total amount of money you owe on all your credit accounts.")
credit_utilization_ratio = st.number_input('Credit Utilization Ratio', min_value=0.0,
                                           help="The ratio of your current credit card balances to your credit limits.")
credit_history_age_category = st.selectbox('Credit History Age Category', options=[
    '1-2 Years', '2-3 Years', '4-5 Years',
    '5-10 Years', '10-15 Years', '15-20 Years', '20-25 Years',
    '25-30 Years', '30-35 Years', '35+ Years'
], help="The length of your credit history, categorized by years.")
payment_of_min_amount = st.selectbox('Payment of Minimum Amount', options=['Yes', 'No'],
                                     help="Indicates whether you are making at least the minimum required payment on your credit accounts.")
total_emi_per_month = st.number_input('Total EMI per Month', min_value=0.0,
                                      help="The total amount you pay in Equated Monthly Installments (EMIs) for loans each month.")
amount_invested_monthly = st.number_input('Amount Invested Monthly', min_value=0.0,
                                          help="The amount of money you invest each month.")
payment_behaviour = st.selectbox('Payment Behaviour', options=[
    'Low_spent_Small_value_payments', 'High_spent_Medium_value_payments', 'High_spent_Large_value_payments',
    'Low_spent_Medium_value_payments', 'High_spent_Small_value_payments', 'Low_spent_Large_value_payments'
], help="Your payment behavior categorized by how you spend and pay.")
monthly_balance = st.number_input('Monthly Balance', min_value=0.0,
                                  help="The amount of money remaining in your account after all expenses are deducted each month.")

# Prepare the new data for prediction
new_data = {
    'Monthly_Inhand_Salary': monthly_inhand_salary,
    'Annual_Income': annual_income,
    'Num_Bank_Accounts': num_bank_accounts,
    'Num_Credit_Card': num_credit_card,
    'Interest_Rate': interest_rate,
    'Num_of_Loan': num_of_loan,
    'Type_of_Loan': ', '.join(type_of_loan),
    'Delay_from_due_date': delay_from_due_date,
    'Num_of_Delayed_Payment': num_of_delayed_payment,
    'Changed_Credit_Limit': changed_credit_limit,
    'Num_Credit_Inquiries': num_credit_inquiries,
    'Credit_Mix': credit_mix,
    'Outstanding_Debt': outstanding_debt,
    'Credit_Utilization_Ratio': credit_utilization_ratio,
    'Credit_History_Age_Category': credit_history_age_category,
    'Payment_of_Min_Amount': 1 if payment_of_min_amount == 'Yes' else 0,
    'Total_EMI_per_month': total_emi_per_month,
    'Amount_invested_monthly': amount_invested_monthly,
    'Payment_Behaviour': payment_behaviour,
    'Monthly_Balance': monthly_balance
}

# Create a DataFrame from the input data
new_data_df = pd.DataFrame([new_data])

# One-Hot Encode Type_of_Loan
for loan_type in loan_types:
    new_data_df[loan_type] = new_data_df['Type_of_Loan'].apply(lambda x: 1 if loan_type in x.split(', ') else 0)
new_data_df.drop(columns=['Type_of_Loan'], inplace=True)

# Encode Credit_Mix and Credit_History_Age_Category
new_data_df['Credit_Mix'] = credit_mix_encoder.transform(new_data_df['Credit_Mix'])
new_data_df['Credit_History_Age_Category'] = credit_history_age_encoder.transform(new_data_df['Credit_History_Age_Category'])

# Ensure dummy variables for Payment_Behaviour
new_data_df = pd.get_dummies(new_data_df, columns=['Payment_Behaviour'], drop_first=True)

# Ensure the same order of columns as the training data
missing_cols = set(feature_columns) - set(new_data_df.columns)
for col in missing_cols:
    new_data_df[col] = 0
new_data_df = new_data_df[feature_columns]

# Scale the new data point using the same scaler as the training set
new_data_scaled = scaler.transform(new_data_df)

# Predict the credit score using the trained model
predicted_credit_score = rf_model.predict(new_data_scaled)

# Decode the predicted credit score
predicted_credit_score_label = credit_score_encoder.inverse_transform(predicted_credit_score)

# Output the result
if st.button('Predict Credit Score'):
    if new_data_df.isnull().values.any() or new_data_df.empty:
        st.write('Poor')
    else:
        st.write(f'Decoded Predicted Credit Score: {predicted_credit_score_label[0]}')
