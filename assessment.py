# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:52:12 2019

@author: pooja_ranawade
"""

"""
A) Assemble a dataframe with one row per customer and the following columns:
    * customer_id
    * gender
    * most_recent_order_date
    * order_count (number of orders placed by this customer)
   Sort the dataframe by customer_id ascending and display the first 10 rows.

B) Plot the count of orders per week.

C) Compute the mean order value for gender 0 and for gender 1. 
    Do you think the difference is significant?

D) Assuming a single gender prediction was made for each customer, 
    generate a confusion matrix for predicted gender. 
    Do not use a library function to do this. 
    What does the confusion matrix tell you about the quality of the predictions?

E) Describe one of your favorite tools or techniques and give a small example of 
    how it's helped you solve a problem. Limit your answer to one paragraph
"""
# reading data
import pandas as pd

orders_data = pd.read_csv('screening_exercise_orders_v201810.csv')

orders_data['date'] = pd.to_datetime(orders_data['date'])
orders_data.info()
orders_data.dtypes

# A
all_customer_id = sorted(list(orders_data['customer_id'].unique()))
ans_A = pd.DataFrame(
    columns=['customer_id', 'gender', 'most_recent_order_date', 'order_count'])
for cid in all_customer_id:
    current = orders_data.loc[orders_data['customer_id'] == cid,
                              'customer_id'].agg(['count']).iloc[0]
    most_recent_date = orders_data.loc[orders_data['customer_id'] == cid,
                                       'date'].agg(['max']).iloc[0]

    gender = orders_data.loc[orders_data[
        'customer_id'] == cid].iloc[0]['gender']

    ans_A.loc[len(ans_A)] = [cid, gender, most_recent_date, current]

with open('ans_A.txt', 'w') as f:
    print(ans_A.head(10), file=f)
ans_A.head(10)


# B
import seaborn.apionly as sns
%matplotlib inline
import matplotlib.pyplot as plt

ans_B = orders_data['date'].groupby(
    orders_data.date.dt.to_period("W")).agg('count')

fig = ans_B.plot(kind='bar', figsize=(20, 10))
fig = fig.get_figure()
fig.savefig("ans_B.png")

# C
total_value = orders_data['value'].agg(['mean']).iloc[0]
mean_value_0 = orders_data.loc[orders_data['gender'] == 0,
                               'value'].agg(['mean']).iloc[0]
mean_value_1 = orders_data.loc[orders_data['gender'] == 1,
                               'value'].agg(['mean']).iloc[0]
diff = mean_value_0 - mean_value_1
with open('ans_C.text', 'w') as f:
    print('Mean order value for gender 0:', mean_value_0, file=f)
    print('Mean order value for gender 1:', mean_value_1, file=f)
    print('Difference between Mean order value for gender 0 and 1:', diff, file=f)

# D
orders_data['new_predict_gender'] = [0 for i in range(13471)]

true_0, true_1, false_0, false_1 = 0, 0, 0, 0
for index, row in orders_data.iterrows():
    if row['gender'] == 0:
        if row['new_predict_gender'] == 0:
            true_0 += 1
        elif row['new_predict_gender'] == 1:
            false_1 += 1
    elif row['gender'] == 1:
        if row['new_predict_gender'] == 0:
            false_0 += 1
        elif row['new_predict_gender'] == 1:
            true_1 += 1

conf_str = str(true_0) + '\t' + str(false_1) + '\n' + \
    str(false_0) + '\t' + str(true_1)
print(conf_str)
print('accuracy=', (true_0 + true_1) / 13471)
print('precision=', (true_0) / (true_0 + false_0))
print('recall=', (true_0) / (true_0 + false_1))
# always wrong predictions for gender=1
# accuracy is 50% as only 2 choices
# precision is also 50% 
