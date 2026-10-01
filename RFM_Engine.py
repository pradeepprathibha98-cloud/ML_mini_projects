import numpy as np

#Raw Transaction matrix

np.random

num_transactions = 20

customer_id = np.random.randint(100, 106, size = (num_transactions, 1))
days_ago = np.random.randint(1, 30, size = (num_transactions, 1))
purchase_amount = np.round(np.random.uniform(10.0, 100.0, size = (num_transactions, 1)), 2)

transactions = np.hstack((customer_id, days_ago, purchase_amount))

print("Raw Transaction Data:\n", transactions[:5])

#Aggregation of the Matrix

unique_customers = np.unique(transactions[:, 0])

aggreg_data = []

for customer_id in unique_customers:
    mask = transactions[:, 0] == customer_id
    customer_txs = transactions[mask]

    recency = np.min(customer_txs[:, 1])
    frequency = len(customer_txs)
    monetary = np.sum(customer_txs[:, 2])

    aggreg_data.append([customer_id, recency, frequency, monetary])

rfm_matrix = np.array(aggreg_data)

print("Aggregated RFM Matrix (Customer_ID, Recency, Frequency, Monetary) : \n", rfm_matrix)

#Quantile Scoring

recency = rfm_matrix[:, 1]
frequency = rfm_matrix[:, 2]
monetary = rfm_matrix[:, 3]

rq = np.percentile(recency, [25, 50, 75])
r_scores = np.select(
    [recency <= rq[0], recency <= rq[1], recency <= rq[2]], [4, 3, 2], default = 1
)

mq = np.percentile(monetary, [25, 50, 75])
m_scores = np.select(
    [monetary >= mq[2], monetary >= mq[1], monetary >= mq[0]], [4, 3, 2], default = 1
)

fq = np.percentile(frequency, [25, 50, 75])
f_scores = np.select(
    [frequency >= fq[2], frequency >= fq[1], frequency >= fq[0]], [4, 3, 2], default = 1
)

print("Recency scoring : ", r_scores)
print("Frequency scoring : ", f_scores)
print("Monetary scoring : ", m_scores)

#Vectorized segmentation 

conditions = [
    (r_scores >= 3) & (m_scores >= 3),
    (r_scores <= 2) & (m_scores >= 3),
    (r_scores >= 3) & (m_scores <= 2)
]

choices = ["VIP Customer", "At Risk", "New Customer"]
segments = np.select(conditions, choices, default = "Regular")

final_scores = np.column_stack((rfm_matrix, r_scores, f_scores, m_scores))

print("Final RFM Scores Table : \n", final_scores)

print("Customer Segments\n")
for i in range(len(rfm_matrix)):
    cust_id = int(rfm_matrix[i, 0])
    print(f"Customer {cust_id} | Segment : {segments[i]}")