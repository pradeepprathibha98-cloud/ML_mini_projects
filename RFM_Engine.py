import numpy as np
import pandas as pd

np.random

num_transactions = 20

customer_id = np.random.randint(100, 106, size = (num_transactions, 1))
days_ago = np.random.randint(1, 30, size = (num_transactions, 1))
purchase_amount = np.round(np.random.uniform(10.0, 100.0, size = (num_transactions, 1)), 2)

transactions = np.hstack((customer_id, days_ago, purchase_amount))

print("Raw Transaction Data:\n", transactions[:5])

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