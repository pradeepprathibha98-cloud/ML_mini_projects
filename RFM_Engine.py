import numpy as np

np.random.seed()

num_transactions = 20

customer_id = np.random.randint(100, 600, size = (num_transactions, 1))
days_ago = np.random.randint(1, 30, size = (num_transactions, 1))
purchase_amount = np.round(np.random.uniform(10.0, 100.0, size = (num_transactions, 1)), 2)

transactions = np.hstack((customer_id, days_ago, purchase_amount))

print("Raw Transaction Data:\n", transactions[:5])