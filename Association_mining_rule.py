# %% Crete a list of list of transactions
dataset = [
    ["Milk", "Onion", "Nutmeg", "Kidney Beans", "Eggs", "Yogurt"],
    ["Dill", "Onion", "Nutmeg", "Kidney Beans", "Eggs", "Yogurt"],
    ["Milk", "Apple", "Kidney Beans", "Eggs"],
    ["Milk", "Unicorn", "Corn", "Kidney Beans", "Yogurt"],
    ["Corn", "Onion", "Onion", "Kidney Beans", "Ice cream", "Eggs"],
]

# %% Import libraries and transform the dataset
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import seaborn as sns
import matplotlib.pyplot as plt

te = TransactionEncoder()
te_ar = te.fit_transform(dataset)
df = pd.DataFrame(te_ar, columns=te.columns_)

# %% Compute frequent itemsets using the Apriori algorithm
frequent_itemset = apriori(df, min_support=0.6,use_colnames=True)

# %% Compute all association rules for frequent_itemsets
rules = association_rules(frequent_itemset, min_threshold=0.1)

# %% top 10 lift association rules
rules.sort_values("lift",ascending=False).head(5)

# %% scatterplot support and confidence
sns.scatterplot(x = "support", y = "confidence", alpha=0.5, data=rules)
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support and confidence")
# %%
