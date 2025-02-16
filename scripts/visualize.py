import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

projects = pd.read_csv("cleaned_project_data.csv")
clients = pd.read_csv("cleaned_client_data.csv")

projects = projects.select_dtypes(include=[float, int])
clients = clients.select_dtypes(include=[float, int])

plt.figure(figsize=(12, 8))
sns.heatmap(projects.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Project Data Correlation Heatmap")
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(projects["Delay (Days)"], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Project Delays")
plt.xlabel("Delay (Days)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=projects["Budget (USD)"], y=projects["Delay (Days)"], alpha=0.5)
plt.title("Budget vs Project Delay")
plt.xlabel("Budget (Normalized)")
plt.ylabel("Delay (Days)")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=projects["Risk Factor"], y=projects["Bugs Found"])
plt.title("Risk Factor vs Bugs Found")
plt.xlabel("Risk Factor")
plt.ylabel("Bugs Found")
plt.show()

sns.pairplot(projects[["Team Size", "Hours Worked", "Bugs Found", "Risk Factor", "Delay (Days)"]], diag_kind='kde')
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(clients.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Client Data Correlation Heatmap")
plt.show()
plt.figure(figsize=(10, 5))
sns.histplot(clients["Satisfaction Score"], bins=10, kde=True, color='skyblue')
plt.title("Distribution of Client Satisfaction Scores")
plt.xlabel("Satisfaction Score")
plt.ylabel("Count")
plt.show()

client_category_cols = [col for col in clients.columns if 'Client Category' in col]
for col in client_category_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=clients[col], y=clients["Revenue Generated (USD)"])
    plt.title(f"{col} vs Revenue Generated")
    plt.xlabel(col)
    plt.ylabel("Revenue Generated (Normalized)")
    plt.show()
