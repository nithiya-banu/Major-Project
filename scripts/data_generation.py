import pandas as pd
import random
from faker import Faker
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os, shutil

if os.path.exists("data"):
    shutil.rmtree("data")
os.mkdir("data/")

fake = Faker()

def generate_project_data(num_records):
    data = []
    for _ in range(num_records):
        # planned_date = fake.date_this_year()
        
        bugs_found = random.randint(10, 100)
        bugs_fixed = random.randint(0, bugs_found)
        critical_issues_found = random.randint(0, bugs_found // 2)

        planned_features = random.randint(5, 50)
        delivered_features = max(0, planned_features - random.randint(0, 10))

        team_skill_level = random.choice(["Low", "Medium", "High"])
        resource_utilization = random.randint(50, 100)
        client_involvement = random.choice(["Low", "Medium", "High"])
        complexity_score = random.choice(["Low", "Medium", "High"])

        # Delay is influenced by these factors
        bug_fix_ratio = bugs_fixed / bugs_found if bugs_found > 0 else 0
        feature_delivery_ratio = delivered_features / planned_features if planned_features > 0 else 0
        critical_issue_ratio = critical_issues_found / bugs_found if bugs_found > 0 else 0

        delay = 0
        delay += (bugs_found * 0.3)
        delay += (critical_issues_found * 1.5)
        delay += (planned_features - delivered_features) * 0.5
        delay += random.randint(0, 10) * (1 if complexity_score == "High" else 0.5)
        delay += (10 - resource_utilization / 10) * 2
        
        delay = max(-20, min(50, delay))
        
        data.append({
            "Delay (Days)": delay,
            "Bugs Found": bugs_found,
            "Bugs Fixed": bugs_fixed,
            "Bug Fix Ratio": bug_fix_ratio,
            "Team Size": random.randint(5, 20),
            "Hours Worked": random.randint(200, 2000),
            "Budget (USD)": random.randint(50000, 200000),
            "Risk Factor": random.randint(1, 10),
            "Critical Issues Found": critical_issues_found,
            "Critical Issue Ratio": critical_issue_ratio,
            "Feature Delivery Ratio": feature_delivery_ratio,
            "Team Skill Level": team_skill_level,
            "Resource Utilization Rate": resource_utilization,
            "Client Involvement": client_involvement,
            "Complexity Score": complexity_score
        })
    return pd.DataFrame(data)
def generate_client_data(num_clients):
    data = []
    for _ in range(num_clients):
        complaints = random.randint(0, 15)
        resolved = random.randint(0, complaints)
        satisfaction_score = random.randint(1, 10)
        revenue = random.randint(10000, 500000)
        communication_rating = random.randint(1, 10)
        past_renewals = random.randint(0, 5)
        churn_risk = random.choice(["Low", "Medium", "High"])
        feedback_sentiment = random.choice(["Positive", "Neutral", "Negative"])

        complaint_resolution_ratio = resolved / complaints if complaints > 0 else 0

        data.append({
            "Satisfaction Score": satisfaction_score,
            "Complaints Raised": complaints,
            "Complaints Resolved": resolved,
            "Complaint Resolution Ratio": complaint_resolution_ratio,
            "Resolution Time (hours)": random.randint(1, 72),
            "Feedback Sentiment": feedback_sentiment,
            "Revenue Generated (USD)": revenue,
            "Service Renewal Likelihood (%)": random.randint(50, 100),
            "Client Communication Rating": communication_rating,
            "Past Renewals": past_renewals,
            "Churn Risk": churn_risk
        })
    return pd.DataFrame(data)

def preprocess_data():
    projects = generate_project_data(40000)
    clients = generate_client_data(1000)

    projects.to_csv("data/data_generated_project_data.csv", index =False)
    clients.to_csv("data/data_generated_client_data.csv", index =False)

    projects = projects.dropna()
    clients = clients.dropna()

    label_encoder = LabelEncoder()
    projects["Team Skill Level"] = label_encoder.fit_transform(projects["Team Skill Level"])
    projects["Client Involvement"] = label_encoder.fit_transform(projects["Client Involvement"])
    projects["Complexity Score"] = label_encoder.fit_transform(projects["Complexity Score"])

    clients["Churn Risk"] = label_encoder.fit_transform(clients["Churn Risk"])
    clients["Feedback Sentiment"] = label_encoder.fit_transform(clients["Feedback Sentiment"])
    scaler = StandardScaler()

    numeric_columns_projects = ["Bugs Found", "Bugs Fixed", "Team Size", "Hours Worked", "Budget (USD)", 
                               "Risk Factor", "Critical Issues Found", "Critical Issue Ratio", "Feature Delivery Ratio", 
                               "Resource Utilization Rate"]

    numeric_columns_clients = ["Complaints Raised", "Complaints Resolved", "Complaint Resolution Ratio", 
                               "Resolution Time (hours)", "Revenue Generated (USD)", "Service Renewal Likelihood (%)", 
                               "Client Communication Rating", "Past Renewals"]

    projects[numeric_columns_projects] = scaler.fit_transform(projects[numeric_columns_projects])
    clients[numeric_columns_clients] = scaler.fit_transform(clients[numeric_columns_clients])

    X_projects = projects.drop(columns=["Delay (Days)"])
    y_projects = projects["Delay (Days)"]

    X_train_projects, X_test_projects, y_train_projects, y_test_projects = train_test_split(X_projects, y_projects, test_size=0.2, random_state=42)

    X_clients = clients.drop(columns=["Satisfaction Score"])
    y_clients = clients["Satisfaction Score"]
    X_train_clients, X_test_clients, y_train_clients, y_test_clients = train_test_split(X_clients, y_clients, test_size=0.2, random_state=42)

    projects.to_csv("data/processed_project_data.csv", index=False)
    clients.to_csv("data/processed_client_data.csv", index=False)


    print("Preprocessing completed and datasets saved!")
if __name__ == "__main__":
    preprocess_data()
