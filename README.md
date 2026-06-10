# AegisAgent Compliance Dashboard

AegisAgent is a secure, serverless compliance auditing platform designed to automate the detection of financial transaction anomalies and generate regulatory-compliant **Suspicious Transaction Report (STR)** narratives. 

Built with security and compliance in mind, AegisAgent conforms to FINTRAC and general anti-money laundering (AML) data-retention standards.

---

## 🚀 Key Features

* **AI-Generated STR Narratives:** Integrates with **AWS Bedrock (Claude)** to translate statistical transaction anomalies into high-quality, structured STRs covering the *Who, What, When, Where, Why,* and *How* of suspicious activities.
* **Immutable S3 Compliance Data Lake:** Implements a strict Write-Once-Read-Many (**WORM**) Object Lock configuration on AWS S3 to prevent data tampering, fulfilling regulatory requirements for record integrity (5-year retention).
* **Robust Feature Engineering:** Utilizes **DBT (Data Build Tool)** and **DuckDB** to generate advanced velocity and transaction frequency metrics.
* **Premium Custom Dashboard:** A Streamlit interface styled with custom dark-mode aesthetics, responsive visual cards, and clear typography.

---

## 🛠️ Technology Stack

* **Frontend Dashboard:** Streamlit (with customized CSS/HTML elements)
* **Data Processing & Database:** DBT Core + DuckDB (embedded OLAP database)
* **Infrastructure as Code (IaC):** Terraform
* **AI Orchestration:** Anthropic Claude (via AWS Bedrock)
* **Serverless Compute:** AWS ECS Fargate + ECR (Dockerized deployment)

---

## 🏛️ Architecture & Security Design

AegisAgent enforces enterprise-level security protocols out of the box:
1. **Zero Hardcoded Keys:** Task configurations use IAM Roles and Policies to request temporary AWS Bedrock and S3 access tokens dynamically.
2. **Network Isolation:** Resources run in an isolated VPC with restricted ingress/egress rules via AWS Security Groups.
3. **Data Protection:** Standard AES256 server-side encryption blocks public access to S3 buckets.

---

## 💻 Local Setup & Execution

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Installation
Clone this repository and set up a virtual environment:
```bash
git clone https://github.com/haseebkn/aegisagent.git
cd aegisagent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard
Launch the web application locally:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🌐 Deploying to AWS

Initialize and deploy the infrastructure using Terraform:
```bash
cd terraform
terraform init
terraform plan
terraform apply
```

This constructs the isolated VPC, ECR registry, ECS cluster, S3 compliance bucket, and execution policies automatically.
