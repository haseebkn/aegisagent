# AegisAgent: Serverless AI Compliance & MLOps Platform

AegisAgent is a secure, serverless Agent-as-a-Service (AaaS) platform designed to automate the detection of financial transaction anomalies and generate regulatory-compliant **Suspicious Transaction Report (STR)** narratives. 

Built for enterprise-grade RegTech environments, AegisAgent seamlessly merges a multi-model machine learning ensemble with a guardrailed generative AI agent, strictly conforming to FINTRAC and PCMLTFA AML data-retention standards.

---

## 🚀 Core Capabilities

* **Stacking Ensemble ML Classifier:** Evaluates transactions dynamically using a stacked architecture of Geographic Random Forests, Category-specific XGBoost models, and Velocity Random Forests, culminating in a highly calibrated Meta-Score.
* **Agentic Guardrails & LLM Orchestration:** Integrates with **AWS Bedrock (Claude Haiku 4.5)** to translate statistical anomalies into structured STRs. Implements a programmatic regex-validation loop that intercepts speculative language (e.g., "may", "might", "could") and forces LLM retries to ensure 100% definitive, objective regulatory reporting.
* **Immutable S3 Compliance Data Lake:** Configured with strict Write-Once-Read-Many (**WORM**) AWS Object Lock and automated lifecycle tiering to Glacier, physically preventing data tampering and fulfilling the 5-year regulatory retention mandate.
* **Production MLOps Pipeline:** Features strict runtime input validation via **Pydantic**, automated model artifact versioning (`v_YYYYMMDD_HHMMSS`), and persistent JSON telemetry tracking for every training lifecycle.
* **Premium Investigator Dashboard:** A Streamlit interface styled with custom dark-mode aesthetics, surfacing real-time model telemetry, feature weights, and immediate Markdown renderings of the Bedrock STRs.

---

## 🛠️ Technology Stack

* **Machine Learning & AI:** Scikit-Learn, XGBoost, AWS Bedrock (Anthropic Claude), Pydantic
* **Data Engineering:** dbt (Data Build Tool), DuckDB (Embedded OLAP)
* **Frontend Visualization:** Streamlit 
* **Infrastructure as Code (IaC):** Terraform (with S3/DynamoDB Remote State Locking)
* **Cloud & CI/CD:** AWS ECS Fargate, ECR, Docker, custom bash deployment orchestration

---

## 🏛️ Cloud Architecture & Security Design

AegisAgent enforces enterprise-level infrastructure security:
1. **Automated CI/CD:** A unified `deploy.sh` script automatically authenticates AWS credentials, builds the optimized local Docker container, and pushes it to Elastic Container Registry (ECR).
2. **Zero Hardcoded Keys:** AWS Fargate task configurations utilize IAM Roles to request temporary, least-privilege AWS Bedrock and S3 access tokens dynamically.
3. **Network Isolation:** Resources execute inside an isolated VPC with restricted ingress/egress rules via AWS Security Groups.
4. **State Protection:** Terraform state is managed remotely in an encrypted S3 bucket and locked via DynamoDB to prevent concurrent pipeline corruption.

---

## 💻 Local Setup & Execution

### 1. Prerequisites
* Python 3.10+
* Docker Desktop (for containerized execution)
* AWS CLI configured with active credentials

### 2. Installation
Clone this repository and set up a virtual environment:
```bash
git clone https://github.com/haseebkn/aegisagent.git
cd aegisagent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install strictly pinned dependencies
pip install -r requirements.txt
```
