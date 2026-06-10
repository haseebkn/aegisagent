#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

AWS_REGION="us-east-1"
PROJECT_NAME="aegis-agent"
terraform() {
  terraform.exe "$@"
}

aws() {
  aws.exe "$@"
}

echo "======================================================================"
echo "               AEGISAGENT PHASE 2 CLOUD DEPLOYMENT"
echo "======================================================================"

echo "--- Step 1: Initializing and Bootstrap Terraform (ECR Repository) ---"
cd terraform
terraform init
terraform apply -target=aws_ecr_repository.aegis_app -auto-approve

# Retrieve the repository URL dynamically from state
ECR_REPO_URL=$(terraform state show aws_ecr_repository.aegis_app | grep repository_url | awk '{print $NF}' | tr -d '"')

if [ -z "$ECR_REPO_URL" ]; then
  echo "State lookup empty, fetching repository URI via AWS CLI..."
  ECR_REPO_URL=$(aws ecr describe-repositories --repository-names aegis-app --region $AWS_REGION --query "repositories[0].repositoryUri" --output text)
fi

ECR_REGISTRY=$(echo "$ECR_REPO_URL" | awk -F '/' '{print $1}')

echo "Repository URL: $ECR_REPO_URL"
echo "Registry Domain: $ECR_REGISTRY"

echo "--- Step 2: Authenticating local Docker CLI with Amazon ECR ---"
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY

echo "--- Step 3: Building optimized local Docker image 'aegis-app:latest' ---"
cd ..
docker build -t aegis-app:latest .

echo "--- Step 4: Tagging and pushing Docker image to ECR ---"
docker tag aegis-app:latest "$ECR_REPO_URL:latest"
docker push "$ECR_REPO_URL:latest"

echo "--- Step 5: Executing final Terraform apply to orchestrate ECS Fargate Stack ---"
cd terraform
terraform apply -auto-approve

echo "======================================================================"
echo "         CLOUD RESOURCE DEPLOYMENT AND ORCHESTRATION COMPLETED"
echo "======================================================================"
