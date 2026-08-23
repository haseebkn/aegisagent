#!/usr/bin/env bash
# Manual cloud deployment: build, push to ECR, apply Terraform.
#
# This is a deliberate manual step, not CI. It is idempotent and safe to re-run.
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_NAME="${PROJECT_NAME:-aegis-agent}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACTS_DIR="${MODELS_ARTIFACTS_DIR:-$REPO_ROOT/models_artifacts}"

echo "======================================================================"
echo "               AEGISAGENT CLOUD DEPLOYMENT"
echo "======================================================================"

# --- Step 0: preflight -------------------------------------------------------
# The image bakes in model artifacts (see Dockerfile). Building without them
# either fails on COPY or, worse, ships whatever stale version happens to be on
# disk. Verify the artifacts exist and match the version pointer before building.
echo "--- Step 0: Preflight ---"
for tool in terraform aws docker; do
  command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: '$tool' not on PATH."; exit 1; }
done

if [ ! -f "$ARTIFACTS_DIR/latest_version.txt" ]; then
  echo "ERROR: $ARTIFACTS_DIR/latest_version.txt not found."
  echo "       Model artifacts are a build input. Run first:"
  echo "         dbt run --profiles-dir ."
  echo "         python scripts/train_models.py"
  exit 1
fi

MODEL_VERSION="$(tr -d '[:space:]' < "$ARTIFACTS_DIR/latest_version.txt")"
if [ ! -d "$ARTIFACTS_DIR/$MODEL_VERSION" ]; then
  echo "ERROR: latest_version.txt names '$MODEL_VERSION' but that directory is missing."
  echo "       Retrain, or repoint latest_version.txt at a version that exists."
  exit 1
fi
for f in model_2_geo_rf.joblib model_3_cat_xgb.joblib model_4_vel_rf.joblib \
         model_4_scaler.joblib meta_model.joblib meta_threshold.txt; do
  [ -f "$ARTIFACTS_DIR/$MODEL_VERSION/$f" ] || {
    echo "ERROR: $MODEL_VERSION is incomplete, missing $f. Retrain before deploying."
    exit 1; }
done
echo "Deploying model version: $MODEL_VERSION (threshold $(cat "$ARTIFACTS_DIR/$MODEL_VERSION/meta_threshold.txt"))"

# --- Step 1: bootstrap ECR ---------------------------------------------------
echo "--- Step 1: Initializing Terraform and bootstrapping ECR ---"
terraform -chdir="$REPO_ROOT/terraform" init -input=false
terraform -chdir="$REPO_ROOT/terraform" apply -input=false -auto-approve \
  -target=aws_ecr_repository.aegis_app

ECR_REPO_URL="$(terraform -chdir="$REPO_ROOT/terraform" output -raw ecr_repository_url 2>/dev/null || true)"
if [ -z "$ECR_REPO_URL" ]; then
  echo "No terraform output available; querying ECR directly..."
  ECR_REPO_URL="$(aws ecr describe-repositories \
    --repository-names aegis-app --region "$AWS_REGION" \
    --query 'repositories[0].repositoryUri' --output text)"
fi
ECR_REGISTRY="${ECR_REPO_URL%%/*}"
echo "Repository: $ECR_REPO_URL"

# --- Step 2: authenticate ----------------------------------------------------
echo "--- Step 2: Authenticating Docker with Amazon ECR ---"
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

# --- Step 3: build -----------------------------------------------------------
echo "--- Step 3: Building image ---"
docker build -t aegis-app:latest "$REPO_ROOT"

# --- Step 4: push ------------------------------------------------------------
echo "--- Step 4: Tagging and pushing to ECR ---"
# Tag with the model version as well as latest, so a deployed task can be traced
# back to the exact artifacts it is serving.
docker tag aegis-app:latest "$ECR_REPO_URL:latest"
docker tag aegis-app:latest "$ECR_REPO_URL:$MODEL_VERSION"
docker push "$ECR_REPO_URL:latest"
docker push "$ECR_REPO_URL:$MODEL_VERSION"

# --- Step 5: apply -----------------------------------------------------------
echo "--- Step 5: Applying full Terraform stack ---"
terraform -chdir="$REPO_ROOT/terraform" apply -input=false -auto-approve

echo "======================================================================"
echo "  DEPLOYMENT COMPLETE - image tagged latest and $MODEL_VERSION"
echo "======================================================================"
