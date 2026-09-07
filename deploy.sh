#!/usr/bin/env bash
# Explicit reference-infrastructure provisioning; this does not deploy a usable app.
#
# --check validates local artifacts without provisioning or paid cloud requests.
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_NAME="${PROJECT_NAME:-aegis-agent}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACTS_DIR="${MODELS_ARTIFACTS_DIR:-$REPO_ROOT/models_artifacts}"
MODE="${1:---check}"
if [[ "$MODE" != "--check" && "$MODE" != "--apply-reference" ]]; then
  echo "Usage: bash deploy.sh [--check|--apply-reference]"
  exit 2
fi
export TF_VAR_aws_region="$AWS_REGION"
export TF_VAR_project_name="$PROJECT_NAME"

echo "======================================================================"
echo "               AEGISAGENT CLOUD DEPLOYMENT"
echo "======================================================================"

# --- Step 0: preflight -------------------------------------------------------
# The image bakes in model artifacts (see Dockerfile). Building without them
# either fails on COPY or, worse, ships whatever stale version happens to be on
# disk. Verify the artifacts exist and match the version pointer before building.
echo "--- Step 0: Preflight ---"
for tool in python docker; do
  command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: '$tool' not on PATH."; exit 1; }
done
if [ "$ARTIFACTS_DIR" != "$REPO_ROOT/models_artifacts" ]; then
  echo "ERROR: Docker builds use models_artifacts/ in this checkout."
  echo "       An external MODELS_ARTIFACTS_DIR would validate different artifacts."
  exit 1
fi

if [ ! -f "$ARTIFACTS_DIR/latest_version.txt" ]; then
  echo "ERROR: $ARTIFACTS_DIR/latest_version.txt not found."
  echo "       Model artifacts are a build input. Run first:"
  echo "         dbt run --profiles-dir ."
  echo "         python scripts/train_models.py"
  exit 1
fi

MODEL_VERSION="$(tr -d '[:space:]' < "$ARTIFACTS_DIR/latest_version.txt")"
if [[ ! "$MODEL_VERSION" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$ ]]; then
  echo "ERROR: Invalid model version in serving pointer."
  exit 1
fi
if [ ! -d "$ARTIFACTS_DIR/$MODEL_VERSION" ]; then
  echo "ERROR: latest_version.txt names '$MODEL_VERSION' but that directory is missing."
  echo "       Restore the governed artifacts; do not manually repoint serving."
  exit 1
fi
for f in model_2_geo_rf.joblib model_3_cat_xgb.joblib model_4_vel_rf.joblib \
         model_4_scaler.joblib meta_model.joblib meta_threshold.txt; do
  [ -f "$ARTIFACTS_DIR/$MODEL_VERSION/$f" ] || {
    echo "ERROR: $MODEL_VERSION is incomplete, missing $f. Retrain before deploying."
    exit 1; }
done
PYTHONPATH="$REPO_ROOT" MODELS_ARTIFACTS_DIR="$ARTIFACTS_DIR" python -c \
  "from scripts.inference_engine import load_models; load_models()"
if [ "$MODE" = "--check" ]; then
  echo "Local artifact preflight passed. No cloud resources changed."
  exit 0
fi
for tool in terraform aws git; do
  command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: '$tool' not on PATH."; exit 1; }
done
if [ ! -f "$REPO_ROOT/terraform/backend.hcl" ]; then
  echo "ERROR: Configure terraform/backend.hcl from backend.hcl.example first."
  exit 1
fi
SOURCE_REVISION="$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD)"
if [ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal)" ]; then
  echo "ERROR: Commit or remove outstanding source changes before a traced cloud build."
  exit 1
fi
export TF_VAR_image_tag="$MODEL_VERSION-$SOURCE_REVISION"
echo "Provisioning reference for model version: $MODEL_VERSION"

# --- Step 1: bootstrap ECR ---------------------------------------------------
echo "--- Step 1: Initializing Terraform and bootstrapping ECR ---"
terraform -chdir="$REPO_ROOT/terraform" init -input=false -backend-config=backend.hcl
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
docker tag aegis-app:latest "$ECR_REPO_URL:$TF_VAR_image_tag"
docker push "$ECR_REPO_URL:latest"
docker push "$ECR_REPO_URL:$TF_VAR_image_tag"

# --- Step 5: apply -----------------------------------------------------------
echo "--- Step 5: Applying full Terraform stack ---"
terraform -chdir="$REPO_ROOT/terraform" apply -input=false -auto-approve

echo "======================================================================"
echo "  REFERENCE PROVISIONED - image tagged $TF_VAR_image_tag; ECS remains at zero tasks."
echo "======================================================================"
