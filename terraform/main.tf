terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# =====================================================================
# VARIABLES & LOCALS
# =====================================================================
variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project_name" {
  type    = string
  default = "aegis-agent"
}

variable "bedrock_inference_profile" {
  type        = string
  description = "Cross-region inference profile the STR agent invokes."
  default     = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
}

variable "bedrock_foundation_model" {
  type        = string
  description = "Underlying foundation model behind the inference profile."
  default     = "anthropic.claude-haiku-4-5-20251001-v1:0"
}

variable "bedrock_profile_regions" {
  type        = list(string)
  description = "Regions the cross-region profile can route to. Permission is required on the foundation model in each."
  default     = ["us-east-1", "us-east-2", "us-west-2"]
}

# =====================================================================
# AMAZON S3 COMPLIANCE DATA LAKE
# =====================================================================
resource "aws_s3_bucket" "compliance_lake" {
  bucket              = "${var.project_name}-compliance-logs-${data.aws_caller_identity.current.account_id}"
  force_destroy       = false
  object_lock_enabled = true # CRITICAL: Required for FINTRAC WORM compliance
}

resource "aws_s3_bucket_server_side_encryption_configuration" "crypto" {
  bucket = aws_s3_bucket.compliance_lake.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "private_boundary" {
  bucket                  = aws_s3_bucket.compliance_lake.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# 1. Enforce Versioning (Required for Object Lock)
resource "aws_s3_bucket_versioning" "compliance_lake_versioning" {
  bucket = aws_s3_bucket.compliance_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

# 2. Apply WORM (Write Once Read Many) Object Lock
resource "aws_s3_bucket_object_lock_configuration" "compliance_lake_lock" {
  bucket = aws_s3_bucket.compliance_lake.id

  rule {
    default_retention {
      mode = "COMPLIANCE" # Strictly prevents deletion/alteration, even by the root user
      days = 1825         # 5 years
    }
  }
}

# 3. Automate Cost Efficiency with Lifecycle Rules
resource "aws_s3_bucket_lifecycle_configuration" "compliance_lake_lifecycle" {
  bucket = aws_s3_bucket.compliance_lake.id

  rule {
    id     = "archive-fintrac-records"
    status = "Enabled"

    filter {
      prefix = ""
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 1825
    }
  }
}

# =====================================================================
# IAM SECURITY ROLES (ELIMINATING HARDCODED KEYS)
# =====================================================================
data "aws_caller_identity" "current" {}

resource "aws_iam_role" "ecs_execution_role" {
  name = "${var.project_name}-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_standard" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "task_role" {
  name = "${var.project_name}-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_policy" "pipeline_permissions" {
  name = "${var.project_name}-pipeline-permissions"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # The application invokes a CROSS-REGION inference profile ("us." prefix).
      # That requires permission on the profile ARN *and* on the underlying
      # foundation model in every region the profile can route to. Granting only
      # the foundation-model ARN in one region -- as this policy previously did --
      # produces AccessDeniedException at invoke time.
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = concat(
          ["arn:aws:bedrock:${var.aws_region}:${data.aws_caller_identity.current.account_id}:inference-profile/${var.bedrock_inference_profile}"],
          [for r in var.bedrock_profile_regions :
          "arn:aws:bedrock:${r}::foundation-model/${var.bedrock_foundation_model}"]
        )
      },
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.compliance_lake.arn,
          "${aws_s3_bucket.compliance_lake.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "task_custom_attach" {
  role       = aws_iam_role.task_role.name
  policy_arn = aws_iam_policy.pipeline_permissions.arn
}

# =====================================================================
# AWS ECR REPOSITORY
# =====================================================================
resource "aws_ecr_repository" "aegis_app" {
  name                 = "aegis-app"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Add ECR Lifecycle Policy to clean up orphaned blobs
resource "aws_ecr_lifecycle_policy" "ecr_cleanup" {
  repository = aws_ecr_repository.aegis_app.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Expire untagged image blobs after 7 days"
      selection = {
        tagStatus   = "untagged"
        countType   = "sinceImagePushed"
        countUnit   = "days"
        countNumber = 7
      }
      action = { type = "expire" }
    }]
  })
}

# =====================================================================
# AWS ECS FARGATE ORCHESTRATION LAYER
# =====================================================================
resource "aws_ecs_cluster" "cluster" {
  name = "${var.project_name}-cluster"
}

resource "aws_cloudwatch_log_group" "logs" {
  name              = "/ecs/${var.project_name}"
  retention_in_days = 1827 # Updated from 7 days to 5 years (AWS uses 1827 for 5 yrs)
}

resource "aws_ecs_task_definition" "pipeline_task" {
  family                   = "${var.project_name}-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024" # 1 vCPU
  memory                   = "2048" # 2 GB RAM
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn            = aws_iam_role.task_role.arn

  container_definitions = jsonencode([{
    name      = "aegis_app"
    image     = "${aws_ecr_repository.aegis_app.repository_url}:latest"
    essential = true

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.logs.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "pipeline"
      }
    }

    environment = [
      { name = "DBT_DB_PATH", value = "/app/aegis_db.duckdb" },
      # /app/models is the dbt project; the joblib artifacts live in
      # /app/models_artifacts. Pointing this at /app/models made the task unable to
      # load any model.
      { name = "MODELS_ARTIFACTS_DIR", value = "/app/models_artifacts" },
      { name = "COMPLIANCE_LOGS_DIR", value = "/app/compliance_logs" },
      { name = "AEGIS_RAW_DATA_DIR", value = "/app" },
      { name = "DBT_PROFILES_DIR", value = "/app" },
      { name = "COMPLIANCE_S3_BUCKET", value = aws_s3_bucket.compliance_lake.id },
      { name = "AWS_DEFAULT_REGION", value = var.aws_region }
    ]
  }])
}

# =====================================================================
# ISOLATED PRODUCTION NETWORKING (Fix C-02)
# =====================================================================
resource "aws_vpc" "production" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = {
    Name = "${var.project_name}-production-vpc"
  }
}

resource "aws_subnet" "public_subnet" {
  vpc_id                  = aws_vpc.production.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "us-east-1a"
  map_public_ip_on_launch = true
  tags = {
    Name = "${var.project_name}-public-subnet"
  }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.production.id
  tags = {
    Name = "${var.project_name}-igw"
  }
}

resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.production.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = {
    Name = "${var.project_name}-public-route-table"
  }
}

resource "aws_route_table_association" "public_subnet_association" {
  subnet_id      = aws_subnet.public_subnet.id
  route_table_id = aws_route_table.public_rt.id
}

resource "aws_security_group" "ecs_sg" {
  name        = "${var.project_name}-ecs-security-group"
  description = "Egress-only security group for compliance service tasks"
  vpc_id      = aws_vpc.production.id

  egress {
    description = "Allow HTTPS/HTTP outbound traffic for Bedrock/S3 and general package/DNS handshakes"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-ecs-sg"
  }
}

# =====================================================================
# ECS SERVICE (Fix C-01)
# =====================================================================
resource "aws_ecs_service" "aegis_compliance_service" {
  name            = "aegis-compliance-service"
  cluster         = aws_ecs_cluster.cluster.id
  task_definition = aws_ecs_task_definition.pipeline_task.arn
  desired_count   = 0
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = [aws_subnet.public_subnet.id]
    security_groups  = [aws_security_group.ecs_sg.id]
    assign_public_ip = true
  }
}

# =====================================================================
# OUTPUTS
# =====================================================================
output "s3_bucket_name" {
  value       = aws_s3_bucket.compliance_lake.id
  description = "The dynamically created secure S3 storage bucket name."
}

output "ecs_task_definition_arn" {
  value       = aws_ecs_task_definition.pipeline_task.arn
  description = "The ARN of the newly registered serverless Fargate configuration."
}
