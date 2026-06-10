terraform {
  backend "s3" {
    bucket         = "aegis-agent-tfstate-878533945497"
    key            = "production/aegis-agent/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "aegis-agent-tflocks"
  }
}
