# Development Environment Configuration
# =====================================
# Use this file with: terraform apply -var-file=environments/dev.tfvars

environment = "dev"
aws_region  = "us-east-1"

# S3 Configuration
s3_bucket_name              = "potential-parakeet-cache"
enable_s3_versioning        = false
s3_lifecycle_expiration_days = 30 # Delete cache files older than 30 days in dev

# Lambda Configuration
lambda_runtime                         = "python3.11"
lambda_memory_size                     = 512  # MB
lambda_timeout                         = 60   # seconds
lambda_ephemeral_storage_size          = 512  # MB
lambda_reserved_concurrent_executions  = 0    # Unreserved (cost optimization)

# API Gateway
api_gateway_stage_name         = "v1"
enable_api_gateway_logging     = true
api_throttle_rate_limit        = 100   # Lower limit for dev
api_throttle_burst_limit       = 200

# EventBridge (Data Ingest)
daily_ingest_schedule = "cron(0 6 * * ? *)" # 6 AM UTC
enable_daily_ingest   = false # Disabled in dev to save costs

# Monitoring
enable_cloudwatch_alarms              = false # Disabled in dev
lambda_error_alarm_threshold          = 10
lambda_duration_alarm_threshold_ms    = 55000
alert_email                           = ""

# Database Configuration
# NOTE: Set these via environment variables or AWS Secrets Manager
# Do not commit actual credentials!
neon_database_url = "" # Set via TF_VAR_neon_database_url
db_username       = ""
db_password       = ""
db_host           = ""
db_name           = "potential_parakeet_dev"

# API Keys
# NOTE: Set these via environment variables
tiingo_api_key  = "" # Set via TF_VAR_tiingo_api_key
polygon_api_key = ""

# Additional Tags
additional_tags = {
  CostCenter = "development"
  Team       = "quant-dev"
}
