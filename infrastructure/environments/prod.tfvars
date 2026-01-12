# Production Environment Configuration
# =====================================
# Use this file with: terraform apply -var-file=environments/prod.tfvars

environment = "prod"
aws_region  = "us-east-1"

# S3 Configuration
s3_bucket_name              = "potential-parakeet-cache"
enable_s3_versioning        = true  # Enable versioning for production
s3_lifecycle_expiration_days = 0    # Never delete (keep all historical data)

# Lambda Configuration
lambda_runtime                         = "python3.11"
lambda_memory_size                     = 1024 # Higher memory for production
lambda_timeout                         = 120  # 2 minutes
lambda_ephemeral_storage_size          = 1024
lambda_reserved_concurrent_executions  = 10   # Reserve capacity for critical functions

# API Gateway
api_gateway_stage_name         = "v1"
enable_api_gateway_logging     = true
api_throttle_rate_limit        = 1000  # Production traffic
api_throttle_burst_limit       = 2000

# EventBridge (Data Ingest)
daily_ingest_schedule = "cron(0 6 * * ? *)" # 6 AM UTC daily
enable_daily_ingest   = true

# Monitoring
enable_cloudwatch_alarms              = true
lambda_error_alarm_threshold          = 5
lambda_duration_alarm_threshold_ms    = 100000 # 100 seconds
alert_email                           = "alerts@yourdomain.com" # CHANGE THIS

# Database Configuration
# NOTE: Set these via environment variables or AWS Secrets Manager Console
# NEVER commit production credentials!
neon_database_url = "" # Set via TF_VAR_neon_database_url or Secrets Manager
db_username       = ""
db_password       = ""
db_host           = ""
db_name           = "potential_parakeet"

# API Keys
# NOTE: Set these via environment variables
tiingo_api_key  = "" # Set via TF_VAR_tiingo_api_key
polygon_api_key = ""

# Additional Tags
additional_tags = {
  CostCenter  = "production"
  Team        = "quant-team"
  Compliance  = "internal"
  Criticality = "high"
}
