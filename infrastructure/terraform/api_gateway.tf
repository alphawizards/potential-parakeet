/**
 * API Gateway Configuration
 * =========================
 * HTTP API (API Gateway v2) for serverless backend.
 *
 * Features:
 * - HTTP API (faster and cheaper than REST API)
 * - Lambda integrations for all API endpoints
 * - JWT Authorizer for Cloudflare Access validation
 * - CORS configuration for frontend
 * - Access logging
 */

# ============================================================================
# HTTP API
# ============================================================================

resource "aws_apigatewayv2_api" "main" {
  name          = "potential-parakeet-api-${var.environment}"
  description   = "Potential Parakeet API Gateway (${var.environment})"
  protocol_type = "HTTP"

  cors_configuration {
    allow_credentials = false
    allow_headers     = ["Content-Type", "Authorization", "X-Requested-With", "CF-Access-JWT-Assertion"]
    allow_methods     = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_origins = var.environment == "prod" ? [
      "https://potential-parakeet.pages.dev"
    ] : [
      "http://localhost:3000",
      "http://localhost:5173",
      "https://potential-parakeet.pages.dev"
    ]
    expose_headers = ["X-Request-Id"]
    max_age        = 3600
  }

  tags = merge(local.common_tags, {
    Name = "api-gateway-${var.environment}"
  })
}

# ============================================================================
# API Gateway Stage
# ============================================================================

resource "aws_apigatewayv2_stage" "main" {
  api_id      = aws_apigatewayv2_api.main.id
  name        = "$default"
  auto_deploy = true

  default_route_settings {
    throttling_rate_limit  = var.api_throttle_rate_limit
    throttling_burst_limit = var.api_throttle_burst_limit
  }

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway.arn
    format = jsonencode({
      requestId         = "$context.requestId"
      ip                = "$context.identity.sourceIp"
      requestTime       = "$context.requestTime"
      httpMethod        = "$context.httpMethod"
      routeKey          = "$context.routeKey"
      status            = "$context.status"
      protocol          = "$context.protocol"
      responseLength    = "$context.responseLength"
      integrationError  = "$context.integrationErrorMessage"
      errorMessage      = "$context.error.message"
      authorizerError   = "$context.authorizer.error"
    })
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/apigateway/potential-parakeet-${var.environment}"
  retention_in_days = var.environment == "prod" ? 30 : 7

  tags = local.common_tags
}

# ============================================================================
# JWT Authorizer (Cloudflare Access) - OPTIONAL
# ============================================================================

# The authorizer validates JWTs from Cloudflare Access
# To enable: Set cloudflare_access_audience to a non-empty value
# Configure your Cloudflare Access application to issue JWTs with:
# - Issuer: https://<your-team>.cloudflareaccess.com
# - Audience: <your-application-aud>

# NOTE: Authorizer is disabled for initial deployment
# Uncomment and configure when Cloudflare Access is set up
# resource "aws_apigatewayv2_authorizer" "cloudflare_jwt" {
#   count            = var.cloudflare_access_audience != "" && var.cloudflare_access_audience != "placeholder" ? 1 : 0
#   api_id           = aws_apigatewayv2_api.main.id
#   name             = "cloudflare-access-jwt"
#   authorizer_type  = "JWT"
#   identity_sources = ["$request.header.CF-Access-JWT-Assertion"]
#
#   jwt_configuration {
#     issuer   = "https://${var.cloudflare_team_domain}.cloudflareaccess.com"
#     audience = [var.cloudflare_access_audience]
#   }
# }

# ============================================================================
# Lambda Integrations
# ============================================================================

# Trades API Integration
resource "aws_apigatewayv2_integration" "trades" {
  api_id                 = aws_apigatewayv2_api.main.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.trades_api.invoke_arn
  payload_format_version = "2.0"
}

# Data API Integration
resource "aws_apigatewayv2_integration" "data" {
  api_id                 = aws_apigatewayv2_api.main.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.data_api.invoke_arn
  payload_format_version = "2.0"
}

# Strategies API Integration
resource "aws_apigatewayv2_integration" "strategies" {
  api_id                 = aws_apigatewayv2_api.main.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.strategies_api.invoke_arn
  payload_format_version = "2.0"
}

# Scanner API Integration
resource "aws_apigatewayv2_integration" "scanner" {
  api_id                 = aws_apigatewayv2_api.main.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.scanner_api.invoke_arn
  payload_format_version = "2.0"
}

# ============================================================================
# Routes (Protected by JWT Authorizer)
# ============================================================================

# Trades routes (unauthenticated for initial deployment)
resource "aws_apigatewayv2_route" "trades" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "ANY /trades/{proxy+}"
  target    = "integrations/${aws_apigatewayv2_integration.trades.id}"
  # NOTE: Add authorization_type = "JWT" and authorizer_id when Cloudflare Access is configured
}

# Data routes
resource "aws_apigatewayv2_route" "data" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "ANY /data/{proxy+}"
  target    = "integrations/${aws_apigatewayv2_integration.data.id}"
}

# Strategies routes
resource "aws_apigatewayv2_route" "strategies" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "ANY /strategies/{proxy+}"
  target    = "integrations/${aws_apigatewayv2_integration.strategies.id}"
}

# Scanner routes
resource "aws_apigatewayv2_route" "scanner" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "ANY /scanner/{proxy+}"
  target    = "integrations/${aws_apigatewayv2_integration.scanner.id}"
}

# Health check route (unauthenticated)
resource "aws_apigatewayv2_route" "health" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.data.id}"
  # No authorizer - public health check
}

# ============================================================================
# Lambda Permissions (Allow API Gateway to invoke)
# ============================================================================

resource "aws_lambda_permission" "trades_api" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.trades_api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.main.execution_arn}/*/*"
}

resource "aws_lambda_permission" "data_api" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.data_api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.main.execution_arn}/*/*"
}

resource "aws_lambda_permission" "strategies_api" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.strategies_api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.main.execution_arn}/*/*"
}

resource "aws_lambda_permission" "scanner_api" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.scanner_api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.main.execution_arn}/*/*"
}
