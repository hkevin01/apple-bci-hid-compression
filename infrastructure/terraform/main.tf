# Production Infrastructure Setup - AWS Multi-Region

## Infrastructure Architecture

### Multi-Region Deployment Strategy

**Primary Regions:**
- **US East (N. Virginia)** - us-east-1: Primary region for North American users
- **EU West (Ireland)** - eu-west-1: Primary region for European users
- **Asia Pacific (Tokyo)** - ap-northeast-1: Primary region for Asian users

**Failover Regions:**
- **US West (Oregon)** - us-west-2: Disaster recovery for North America
- **EU Central (Frankfurt)** - eu-central-1: Disaster recovery for Europe

## Core Infrastructure Components

### 1. Container Orchestration
```yaml
# Kubernetes (EKS) Configuration
apiVersion: v1
kind: Namespace
metadata:
  name: bci-hid-compression
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bci-compression-api
  namespace: bci-hid-compression
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bci-compression-api
  template:
    metadata:
      labels:
        app: bci-compression-api
    spec:
      containers:
      - name: api
        image: bci-hid-compression:latest
        ports:
        - containerPort: 8080
        env:
        - name: REGION
          value: "${AWS_REGION}"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### 2. API Gateway & Load Balancing
```yaml
# Application Load Balancer Configuration
apiVersion: v1
kind: Service
metadata:
  name: bci-compression-service
  namespace: bci-hid-compression
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-scheme: "internet-facing"
spec:
  type: LoadBalancer
  ports:
  - port: 443
    targetPort: 8080
    protocol: TCP
  selector:
    app: bci-compression-api
```

### 3. Database & Storage
```yaml
# RDS PostgreSQL Multi-AZ Configuration
Resources:
  BCICompressionDB:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceIdentifier: bci-compression-db
      DBInstanceClass: db.r6g.large
      Engine: postgres
      EngineVersion: "15.4"
      MasterUsername: bciuser
      MasterUserPassword: !Ref DatabasePassword
      AllocatedStorage: 100
      StorageType: gp3
      MultiAZ: true
      BackupRetentionPeriod: 30
      DeletionProtection: true
      StorageEncrypted: true
```

## Terraform Infrastructure as Code

Let me create the Terraform configuration for our infrastructure:

<function_calls>
<invoke name="create_file">
<parameter name="content"># Terraform Configuration - AWS Multi-Region Infrastructure
# File: infrastructure/terraform/main.tf

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "bci-hid-compression-terraform-state"
    key    = "infrastructure/terraform.tfstate"
    region = "us-east-1"
  }
}

# Provider configurations for multi-region
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
}

provider "aws" {
  alias  = "eu_west_1"
  region = "eu-west-1"
}

provider "aws" {
  alias  = "ap_northeast_1"
  region = "ap-northeast-1"
}

# Variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "bci-hid-compression"
}

# VPC and Networking
module "vpc_us_east_1" {
  source = "./modules/vpc"
  providers = {
    aws = aws.us_east_1
  }

  region       = "us-east-1"
  project_name = var.project_name
  environment  = var.environment
}

module "vpc_eu_west_1" {
  source = "./modules/vpc"
  providers = {
    aws = aws.eu_west_1
  }

  region       = "eu-west-1"
  project_name = var.project_name
  environment  = var.environment
}

module "vpc_ap_northeast_1" {
  source = "./modules/vpc"
  providers = {
    aws = aws.ap_northeast_1
  }

  region       = "ap-northeast-1"
  project_name = var.project_name
  environment  = var.environment
}

# EKS Clusters
module "eks_us_east_1" {
  source = "./modules/eks"
  providers = {
    aws = aws.us_east_1
  }

  cluster_name = "${var.project_name}-us-east-1"
  vpc_id       = module.vpc_us_east_1.vpc_id
  subnet_ids   = module.vpc_us_east_1.private_subnet_ids
}

module "eks_eu_west_1" {
  source = "./modules/eks"
  providers = {
    aws = aws.eu_west_1
  }

  cluster_name = "${var.project_name}-eu-west-1"
  vpc_id       = module.vpc_eu_west_1.vpc_id
  subnet_ids   = module.vpc_eu_west_1.private_subnet_ids
}

module "eks_ap_northeast_1" {
  source = "./modules/eks"
  providers = {
    aws = aws.ap_northeast_1
  }

  cluster_name = "${var.project_name}-ap-northeast-1"
  vpc_id       = module.vpc_ap_northeast_1.vpc_id
  subnet_ids   = module.vpc_ap_northeast_1.private_subnet_ids
}

# RDS Databases
module "rds_us_east_1" {
  source = "./modules/rds"
  providers = {
    aws = aws.us_east_1
  }

  db_identifier = "${var.project_name}-db-us-east-1"
  vpc_id        = module.vpc_us_east_1.vpc_id
  subnet_ids    = module.vpc_us_east_1.database_subnet_ids
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "api_distribution" {
  provider = aws.us_east_1

  origin {
    domain_name = module.eks_us_east_1.cluster_endpoint
    origin_id   = "EKS-US-East-1"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  origin {
    domain_name = module.eks_eu_west_1.cluster_endpoint
    origin_id   = "EKS-EU-West-1"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  enabled = true

  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "EKS-US-East-1"
    compress               = true
    viewer_protocol_policy = "redirect-to-https"

    forwarded_values {
      query_string = true
      headers      = ["Authorization", "Content-Type"]

      cookies {
        forward = "none"
      }
    }
  }

  # Geographic routing for optimal performance
  ordered_cache_behavior {
    path_pattern     = "/api/v1/*"
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "EKS-US-East-1"

    forwarded_values {
      query_string = true
      headers      = ["*"]

      cookies {
        forward = "none"
      }
    }

    viewer_protocol_policy = "https-only"
    min_ttl                = 0
    default_ttl            = 0
    max_ttl                = 0
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate.api_cert.arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  tags = {
    Name        = "${var.project_name}-api-distribution"
    Environment = var.environment
  }
}

# SSL Certificate
resource "aws_acm_certificate" "api_cert" {
  provider    = aws.us_east_1
  domain_name = "api.bci-hid-compression.com"

  subject_alternative_names = [
    "*.api.bci-hid-compression.com",
    "api-us.bci-hid-compression.com",
    "api-eu.bci-hid-compression.com",
    "api-ap.bci-hid-compression.com"
  ]

  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name        = "${var.project_name}-api-certificate"
    Environment = var.environment
  }
}

# Outputs
output "cloudfront_distribution_id" {
  value = aws_cloudfront_distribution.api_distribution.id
}

output "cloudfront_domain_name" {
  value = aws_cloudfront_distribution.api_distribution.domain_name
}

output "eks_cluster_endpoints" {
  value = {
    us_east_1     = module.eks_us_east_1.cluster_endpoint
    eu_west_1     = module.eks_eu_west_1.cluster_endpoint
    ap_northeast_1 = module.eks_ap_northeast_1.cluster_endpoint
  }
}
