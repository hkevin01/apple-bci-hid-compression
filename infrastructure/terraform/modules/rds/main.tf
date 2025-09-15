# RDS Module - infrastructure/terraform/modules/rds/main.tf

variable "db_identifier" {
  description = "RDS instance identifier"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_ids" {
  description = "Database subnet IDs"
  type        = list(string)
}

# Random password for database
resource "random_password" "db_password" {
  length  = 32
  special = true
}

# Database subnet group
resource "aws_db_subnet_group" "main" {
  name       = "${var.db_identifier}-subnet-group"
  subnet_ids = var.subnet_ids

  tags = {
    Name = "${var.db_identifier}-subnet-group"
  }
}

# Security group for RDS
resource "aws_security_group" "rds" {
  name_prefix = "${var.db_identifier}-rds-"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.db_identifier}-rds-sg"
  }
}

# KMS Key for RDS encryption
resource "aws_kms_key" "rds" {
  description             = "RDS encryption key for ${var.db_identifier}"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = {
    Name = "${var.db_identifier}-encryption-key"
  }
}

resource "aws_kms_alias" "rds" {
  name          = "alias/${var.db_identifier}-rds-key"
  target_key_id = aws_kms_key.rds.key_id
}

# RDS Instance
resource "aws_db_instance" "main" {
  identifier = var.db_identifier

  # Engine settings
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.r6g.large"

  # Storage settings
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type          = "gp3"
  storage_encrypted     = true
  kms_key_id           = aws_kms_key.rds.arn

  # Database settings
  db_name  = "bcicompression"
  username = "bciuser"
  password = random_password.db_password.result

  # Network settings
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false

  # Backup settings
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  # High availability
  multi_az = true

  # Monitoring
  monitoring_interval          = 60
  monitoring_role_arn         = aws_iam_role.rds_monitoring.arn
  performance_insights_enabled = true

  # Security
  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "${var.db_identifier}-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"

  # Enable logging
  enabled_cloudwatch_logs_exports = ["postgresql"]

  tags = {
    Name = var.db_identifier
  }
}

# IAM role for RDS monitoring
resource "aws_iam_role" "rds_monitoring" {
  name = "${var.db_identifier}-rds-monitoring-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# Store database password in AWS Secrets Manager
resource "aws_secretsmanager_secret" "db_password" {
  name        = "${var.db_identifier}-database-password"
  description = "Database password for ${var.db_identifier}"

  tags = {
    Name = "${var.db_identifier}-database-password"
  }
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id = aws_secretsmanager_secret.db_password.id
  secret_string = jsonencode({
    username = aws_db_instance.main.username
    password = random_password.db_password.result
    host     = aws_db_instance.main.endpoint
    port     = aws_db_instance.main.port
    dbname   = aws_db_instance.main.db_name
  })
}

# CloudWatch Log Group for RDS logs
resource "aws_cloudwatch_log_group" "rds" {
  name              = "/aws/rds/instance/${var.db_identifier}/postgresql"
  retention_in_days = 30

  tags = {
    Name = "${var.db_identifier}-postgresql-logs"
  }
}

# Outputs
output "db_instance_id" {
  value = aws_db_instance.main.id
}

output "db_instance_endpoint" {
  value = aws_db_instance.main.endpoint
}

output "db_instance_port" {
  value = aws_db_instance.main.port
}

output "db_instance_name" {
  value = aws_db_instance.main.db_name
}

output "db_instance_username" {
  value = aws_db_instance.main.username
}

output "secret_manager_secret_name" {
  value = aws_secretsmanager_secret.db_password.name
}
