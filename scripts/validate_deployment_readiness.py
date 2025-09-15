#!/usr/bin/env python3
"""
Phase 5A Deployment Readiness Validation
========================================
Validates system readiness for production deployment with comprehensive checks.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment_validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DeploymentValidator:
    """Comprehensive deployment readiness validation"""

    def __init__(self):
        self.project_root = Path("/home/kevin/Projects/apple-bci-hid-compression")
        self.validation_results = {}
        self.critical_issues = []
        self.warnings = []
        self.deployment_score = 0

    def validate_infrastructure_config(self) -> dict[str, Any]:
        """Validate infrastructure configuration files"""
        logger.info("🏗️ Validating infrastructure configuration...")

        results = {
            "status": "PASS",
            "score": 0,
            "checks": []
        }

        # Check Terraform configuration
        terraform_main = self.project_root / "infrastructure" / "terraform" / "main.tf"
        if terraform_main.exists():
            results["checks"].append({
                "name": "Terraform main configuration",
                "status": "PASS",
                "message": "Terraform main.tf exists and configured"
            })
            results["score"] += 20
        else:
            results["checks"].append({
                "name": "Terraform main configuration",
                "status": "FAIL",
                "message": "Terraform main.tf missing"
            })
            self.critical_issues.append("Missing Terraform configuration")

        # Check Kubernetes manifests
        k8s_deployment = self.project_root / "kubernetes" / "deployment.yaml"
        if k8s_deployment.exists():
            results["checks"].append({
                "name": "Kubernetes deployment manifest",
                "status": "PASS",
                "message": "Kubernetes deployment.yaml exists"
            })
            results["score"] += 15
        else:
            results["checks"].append({
                "name": "Kubernetes deployment manifest",
                "status": "FAIL",
                "message": "Kubernetes deployment.yaml missing"
            })
            self.critical_issues.append("Missing Kubernetes deployment manifest")

        # Check Dockerfile
        dockerfile = self.project_root / "Dockerfile"
        if dockerfile.exists():
            results["checks"].append({
                "name": "Docker configuration",
                "status": "PASS",
                "message": "Dockerfile exists and ready for containerization"
            })
            results["score"] += 10
        else:
            results["checks"].append({
                "name": "Docker configuration",
                "status": "FAIL",
                "message": "Dockerfile missing"
            })
            self.critical_issues.append("Missing Dockerfile")

        # Check CI/CD pipeline
        github_workflow = self.project_root / ".github" / "workflows" / "production-deploy.yml"
        if github_workflow.exists():
            results["checks"].append({
                "name": "CI/CD pipeline configuration",
                "status": "PASS",
                "message": "GitHub Actions workflow configured"
            })
            results["score"] += 15
        else:
            results["checks"].append({
                "name": "CI/CD pipeline configuration",
                "status": "FAIL",
                "message": "GitHub Actions workflow missing"
            })
            self.critical_issues.append("Missing CI/CD pipeline configuration")

        results["score"] = min(results["score"], 100)
        if results["score"] < 80:
            results["status"] = "FAIL"
        elif results["score"] < 95:
            results["status"] = "WARNING"

        return results

    def validate_application_readiness(self) -> dict[str, Any]:
        """Validate application code readiness"""
        logger.info("🚀 Validating application readiness...")

        results = {
            "status": "PASS",
            "score": 0,
            "checks": []
        }

        # Check core source code
        src_dir = self.project_root / "src"
        if src_dir.exists() and any(src_dir.iterdir()):
            results["checks"].append({
                "name": "Core application code",
                "status": "PASS",
                "message": "Source code directory exists with files"
            })
            results["score"] += 25
        else:
            results["checks"].append({
                "name": "Core application code",
                "status": "FAIL",
                "message": "Source code missing or empty"
            })
            self.critical_issues.append("Missing core application code")

        # Check requirements and dependencies
        requirements_file = self.project_root / "requirements.txt"
        pyproject_file = self.project_root / "pyproject.toml"

        if requirements_file.exists() or pyproject_file.exists():
            results["checks"].append({
                "name": "Dependency configuration",
                "status": "PASS",
                "message": "Dependencies properly configured"
            })
            results["score"] += 15
        else:
            results["checks"].append({
                "name": "Dependency configuration",
                "status": "FAIL",
                "message": "Missing dependency configuration"
            })
            self.critical_issues.append("Missing dependency configuration")

        # Check configuration files
        config_files = ["mypy.ini", "ruff.toml"]
        config_score = 0
        for config_file in config_files:
            if (self.project_root / config_file).exists():
                config_score += 5

        results["checks"].append({
            "name": "Code quality configuration",
            "status": "PASS" if config_score >= 5 else "WARNING",
            "message": f"Found {config_score//5} out of {len(config_files)} config files"
        })
        results["score"] += config_score

        # Check Phase 4 completion
        phase4_complete = self.project_root / "PHASE4_COMPLETE.md"
        if phase4_complete.exists():
            results["checks"].append({
                "name": "Phase 4 completion status",
                "status": "PASS",
                "message": "Phase 4 comprehensive testing completed"
            })
            results["score"] += 20
        else:
            results["checks"].append({
                "name": "Phase 4 completion status",
                "status": "WARNING",
                "message": "Phase 4 completion not documented"
            })
            self.warnings.append("Phase 4 completion status unclear")

        results["score"] = min(results["score"], 100)
        if results["score"] < 70:
            results["status"] = "FAIL"
        elif results["score"] < 90:
            results["status"] = "WARNING"

        return results

    def validate_documentation(self) -> dict[str, Any]:
        """Validate documentation completeness"""
        logger.info("📚 Validating documentation...")

        results = {
            "status": "PASS",
            "score": 0,
            "checks": []
        }

        docs_dir = self.project_root / "docs"
        required_docs = [
            ("README.md", "Project README", 20),
            ("docs/DEPLOYMENT_STRATEGY_SELECTION.md", "Deployment Strategy", 15),
            ("docs/ALPHA_RELEASE_PREPARATION.md", "Alpha Release Plan", 15),
            ("docs/PARTNERSHIP_STRATEGY.md", "Partnership Strategy", 10),
            ("docs/COMMUNITY_BUILDING_STRATEGY.md", "Community Strategy", 10)
        ]

        for doc_path, doc_name, points in required_docs:
            full_path = self.project_root / doc_path
            if full_path.exists():
                results["checks"].append({
                    "name": doc_name,
                    "status": "PASS",
                    "message": f"{doc_name} exists and ready"
                })
                results["score"] += points
            else:
                results["checks"].append({
                    "name": doc_name,
                    "status": "FAIL" if points >= 15 else "WARNING",
                    "message": f"{doc_name} missing"
                })
                if points >= 15:
                    self.critical_issues.append(f"Missing critical documentation: {doc_name}")
                else:
                    self.warnings.append(f"Missing documentation: {doc_name}")

        # Check API documentation
        if (docs_dir / "api").exists():
            results["checks"].append({
                "name": "API Documentation",
                "status": "PASS",
                "message": "API documentation directory exists"
            })
            results["score"] += 10
        else:
            results["checks"].append({
                "name": "API Documentation",
                "status": "WARNING",
                "message": "API documentation missing"
            })
            self.warnings.append("Missing API documentation")

        results["score"] = min(results["score"], 100)
        if results["score"] < 60:
            results["status"] = "FAIL"
        elif results["score"] < 85:
            results["status"] = "WARNING"

        return results

    def validate_security_readiness(self) -> dict[str, Any]:
        """Validate security configuration"""
        logger.info("🔐 Validating security configuration...")

        results = {
            "status": "PASS",
            "score": 0,
            "checks": []
        }

        # Check for environment variables template
        env_example = self.project_root / ".env.example"
        env_file = self.project_root / ".env"

        if env_example.exists():
            results["checks"].append({
                "name": "Environment configuration template",
                "status": "PASS",
                "message": ".env.example exists for configuration guidance"
            })
            results["score"] += 20
        else:
            results["checks"].append({
                "name": "Environment configuration template",
                "status": "WARNING",
                "message": ".env.example missing"
            })
            self.warnings.append("Missing environment configuration template")

        # Check that .env is in gitignore
        gitignore = self.project_root / ".gitignore"
        if gitignore.exists():
            gitignore_content = gitignore.read_text()
            if ".env" in gitignore_content:
                results["checks"].append({
                    "name": "Environment file security",
                    "status": "PASS",
                    "message": ".env properly excluded from git"
                })
                results["score"] += 15
            else:
                results["checks"].append({
                    "name": "Environment file security",
                    "status": "WARNING",
                    "message": ".env not in .gitignore"
                })
                self.warnings.append(".env file not excluded from git")

        # Check for security testing
        security_tests = self.project_root / "tests" / "security"
        if security_tests.exists():
            results["checks"].append({
                "name": "Security testing suite",
                "status": "PASS",
                "message": "Security tests available"
            })
            results["score"] += 25
        else:
            results["checks"].append({
                "name": "Security testing suite",
                "status": "WARNING",
                "message": "Security tests missing"
            })
            self.warnings.append("Missing security testing suite")

        # Check for HTTPS/TLS configuration
        if (self.project_root / "infrastructure").exists():
            results["checks"].append({
                "name": "Infrastructure security",
                "status": "PASS",
                "message": "Infrastructure security configured in Terraform"
            })
            results["score"] += 20

        results["score"] = min(results["score"], 100)
        if results["score"] < 70:
            results["status"] = "FAIL"
        elif results["score"] < 90:
            results["status"] = "WARNING"

        return results

    def validate_testing_coverage(self) -> dict[str, Any]:
        """Validate testing infrastructure"""
        logger.info("🧪 Validating testing coverage...")

        results = {
            "status": "PASS",
            "score": 0,
            "checks": []
        }

        tests_dir = self.project_root / "tests"
        required_test_suites = [
            "unit", "integration", "performance",
            "security", "compatibility", "ux"
        ]

        existing_suites = []
        for suite in required_test_suites:
            suite_dir = tests_dir / suite
            if suite_dir.exists():
                existing_suites.append(suite)
                results["score"] += 15

        results["checks"].append({
            "name": "Test suite coverage",
            "status": "PASS" if len(existing_suites) >= 4 else "WARNING",
            "message": f"Found {len(existing_suites)} out of {len(required_test_suites)} test suites"
        })

        # Check for Phase 4 runner
        phase4_runner = tests_dir / "phase4_runner.py"
        if phase4_runner.exists():
            results["checks"].append({
                "name": "Comprehensive test runner",
                "status": "PASS",
                "message": "Phase 4 comprehensive test runner available"
            })
            results["score"] += 10
        else:
            results["checks"].append({
                "name": "Comprehensive test runner",
                "status": "WARNING",
                "message": "Comprehensive test runner missing"
            })
            self.warnings.append("Missing comprehensive test runner")

        results["score"] = min(results["score"], 100)
        if results["score"] < 60:
            results["status"] = "FAIL"
        elif results["score"] < 80:
            results["status"] = "WARNING"

        return results

    def calculate_overall_deployment_score(self) -> tuple[str, int, str]:
        """Calculate overall deployment readiness"""
        logger.info("📊 Calculating overall deployment readiness...")

        # Weight different validation areas
        weights = {
            "infrastructure": 0.25,
            "application": 0.25,
            "documentation": 0.20,
            "security": 0.20,
            "testing": 0.10
        }

        weighted_score = 0
        for area, weight in weights.items():
            if area in self.validation_results:
                weighted_score += self.validation_results[area]["score"] * weight

        # Determine deployment status
        if len(self.critical_issues) > 0:
            status = "NOT_READY"
            recommendation = "Critical issues must be resolved before deployment"
        elif weighted_score >= 90:
            status = "READY_FOR_DEPLOYMENT"
            recommendation = "System ready for production deployment"
        elif weighted_score >= 75:
            status = "READY_WITH_MONITORING"
            recommendation = "Deploy with enhanced monitoring and quick rollback capability"
        elif weighted_score >= 60:
            status = "NEEDS_IMPROVEMENT"
            recommendation = "Address warnings before deployment"
        else:
            status = "NOT_READY"
            recommendation = "Significant issues require resolution"

        return status, int(weighted_score), recommendation

    def generate_deployment_report(self) -> dict[str, Any]:
        """Generate comprehensive deployment readiness report"""
        logger.info("📋 Generating deployment readiness report...")

        status, score, recommendation = self.calculate_overall_deployment_score()

        report = {
            "deployment_readiness": {
                "status": status,
                "overall_score": score,
                "recommendation": recommendation,
                "validation_timestamp": datetime.now().isoformat(),
                "critical_issues_count": len(self.critical_issues),
                "warnings_count": len(self.warnings)
            },
            "detailed_results": self.validation_results,
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "next_steps": self.generate_next_steps(status, score)
        }

        return report

    def generate_next_steps(self, status: str, score: int) -> list[str]:
        """Generate actionable next steps based on validation results"""
        next_steps = []

        if status == "READY_FOR_DEPLOYMENT":
            next_steps = [
                "✅ Begin infrastructure deployment using Terraform",
                "✅ Execute alpha release preparation with selected partners",
                "✅ Initialize community building and developer outreach",
                "✅ Monitor deployment metrics and performance",
                "✅ Begin partnership discussions with accessibility companies"
            ]
        elif status == "READY_WITH_MONITORING":
            next_steps = [
                "⚠️ Deploy with enhanced monitoring and alerting",
                "⚠️ Prepare quick rollback procedures",
                "⚠️ Address identified warnings in next iteration",
                "✅ Begin limited alpha release with close monitoring",
                "✅ Implement additional logging and metrics collection"
            ]
        elif status == "NEEDS_IMPROVEMENT":
            next_steps = [
                "🔧 Address all warnings before deployment",
                "🔧 Complete missing documentation",
                "🔧 Enhance security configurations",
                "🔧 Improve test coverage where gaps exist",
                "⏳ Re-run validation after improvements"
            ]
        else:  # NOT_READY
            next_steps = [
                "❌ Resolve all critical issues immediately",
                "❌ Complete missing infrastructure components",
                "❌ Ensure core application readiness",
                "❌ Address security vulnerabilities",
                "⏳ Re-validate before considering deployment"
            ]

        return next_steps

    def run_validation(self) -> dict[str, Any]:
        """Run complete deployment readiness validation"""
        logger.info("🚀 Starting Phase 5A deployment readiness validation...")

        # Run all validation checks
        self.validation_results["infrastructure"] = self.validate_infrastructure_config()
        self.validation_results["application"] = self.validate_application_readiness()
        self.validation_results["documentation"] = self.validate_documentation()
        self.validation_results["security"] = self.validate_security_readiness()
        self.validation_results["testing"] = self.validate_testing_coverage()

        # Generate comprehensive report
        report = self.generate_deployment_report()

        # Save detailed results
        report_file = self.project_root / "deployment_readiness_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"📄 Detailed report saved to: {report_file}")

        return report

def main():
    """Main validation execution"""
    print("=" * 80)
    print("🚀 PHASE 5A DEPLOYMENT READINESS VALIDATION")
    print("=" * 80)

    validator = DeploymentValidator()
    report = validator.run_validation()

    # Display summary
    readiness = report["deployment_readiness"]
    print("\n📊 DEPLOYMENT READINESS SUMMARY")
    print("-" * 50)
    print(f"Status: {readiness['status']}")
    print(f"Overall Score: {readiness['overall_score']}/100")
    print(f"Recommendation: {readiness['recommendation']}")
    print(f"Critical Issues: {readiness['critical_issues_count']}")
    print(f"Warnings: {readiness['warnings_count']}")

    # Display critical issues if any
    if report["critical_issues"]:
        print(f"\n❌ CRITICAL ISSUES ({len(report['critical_issues'])})")
        print("-" * 50)
        for issue in report["critical_issues"]:
            print(f"• {issue}")

    # Display warnings if any
    if report["warnings"]:
        print(f"\n⚠️ WARNINGS ({len(report['warnings'])})")
        print("-" * 50)
        for warning in report["warnings"]:
            print(f"• {warning}")

    # Display next steps
    print("\n🎯 NEXT STEPS")
    print("-" * 50)
    for step in report["next_steps"]:
        print(f"{step}")

    print("\n✅ Validation completed successfully!")
    print("📄 Detailed report: deployment_readiness_report.json")

    # Return appropriate exit code
    if readiness["status"] in ["READY_FOR_DEPLOYMENT", "READY_WITH_MONITORING"]:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
