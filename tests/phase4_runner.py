"""Phase 4 comprehensive testing suite runner for Apple BCI-HID system."""

import asyncio
import json
import logging
import os
import sys
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

try:  # Optional numpy for seeding; ignore if unavailable
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore

# Add src (project source) to path for core module imports
_SRC_PATH = os.path.join(os.path.dirname(__file__), '..', 'src')
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

# Relative imports (requires tests package with __init__.py)
from .compatibility.compatibility_testing import CompatibilityTestingSuite
from .integration.real_world_testing import RealWorldTestingSuite
from .performance.automated_benchmarks import AutomatedBenchmarkingSuite
from .security.security_testing import SecurityTestingSuite
from .ux.user_experience_testing import UXTestingSuite


@dataclass
class Phase4TestResult:
    """Phase 4 comprehensive test result."""
    suite_name: str
    start_time: datetime
    end_time: datetime
    success: bool
    score: float  # 0.0 - 1.0
    details: Dict[str, Any]
    recommendations: List[str]
    critical_issues: List[str]


@dataclass
class Phase4Summary:
    """Phase 4 testing summary."""
    total_duration_minutes: float
    suites_run: int
    suites_passed: int
    suites_failed: int
    overall_score: float
    readiness_status: str
    critical_blockers: int
    recommendations: List[str]
    results: List[Phase4TestResult] = field(default_factory=list)


class Phase4TestRunner:
    """Comprehensive Phase 4 testing orchestrator."""

    def __init__(self):
        self.results: List[Phase4TestResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('phase4_testing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Optional deterministic seeding
        seed_env = os.getenv("PHASE4_TEST_SEED")
        if seed_env:
            try:
                seed_val = int(seed_env)
                random.seed(seed_val)
                if _np is not None:
                    _np.random.seed(seed_val)
                self.logger.info(
                    "Deterministic seeding enabled seed=%d", seed_val
                )
            except ValueError:
                self.logger.warning(
                    "PHASE4_TEST_SEED provided but not an int; ignoring"
                )

        print("🚀 Phase 4 Comprehensive Testing Suite")
        print("=" * 60)
        print("   Performance Testing")
        print("   Real-World Integration Testing")
        print("   Security Testing")
        print("   User Experience Testing")
        print("   Compatibility Testing")
        print("=" * 60)

    async def run_all_phase4_tests(self) -> Phase4Summary:
        """Run all Phase 4 testing suites."""
        self.start_time = datetime.now()

        print("\n🚀 Starting Phase 4 Comprehensive Testing")
        print(
            f"   Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print("=" * 60)

        test_suites = [
            ("Performance Testing", self._run_performance_tests),
            ("Real-World Integration Testing", self._run_integration_tests),
            ("Security Testing", self._run_security_tests),
            ("User Experience Testing", self._run_ux_tests),
            ("Compatibility Testing", self._run_compatibility_tests),
        ]

        for suite_name, test_function in test_suites:
            try:
                print(f"\n🧪 Running {suite_name}")
                print("-" * 40)
                suite_start = datetime.now()
                result = await test_function()
                suite_end = datetime.now()

                suite_result = Phase4TestResult(
                    suite_name=suite_name,
                    start_time=suite_start,
                    end_time=suite_end,
                    success=result["success"],
                    score=result["score"],
                    details=result["details"],
                    recommendations=result["recommendations"],
                    critical_issues=result["critical_issues"],
                )
                self.results.append(suite_result)

                duration = (suite_end - suite_start).total_seconds() / 60
                status = "✅ PASSED" if result["success"] else "❌ FAILED"
                print(f"\n{status} - {suite_name}")
                print(f"   Duration: {duration:.1f} minutes")
                print(f"   Score: {result['score']*100:.1f}%")
                if result["critical_issues"]:
                    issue_count = len(result["critical_issues"])
                    print(f"   🚨 Critical Issues: {issue_count}")
                    for issue in result["critical_issues"][:3]:
                        print(f"     • {issue}")
            except Exception as e:  # pragma: no cover - protective diagnostic
                print(f"\n💥 {suite_name} FAILED: {e}")
                self.logger.error("%s failed: %s", suite_name, e)
                suite_result = Phase4TestResult(
                    suite_name=suite_name,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    success=False,
                    score=0.0,
                    details={"error": str(e)},
                    recommendations=[f"Fix {suite_name} execution issues"],
                    critical_issues=[f"{suite_name} failed to execute: {e}"],
                )
                self.results.append(suite_result)

        self.end_time = datetime.now()
        summary = self._generate_phase4_summary()
        self._print_phase4_summary(summary)
        self._save_phase4_results(summary)
        return summary

    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance testing suite."""
        try:
            perf_suite = AutomatedBenchmarkingSuite()
            report = await perf_suite.run_comprehensive_benchmarks()

            # Evaluate results
            metrics = report.get('performance_metrics', {})
            avg_latency = metrics.get(
                'avg_processing_latency_ms', float('inf')
            )
            memory_efficiency = metrics.get('memory_efficiency_score', 0.0)
            throughput = metrics.get('avg_throughput_signals_per_sec', 0)

            # Calculate score based on performance metrics
            latency_score = (
                max(0, 1.0 - (avg_latency - 50) / 100)
                if avg_latency < 150
                else 0.0
            )
            memory_score = memory_efficiency
            # Target 1000 signals/sec
            throughput_score = min(1.0, throughput / 1000)

            overall_score = (
                latency_score + memory_score + throughput_score
            ) / 3

            # Determine success
            success = (
                avg_latency <= 100 and  # Under 100ms latency
                memory_efficiency >= 0.8 and  # 80%+ memory efficiency
                throughput >= 500  # At least 500 signals/sec
            )

            # Extract recommendations and critical issues
            recommendations = report.get('recommendations', [])
            critical_issues = []

            if avg_latency > 150:
                critical_issues.append(
                    f"High latency: {avg_latency:.1f}ms (target <100ms)"
                )
            if memory_efficiency < 0.6:
                critical_issues.append(
                    f"Poor memory efficiency: {memory_efficiency*100:.1f}%"
                )
            if throughput < 100:
                critical_issues.append(
                    f"Low throughput: {throughput:.1f} signals/sec"
                )

            return {
                'success': success,
                'score': overall_score,
                'details': report,
                'recommendations': recommendations,
                'critical_issues': critical_issues
            }

        except Exception as e:
            return {
                'success': False,
                'score': 0.0,
                'details': {'error': str(e)},
                'recommendations': ['Fix performance testing infrastructure'],
                'critical_issues': [f'Performance testing failed: {e}']
            }

    async def _run_integration_tests(self) -> Dict[str, Any]:
        """Run real-world integration testing suite."""
        try:
            integration_suite = RealWorldTestingSuite()
            report = await integration_suite.run_all_tests()

            # Evaluate results
            summary = report.get('test_summary', {})
            metrics = report.get('performance_metrics', {})

            success_rate = summary.get('success_rate', 0.0)
            avg_latency = metrics.get('avg_latency_ms', float('inf'))
            avg_satisfaction = metrics.get('avg_user_satisfaction', 0.0)

            # Calculate score
            success_score = success_rate
            latency_score = (
                max(0, 1.0 - (avg_latency - 50) / 100)
                if avg_latency < 200
                else 0.0
            )
            # Convert from 0-10 to 0-1
            satisfaction_score = avg_satisfaction / 10.0

            overall_score = (
                success_score + latency_score + satisfaction_score
            ) / 3

            # Determine success
            success = (
                success_rate >= 0.8 and  # 80%+ success rate
                avg_latency <= 150 and   # Under 150ms for real-world
                avg_satisfaction >= 7.0   # 7/10 user satisfaction
            )

            # Extract recommendations and critical issues
            recommendations = report.get('recommendations', [])
            critical_issues = []

            if success_rate < 0.7:
                critical_issues.append(
                    f"Low success rate: {success_rate*100:.1f}%"
                )
            if avg_latency > 200:
                critical_issues.append(
                    f"High real-world latency: {avg_latency:.1f}ms"
                )
            if avg_satisfaction < 6.0:
                critical_issues.append(
                    f"Poor user satisfaction: {avg_satisfaction:.1f}/10"
                )

            return {
                'success': success,
                'score': overall_score,
                'details': report,
                'recommendations': recommendations,
                'critical_issues': critical_issues
            }

        except Exception as e:
            return {
                'success': False,
                'score': 0.0,
                'details': {'error': str(e)},
                'recommendations': ['Fix integration testing infrastructure'],
                'critical_issues': [f'Integration testing failed: {e}']
            }

    async def _run_security_tests(self) -> Dict[str, Any]:
        """Run security testing suite."""
        try:
            security_suite = SecurityTestingSuite()
            report = await security_suite.run_all_security_tests()

            # Evaluate results
            summary = report.get('test_summary', {})
            metrics = report.get('security_metrics', {})

            success_rate = summary.get('success_rate', 0.0)
            # Convert 0-10 to 0-1
            security_score = metrics.get('security_score', 0.0) / 10.0
            critical_vulns = summary.get('critical_vulnerabilities', 0)

            # Calculate overall score
            overall_score = (success_rate + security_score) / 2

            # Determine success (strict security requirements)
            success = (
                success_rate >= 0.9 and    # 90%+ security tests pass
                security_score >= 0.8 and  # 8/10 security score
                critical_vulns == 0        # No critical vulnerabilities
            )

            # Extract recommendations and critical issues
            recommendations = report.get('security_recommendations', [])
            critical_issues = []

            if critical_vulns > 0:
                critical_issues.append(
                    f"Critical vulnerabilities found: {critical_vulns}"
                )
            if security_score < 0.7:
                critical_issues.append(
                    f"Low security score: {security_score*10:.1f}/10"
                )
            if success_rate < 0.8:
                critical_issues.append(
                    f"Security test failures: {(1-success_rate)*100:.1f}%"
                )

            return {
                'success': success,
                'score': overall_score,
                'details': report,
                'recommendations': recommendations,
                'critical_issues': critical_issues
            }

        except Exception as e:
            return {
                'success': False,
                'score': 0.0,
                'details': {'error': str(e)},
                'recommendations': ['Fix security testing infrastructure'],
                'critical_issues': [f'Security testing failed: {e}']
            }

    async def _run_ux_tests(self) -> Dict[str, Any]:
        """Run user experience testing suite."""
        try:
            ux_suite = UXTestingSuite()
            report = await ux_suite.run_all_ux_tests()

            # Evaluate results
            summary = report.get('test_summary', {})
            metrics = report.get('overall_metrics', {})
            accessibility = report.get('accessibility_analysis', {})

            success_rate = summary.get('success_rate', 0.0)
            avg_satisfaction = metrics.get('average_user_satisfaction', 0.0)
            avg_success_rate = metrics.get('average_task_success_rate', 0.0)
            accessibility_user_perf = accessibility.get(
                'accessibility_user_performance', {}
            )
            accessibility_score = accessibility_user_perf.get(
                'avg_satisfaction', 0.0
            )

            # Calculate score
            success_score = success_rate
            satisfaction_score = avg_satisfaction / 10.0
            task_score = avg_success_rate
            accessibility_ux_score = accessibility_score / 10.0

            overall_score = (
                success_score
                + satisfaction_score
                + task_score
                + accessibility_ux_score
            ) / 4

            # Determine success
            success = (
                success_rate >= 0.8 and         # 80%+ test completion
                avg_satisfaction >= 7.0 and     # 7/10 user satisfaction
                avg_success_rate >= 0.8 and     # 80%+ task success
                accessibility_score >= 6.5  # 6.5/10 accessibility satisfaction
            )

            # Extract recommendations and critical issues
            recommendations = report.get('insights_and_recommendations', [])
            critical_issues = []

            if avg_satisfaction < 6.0:
                critical_issues.append(
                    f"Poor user satisfaction: {avg_satisfaction:.1f}/10"
                )
            if avg_success_rate < 0.7:
                critical_issues.append(
                    f"Low task success rate: {avg_success_rate*100:.1f}%"
                )
            if accessibility_score < 5.0:
                critical_issues.append(
                    f"Poor accessibility experience: {accessibility_score:.1f}/10"
                )

            return {
                'success': success,
                'score': overall_score,
                'details': report,
                'recommendations': recommendations,
                'critical_issues': critical_issues
            }

        except Exception as e:
            return {
                'success': False,
                'score': 0.0,
                'details': {'error': str(e)},
                'recommendations': ['Fix UX testing infrastructure'],
                'critical_issues': [f'UX testing failed: {e}']
            }

    async def _run_compatibility_tests(self) -> Dict[str, Any]:
        """Run compatibility testing suite."""
        try:
            compat_suite = CompatibilityTestingSuite()
            report = await compat_suite.run_all_compatibility_tests()

            # Evaluate results
            summary = report.get('test_summary', {})
            scores = report.get('compatibility_scores', {})
            performance = report.get('performance_impact_analysis', {})

            compatibility_rate = summary.get('compatibility_rate', 0.0)
            overall_compat_score = scores.get('overall', 0.0)
            critical_failures = summary.get('critical_failures', 0)
            avg_impact = performance.get('average_impact', 0.0)

            # Calculate score
            compat_score = (compatibility_rate + overall_compat_score) / 2
            impact_score = max(0, 1.0 - avg_impact)  # Lower impact is better

            overall_score = (compat_score + impact_score) / 2

            # Determine success
            success = (
                compatibility_rate >= 0.8 and  # 80%+ compatibility
                critical_failures == 0 and     # No critical failures
                avg_impact <= 0.3               # Low performance impact
            )

            # Extract recommendations and critical issues
            recommendations = report.get('recommendations', [])
            critical_issues = []

            if critical_failures > 0:
                critical_issues.append(
                    f"Critical compatibility failures: {critical_failures}"
                )
            if compatibility_rate < 0.7:
                critical_issues.append(
                    f"Low compatibility rate: {compatibility_rate*100:.1f}%"
                )
            if avg_impact > 0.5:
                critical_issues.append(
                    f"High performance impact: {avg_impact*100:.1f}%"
                )

            return {
                'success': success,
                'score': overall_score,
                'details': report,
                'recommendations': recommendations,
                'critical_issues': critical_issues
            }

        except Exception as e:
            return {
                'success': False,
                'score': 0.0,
                'details': {'error': str(e)},
                'recommendations': [
                    'Fix compatibility testing infrastructure'
                ],
                'critical_issues': [f'Compatibility testing failed: {e}']
            }

    def _generate_phase4_summary(self) -> Phase4Summary:
        """Generate comprehensive Phase 4 summary."""
        if not self.start_time or not self.end_time:
            self.end_time = datetime.now()

        duration = (self.end_time - self.start_time).total_seconds() / 60

        suites_run = len(self.results)
        suites_passed = len([r for r in self.results if r.success])
        suites_failed = suites_run - suites_passed

        # Calculate overall score
        overall_score = (
            sum(r.score for r in self.results) / suites_run if suites_run > 0 else 0.0
        )

        # Count critical blockers
        critical_blockers = sum(len(r.critical_issues) for r in self.results)

        # Determine readiness status
        if (
            suites_passed == suites_run
            and overall_score >= 0.8
            and critical_blockers == 0
        ):
            readiness_status = "READY_FOR_DEPLOYMENT"
        elif (
            suites_passed >= suites_run * 0.8
            and overall_score >= 0.7
            and critical_blockers <= 2
        ):
            readiness_status = "READY_WITH_MINOR_ISSUES"
        elif suites_passed >= suites_run * 0.6 and overall_score >= 0.6:
            readiness_status = "NEEDS_IMPROVEMENT"
        else:
            readiness_status = "NOT_READY"

        # Aggregate recommendations
        all_recommendations = []
        for result in self.results:
            all_recommendations.extend(result.recommendations)

        # Deduplicate and prioritize recommendations
        unique_recommendations = list(dict.fromkeys(all_recommendations))[:10]

        return Phase4Summary(
            total_duration_minutes=duration,
            suites_run=suites_run,
            suites_passed=suites_passed,
            suites_failed=suites_failed,
            overall_score=overall_score,
            readiness_status=readiness_status,
            critical_blockers=critical_blockers,
            recommendations=unique_recommendations,
            results=self.results,
        )

    def _print_phase4_summary(self, summary: Phase4Summary):
        """Print comprehensive Phase 4 summary (synchronous)."""
        print("\n" + "=" * 60)
        print("🚀 PHASE 4 COMPREHENSIVE TESTING SUMMARY")
        print("=" * 60)

        # Overall status
        status_emoji = {
            "READY_FOR_DEPLOYMENT": "✅",
            "READY_WITH_MINOR_ISSUES": "🟡",
            "NEEDS_IMPROVEMENT": "🟠",
            "NOT_READY": "❌"
        }.get(summary.readiness_status, "❓")

        readiness_text = summary.readiness_status.replace('_', ' ')
        print(
            f"\n{status_emoji} Deployment Readiness: {readiness_text}"
        )
        print(f"   Overall Score: {summary.overall_score*100:.1f}%")
        print(
            f"   Total Duration: {summary.total_duration_minutes:.1f} minutes"
        )

        # Test suite results
        print("\nTest Suite Results:")
        print(f"   Total Suites: {summary.suites_run}")
        print(f"   Passed: {summary.suites_passed}")
        print(f"   Failed: {summary.suites_failed}")
        success_pct = (
            (summary.suites_passed / summary.suites_run) * 100
            if summary.suites_run
            else 0.0
        )
        print(f"   Success Rate: {success_pct:.1f}%")

    # Individual suite performance
    print("\nSuite Performance:")
        for result in summary.results:
            status = "✅" if result.success else "❌"
            duration = (
                result.end_time - result.start_time
            ).total_seconds() / 60
            score_pct = result.score * 100
            print(
                f"   {status} {result.suite_name}: {score_pct:.1f}% ({duration:.1f}min)"
            )

        # Critical issues
        if summary.critical_blockers > 0:
            print(f"\n🚨 Critical Issues ({summary.critical_blockers} total):")
            for result in summary.results:
                if result.critical_issues:
                    print(f"   {result.suite_name}:")
                    for issue in result.critical_issues[:2]:  # Show first 2 per suite
                        print(f"     • {issue}")

        # Top recommendations
        if summary.recommendations:
            print(f"\n💡 Top Recommendations:")
            for i, rec in enumerate(summary.recommendations[:5], 1):
                print(f"   {i}. {rec}")

        # Deployment guidance
        print(f"\n🎯 Deployment Guidance:")
        if summary.readiness_status == "READY_FOR_DEPLOYMENT":
            print("   System is ready for production deployment.")
            print("   All tests pass with excellent scores.")
        elif summary.readiness_status == "READY_WITH_MINOR_ISSUES":
            print("   System is ready for deployment with monitoring.")
            print("   Address minor issues in next iteration.")
        elif summary.readiness_status == "NEEDS_IMPROVEMENT":
            print("   System needs improvement before deployment.")
            print("   Focus on critical issues and low-scoring areas.")
        else:
            print("   System is not ready for deployment.")
            print("   Significant issues must be resolved first.")

    def _save_phase4_results(self, summary: Phase4Summary):
        """Save comprehensive Phase 4 results (synchronous)."""
        # Convert summary to dictionary
        summary_dict = {
            'phase4_summary': {
                'total_duration_minutes': summary.total_duration_minutes,
                'suites_run': summary.suites_run,
                'suites_passed': summary.suites_passed,
                'suites_failed': summary.suites_failed,
                'overall_score': summary.overall_score,
                'readiness_status': summary.readiness_status,
                'critical_blockers': summary.critical_blockers,
                'recommendations': summary.recommendations,
                'test_timestamp': self.start_time.isoformat() if self.start_time else None
            },
            'suite_results': [
                {
                    'suite_name': r.suite_name,
                    'start_time': r.start_time.isoformat(),
                    'end_time': r.end_time.isoformat(),
                    'duration_minutes': (r.end_time - r.start_time).total_seconds() / 60,
                    'success': r.success,
                    'score': r.score,
                    'recommendations': r.recommendations,
                    'critical_issues': r.critical_issues,
                    'details': r.details
                }
                for r in summary.results
            ]
        }

        # Save to JSON file
        filename = f"phase4_comprehensive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(summary_dict, f, indent=2, default=str)

        print(f"\n📁 Phase 4 results saved to: {filename}")

        # Also save a simple status file
        status_filename = "phase4_status.json"
        status_data = {
            'last_run': self.start_time.isoformat() if self.start_time else None,
            'readiness_status': summary.readiness_status,
            'overall_score': summary.overall_score,
            'critical_blockers': summary.critical_blockers,
            'deployment_ready': summary.readiness_status in ["READY_FOR_DEPLOYMENT", "READY_WITH_MINOR_ISSUES"]
        }

        with open(status_filename, 'w') as f:
            json.dump(status_data, f, indent=2)

        print(f"📊 Status summary saved to: {status_filename}")


async def main():
    """Main Phase 4 testing function."""
    print("🚀 Apple BCI-HID Compression Bridge - Phase 4 Comprehensive Testing")
    print("=" * 80)
    print("   This will run ALL Phase 4 testing suites:")
    print("   • Performance Testing & Optimization")
    print("   • Real-World Integration Testing")
    print("   • Security Testing & Vulnerability Assessment")
    print("   • User Experience Testing")
    print("   • Compatibility Testing")
    print("=" * 80)

    # Confirm with user (comment out for automated runs)
    # response = input("\nProceed with comprehensive testing? (y/N): ")
    # if response.lower() != 'y':
    #     print("Testing cancelled.")
    #     return False

    runner = Phase4TestRunner()

    try:
        # Run all Phase 4 tests
        summary = await runner.run_all_phase4_tests()

        # Final status
        if summary.readiness_status in ["READY_FOR_DEPLOYMENT", "READY_WITH_MINOR_ISSUES"]:
            print(f"\n🎉 Phase 4 Testing COMPLETED SUCCESSFULLY!")
            print(f"   Overall Score: {summary.overall_score*100:.1f}%")
            print(f"   Deployment Status: {summary.readiness_status.replace('_', ' ')}")
            return True
        else:
            print(f"\n⚠️  Phase 4 Testing completed with issues.")
            print(f"   Overall Score: {summary.overall_score*100:.1f}%")
            print(f"   Status: {summary.readiness_status.replace('_', ' ')}")
            print(f"   Critical Blockers: {summary.critical_blockers}")
            return False

    except Exception as e:
        print(f"\n💥 Phase 4 Testing FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
