"""User Experience (UX) testing suite for Apple BCI-HID system."""

import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Tuple


class UsabilityMetric(Enum):
    """Usability metrics for UX testing."""
    TASK_SUCCESS_RATE = "task_success_rate"
    TASK_COMPLETION_TIME = "task_completion_time"
    ERROR_RATE = "error_rate"
    LEARNING_CURVE = "learning_curve"
    USER_SATISFACTION = "user_satisfaction"
    COGNITIVE_LOAD = "cognitive_load"
    ACCESSIBILITY_SCORE = "accessibility_score"


@dataclass
class UXTestCase:
    """User experience test case definition."""
    name: str
    description: str
    task_type: str
    difficulty: str  # 'beginner', 'intermediate', 'advanced'
    target_time_seconds: int
    success_criteria: Dict[str, float]
    accessibility_features: List[str]
    user_personas: List[str]


@dataclass
class UXTestResult:
    """User experience test result."""
    test_case: str
    user_persona: str
    task_completion_time: float
    task_success: bool
    error_count: int
    user_satisfaction_score: float  # 1-10 scale
    cognitive_load_score: float     # 1-10 scale (lower is better)
    accessibility_score: float     # 1-10 scale
    learning_attempts: int
    feedback: List[str]
    metrics: Dict[str, float]


class UXTestingSuite:
    """Comprehensive user experience testing for BCI-HID system."""

    def __init__(self):
        self.results: List[UXTestResult] = []
        self.test_cases = self._define_ux_test_cases()
        self.user_personas = self._define_user_personas()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ux_testing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        print("👥 User Experience Testing Suite Initialized")
        print(f"   Test Cases: {len(self.test_cases)}")
        print(f"   User Personas: {len(self.user_personas)}")

    def _define_user_personas(self) -> Dict[str, Dict[str, Any]]:
        """Define user personas for testing."""
        return {
            'tech_enthusiast': {
                'name': 'Alex - Tech Enthusiast',
                'description': 'Early adopter, comfortable with new technology',
                'tech_experience': 'high',
                'age_group': '25-35',
                'accessibility_needs': [],
                'learning_style': 'visual',
                'patience_level': 'high',
                'error_tolerance': 'medium',
                'expected_performance': {
                    'task_completion_multiplier': 0.8,
                    'learning_curve_multiplier': 0.7,
                    'satisfaction_baseline': 8.0
                }
            },

            'office_worker': {
                'name': 'Jamie - Office Professional',
                'description': 'Regular computer user, productivity focused',
                'tech_experience': 'medium',
                'age_group': '30-50',
                'accessibility_needs': [],
                'learning_style': 'hands_on',
                'patience_level': 'medium',
                'error_tolerance': 'low',
                'expected_performance': {
                    'task_completion_multiplier': 1.0,
                    'learning_curve_multiplier': 1.0,
                    'satisfaction_baseline': 7.0
                }
            },

            'senior_user': {
                'name': 'Pat - Senior User',
                'description': 'Limited tech experience, needs clear guidance',
                'tech_experience': 'low',
                'age_group': '60+',
                'accessibility_needs': ['large_text', 'high_contrast', 'slow_pace'],
                'learning_style': 'step_by_step',
                'patience_level': 'low',
                'error_tolerance': 'very_low',
                'expected_performance': {
                    'task_completion_multiplier': 1.5,
                    'learning_curve_multiplier': 2.0,
                    'satisfaction_baseline': 6.0
                }
            },

            'accessibility_user': {
                'name': 'Morgan - Accessibility User',
                'description': 'User with motor impairments, relies on assistive tech',
                'tech_experience': 'medium',
                'age_group': '20-60',
                'accessibility_needs': ['voice_over', 'switch_control', 'head_tracking'],
                'learning_style': 'auditory',
                'patience_level': 'high',
                'error_tolerance': 'high',
                'expected_performance': {
                    'task_completion_multiplier': 1.3,
                    'learning_curve_multiplier': 1.2,
                    'satisfaction_baseline': 7.5
                }
            },

            'gamer': {
                'name': 'Riley - Gamer',
                'description': 'Gaming enthusiast, expects low latency and precision',
                'tech_experience': 'high',
                'age_group': '16-30',
                'accessibility_needs': [],
                'learning_style': 'trial_error',
                'patience_level': 'low',
                'error_tolerance': 'very_low',
                'expected_performance': {
                    'task_completion_multiplier': 0.7,
                    'learning_curve_multiplier': 0.5,
                    'satisfaction_baseline': 8.5
                }
            }
        }

    def _define_ux_test_cases(self) -> List[UXTestCase]:
        """Define comprehensive UX test cases."""
        return [
            # Basic Navigation Tests
            UXTestCase(
                name="first_time_setup",
                description="Initial system setup and calibration",
                task_type="setup",
                difficulty="beginner",
                target_time_seconds=300,  # 5 minutes
                success_criteria={
                    "completion_rate": 0.9,
                    "error_rate": 0.1,
                    "satisfaction": 7.0
                },
                accessibility_features=["voice_guidance", "large_text", "high_contrast"],
                user_personas=["tech_enthusiast", "office_worker", "senior_user", "accessibility_user"]
            ),

            UXTestCase(
                name="basic_click_navigation",
                description="Basic clicking and navigation tasks",
                task_type="navigation",
                difficulty="beginner",
                target_time_seconds=60,
                success_criteria={
                    "completion_rate": 0.95,
                    "error_rate": 0.05,
                    "satisfaction": 7.5
                },
                accessibility_features=["voice_over", "switch_control"],
                user_personas=["tech_enthusiast", "office_worker", "senior_user", "accessibility_user", "gamer"]
            ),

            UXTestCase(
                name="scroll_and_swipe",
                description="Scrolling and swiping gestures",
                task_type="gesture",
                difficulty="beginner",
                target_time_seconds=45,
                success_criteria={
                    "completion_rate": 0.9,
                    "error_rate": 0.1,
                    "satisfaction": 7.0
                },
                accessibility_features=["gesture_customization", "sensitivity_adjustment"],
                user_personas=["tech_enthusiast", "office_worker", "gamer"]
            ),

            # Intermediate Tasks
            UXTestCase(
                name="document_editing",
                description="Text editing with copy, paste, select operations",
                task_type="productivity",
                difficulty="intermediate",
                target_time_seconds=180,  # 3 minutes
                success_criteria={
                    "completion_rate": 0.85,
                    "error_rate": 0.15,
                    "satisfaction": 7.5
                },
                accessibility_features=["voice_dictation", "word_prediction"],
                user_personas=["tech_enthusiast", "office_worker", "accessibility_user"]
            ),

            UXTestCase(
                name="multitasking_workflow",
                description="Switching between multiple applications",
                task_type="productivity",
                difficulty="intermediate",
                target_time_seconds=240,  # 4 minutes
                success_criteria={
                    "completion_rate": 0.8,
                    "error_rate": 0.2,
                    "satisfaction": 7.0
                },
                accessibility_features=["app_switching_shortcuts", "workspace_management"],
                user_personas=["tech_enthusiast", "office_worker"]
            ),

            UXTestCase(
                name="gesture_customization",
                description="Customizing gesture mappings and sensitivity",
                task_type="customization",
                difficulty="intermediate",
                target_time_seconds=300,  # 5 minutes
                success_criteria={
                    "completion_rate": 0.75,
                    "error_rate": 0.25,
                    "satisfaction": 8.0
                },
                accessibility_features=["custom_gestures", "sensitivity_profiles"],
                user_personas=["tech_enthusiast", "accessibility_user", "gamer"]
            ),

            # Advanced Tasks
            UXTestCase(
                name="gaming_session",
                description="Gaming with rapid, precise gestures",
                task_type="gaming",
                difficulty="advanced",
                target_time_seconds=600,  # 10 minutes
                success_criteria={
                    "completion_rate": 0.7,
                    "error_rate": 0.3,
                    "satisfaction": 8.5
                },
                accessibility_features=["low_latency_mode", "precision_enhancement"],
                user_personas=["tech_enthusiast", "gamer"]
            ),

            UXTestCase(
                name="accessibility_workflow",
                description="Complete workflow using accessibility features",
                task_type="accessibility",
                difficulty="advanced",
                target_time_seconds=480,  # 8 minutes
                success_criteria={
                    "completion_rate": 0.8,
                    "error_rate": 0.2,
                    "satisfaction": 8.0
                },
                accessibility_features=["full_accessibility_suite"],
                user_personas=["accessibility_user", "senior_user"]
            ),

            UXTestCase(
                name="extended_use_session",
                description="Extended 30-minute usage session",
                task_type="endurance",
                difficulty="advanced",
                target_time_seconds=1800,  # 30 minutes
                success_criteria={
                    "completion_rate": 0.85,
                    "error_rate": 0.15,
                    "satisfaction": 7.5
                },
                accessibility_features=["fatigue_monitoring", "adaptive_sensitivity"],
                user_personas=["tech_enthusiast", "office_worker", "accessibility_user"]
            )
        ]

    async def run_all_ux_tests(self) -> Dict[str, Any]:
        """Run all UX test cases for all user personas."""
        print("\n👥 Starting User Experience Testing Suite")
        print("=" * 60)

        total_tests = 0
        completed_tests = 0
        failed_tests = 0

        for test_case in self.test_cases:
            for persona_id in test_case.user_personas:
                total_tests += 1

                try:
                    print(f"\n🧪 Running UX Test: {test_case.name}")
                    print(f"   User Persona: {self.user_personas[persona_id]['name']}")
                    print(f"   Difficulty: {test_case.difficulty}")
                    print(f"   Target Time: {test_case.target_time_seconds}s")

                    result = await self.run_single_ux_test(test_case, persona_id)
                    self.results.append(result)

                    if result.task_success:
                        print("   ✅ COMPLETED")
                        completed_tests += 1
                    else:
                        print("   ❌ FAILED")
                        failed_tests += 1

                    print(f"   Time: {result.task_completion_time:.1f}s")
                    print(f"   Satisfaction: {result.user_satisfaction_score:.1f}/10")

                except Exception as e:
                    print(f"   💥 ERROR: {e}")
                    failed_tests += 1
                    self.logger.error(f"UX test {test_case.name} for {persona_id} failed: {e}")

        # Generate UX report
        return self._generate_ux_report(completed_tests, failed_tests, total_tests)

    async def run_single_ux_test(self, test_case: UXTestCase, persona_id: str) -> UXTestResult:
        """Run a single UX test case for a specific user persona."""
        persona = self.user_personas[persona_id]
        start_time = time.perf_counter()

        # Simulate user behavior based on persona
        performance_modifier = persona['expected_performance']

        # Calculate expected completion time based on persona
        expected_time = test_case.target_time_seconds * performance_modifier['task_completion_multiplier']

        # Simulate task execution with persona-specific variations
        actual_time, success, errors, attempts = await self._simulate_task_execution(
            test_case, persona, expected_time
        )

        # Calculate satisfaction based on performance and persona
        satisfaction = self._calculate_user_satisfaction(
            test_case, persona, actual_time, success, errors
        )

        # Calculate cognitive load
        cognitive_load = self._calculate_cognitive_load(test_case, persona, errors, attempts)

        # Calculate accessibility score
        accessibility_score = self._calculate_accessibility_score(test_case, persona)

        # Generate user feedback
        feedback = self._generate_user_feedback(test_case, persona, success, errors, actual_time)

        # Calculate metrics
        metrics = self._calculate_ux_metrics(test_case, actual_time, success, errors, satisfaction)

        return UXTestResult(
            test_case=test_case.name,
            user_persona=persona_id,
            task_completion_time=actual_time,
            task_success=success,
            error_count=errors,
            user_satisfaction_score=satisfaction,
            cognitive_load_score=cognitive_load,
            accessibility_score=accessibility_score,
            learning_attempts=attempts,
            feedback=feedback,
            metrics=metrics
        )

    async def _simulate_task_execution(self, test_case: UXTestCase, persona: Dict[str, Any],
                                     expected_time: float) -> Tuple[float, bool, int, int]:
        """Simulate task execution based on user persona."""
        import random
        random.seed(42)  # For reproducible results

        # Base task execution simulation
        base_time = expected_time

        # Adjust for persona characteristics
        if persona['tech_experience'] == 'low':
            base_time *= 1.2
        elif persona['tech_experience'] == 'high':
            base_time *= 0.9

        # Adjust for accessibility needs
        if persona['accessibility_needs']:
            base_time *= 1.1

        # Simulate learning attempts for complex tasks
        attempts = 1
        if test_case.difficulty == 'advanced':
            attempts = max(1, int(persona['expected_performance']['learning_curve_multiplier']))

        # Simulate errors based on persona error tolerance
        error_probability = {
            'very_low': 0.05,
            'low': 0.1,
            'medium': 0.15,
            'high': 0.2
        }.get(persona['error_tolerance'], 0.1)

        errors = 0
        if random.random() < error_probability:
            errors = random.randint(1, 3)
            base_time += errors * 15  # 15 seconds per error recovery

        # Determine task success
        success_probability = test_case.success_criteria['completion_rate']

        # Adjust success probability based on persona
        if persona['tech_experience'] == 'high':
            success_probability += 0.1
        elif persona['tech_experience'] == 'low':
            success_probability -= 0.1

        if persona['patience_level'] == 'low' and base_time > expected_time * 1.5:
            success_probability -= 0.2  # Impatient users give up

        success = random.random() < min(0.95, max(0.1, success_probability))

        # Add some realistic variation
        time_variation = random.uniform(0.8, 1.2)
        actual_time = base_time * time_variation

        # Simulate real-time delay
        await asyncio.sleep(0.1)  # 100ms simulation delay

        return actual_time, success, errors, attempts

    def _calculate_user_satisfaction(self, test_case: UXTestCase, persona: Dict[str, Any],
                                   actual_time: float, success: bool, errors: int) -> float:
        """Calculate user satisfaction score based on performance and persona."""
        base_satisfaction = persona['expected_performance']['satisfaction_baseline']

        # Adjust for task success
        if not success:
            base_satisfaction -= 2.0

        # Adjust for completion time
        time_ratio = actual_time / test_case.target_time_seconds
        if time_ratio > 1.5:
            base_satisfaction -= 1.0
        elif time_ratio < 0.8:
            base_satisfaction += 0.5

        # Adjust for errors
        base_satisfaction -= errors * 0.5

        # Adjust for accessibility features
        if persona['accessibility_needs'] and any(
            feature in test_case.accessibility_features
            for feature in persona['accessibility_needs']
        ):
            base_satisfaction += 1.0

        # Persona-specific adjustments
        if persona['tech_experience'] == 'high' and test_case.difficulty == 'beginner':
            base_satisfaction -= 0.5  # Bored by simple tasks

        if persona['patience_level'] == 'low' and actual_time > test_case.target_time_seconds:
            base_satisfaction -= 0.5

        return max(1.0, min(10.0, base_satisfaction))

    def _calculate_cognitive_load(self, test_case: UXTestCase, persona: Dict[str, Any],
                                errors: int, attempts: int) -> float:
        """Calculate cognitive load score (lower is better)."""
        base_load = {
            'beginner': 3.0,
            'intermediate': 5.0,
            'advanced': 7.0
        }.get(test_case.difficulty, 5.0)

        # Adjust for persona tech experience
        if persona['tech_experience'] == 'high':
            base_load -= 1.0
        elif persona['tech_experience'] == 'low':
            base_load += 1.5

        # Adjust for errors and attempts
        base_load += errors * 0.5
        base_load += (attempts - 1) * 0.3

        # Adjust for learning style match
        if test_case.task_type == 'setup' and persona['learning_style'] == 'step_by_step':
            base_load -= 0.5
        elif test_case.task_type == 'gaming' and persona['learning_style'] == 'trial_error':
            base_load -= 0.5

        return max(1.0, min(10.0, base_load))

    def _calculate_accessibility_score(self, test_case: UXTestCase, persona: Dict[str, Any]) -> float:
        """Calculate accessibility score."""
        base_score = 8.0

        # Check if persona's accessibility needs are met
        if persona['accessibility_needs']:
            needs_met = sum(
                1 for need in persona['accessibility_needs']
                if any(need in feature for feature in test_case.accessibility_features)
            )
            coverage = needs_met / len(persona['accessibility_needs'])
            base_score = 5.0 + (coverage * 5.0)

        # Bonus for comprehensive accessibility features
        if len(test_case.accessibility_features) >= 3:
            base_score += 0.5

        return max(1.0, min(10.0, base_score))

    def _generate_user_feedback(self, test_case: UXTestCase, persona: Dict[str, Any],
                              success: bool, errors: int, actual_time: float) -> List[str]:
        """Generate realistic user feedback based on performance."""
        feedback = []

        if success:
            if actual_time <= test_case.target_time_seconds:
                feedback.append("Task completed efficiently, felt intuitive")
            else:
                feedback.append("Task completed but took longer than expected")
        else:
            feedback.append("Could not complete the task successfully")

        if errors > 0:
            if errors == 1:
                feedback.append("Made one mistake but recovered easily")
            else:
                feedback.append(f"Made {errors} errors, system could be more forgiving")

        # Persona-specific feedback
        if persona['tech_experience'] == 'low':
            if test_case.difficulty == 'advanced':
                feedback.append("Task felt too complex, need more guidance")
            else:
                feedback.append("Appreciated clear instructions and feedback")

        if persona['accessibility_needs']:
            if any(need in feature for need in persona['accessibility_needs']
                   for feature in test_case.accessibility_features):
                feedback.append("Accessibility features worked well")
            else:
                feedback.append("Could use better accessibility support")

        if persona['patience_level'] == 'low' and actual_time > test_case.target_time_seconds * 1.2:
            feedback.append("Process felt too slow, would prefer faster responses")

        return feedback

    def _calculate_ux_metrics(self, test_case: UXTestCase, actual_time: float,
                            success: bool, errors: int, satisfaction: float) -> Dict[str, float]:
        """Calculate comprehensive UX metrics."""
        return {
            UsabilityMetric.TASK_SUCCESS_RATE.value: 1.0 if success else 0.0,
            UsabilityMetric.TASK_COMPLETION_TIME.value: actual_time,
            UsabilityMetric.ERROR_RATE.value: errors / max(1, actual_time / 60),  # Errors per minute
            UsabilityMetric.USER_SATISFACTION.value: satisfaction,
            'time_efficiency': test_case.target_time_seconds / actual_time if actual_time > 0 else 0,
            'error_recovery_time': errors * 15 if errors > 0 else 0,  # Assumed 15s per error
        }

    def _generate_ux_report(self, completed: int, failed: int, total: int) -> Dict[str, Any]:
        """Generate comprehensive UX report."""

        # Calculate overall metrics
        overall_metrics = self._calculate_overall_ux_metrics()

        # Analyze by persona
        persona_analysis = self._analyze_by_persona()

        # Analyze by test case
        test_case_analysis = self._analyze_by_test_case()

        # Analyze accessibility
        accessibility_analysis = self._analyze_accessibility()

        # Generate insights and recommendations
        insights = self._generate_ux_insights()

        report = {
            'test_summary': {
                'total_tests': total,
                'completed_tests': completed,
                'failed_tests': failed,
                'success_rate': completed / total if total > 0 else 0,
                'total_test_time': sum(r.task_completion_time for r in self.results),
                'unique_personas': len(set(r.user_persona for r in self.results)),
                'unique_test_cases': len(set(r.test_case for r in self.results))
            },
            'overall_metrics': overall_metrics,
            'persona_analysis': persona_analysis,
            'test_case_analysis': test_case_analysis,
            'accessibility_analysis': accessibility_analysis,
            'user_feedback_summary': self._summarize_user_feedback(),
            'detailed_results': [self._result_to_dict(r) for r in self.results],
            'insights_and_recommendations': insights
        }

        return report

    def _calculate_overall_ux_metrics(self) -> Dict[str, float]:
        """Calculate overall UX metrics across all tests."""
        if not self.results:
            return {}

        return {
            'average_task_success_rate': statistics.mean(
                r.metrics[UsabilityMetric.TASK_SUCCESS_RATE.value] for r in self.results
            ),
            'average_completion_time': statistics.mean(r.task_completion_time for r in self.results),
            'median_completion_time': statistics.median(r.task_completion_time for r in self.results),
            'average_error_rate': statistics.mean(
                r.metrics[UsabilityMetric.ERROR_RATE.value] for r in self.results
            ),
            'average_user_satisfaction': statistics.mean(r.user_satisfaction_score for r in self.results),
            'average_cognitive_load': statistics.mean(r.cognitive_load_score for r in self.results),
            'average_accessibility_score': statistics.mean(r.accessibility_score for r in self.results),
            'total_errors': sum(r.error_count for r in self.results),
            'total_learning_attempts': sum(r.learning_attempts for r in self.results)
        }

    def _analyze_by_persona(self) -> Dict[str, Any]:
        """Analyze UX metrics by user persona."""
        persona_analysis = {}

        for persona_id in self.user_personas.keys():
            persona_results = [r for r in self.results if r.user_persona == persona_id]

            if persona_results:
                persona_analysis[persona_id] = {
                    'test_count': len(persona_results),
                    'success_rate': statistics.mean(r.task_success for r in persona_results),
                    'avg_completion_time': statistics.mean(r.task_completion_time for r in persona_results),
                    'avg_satisfaction': statistics.mean(r.user_satisfaction_score for r in persona_results),
                    'avg_cognitive_load': statistics.mean(r.cognitive_load_score for r in persona_results),
                    'avg_accessibility_score': statistics.mean(r.accessibility_score for r in persona_results),
                    'total_errors': sum(r.error_count for r in persona_results),
                    'avg_learning_attempts': statistics.mean(r.learning_attempts for r in persona_results)
                }

        return persona_analysis

    def _analyze_by_test_case(self) -> Dict[str, Any]:
        """Analyze UX metrics by test case."""
        test_case_analysis = {}

        for test_case in self.test_cases:
            case_results = [r for r in self.results if r.test_case == test_case.name]

            if case_results:
                test_case_analysis[test_case.name] = {
                    'test_count': len(case_results),
                    'success_rate': statistics.mean(r.task_success for r in case_results),
                    'avg_completion_time': statistics.mean(r.task_completion_time for r in case_results),
                    'target_time': test_case.target_time_seconds,
                    'time_efficiency': test_case.target_time_seconds / statistics.mean(
                        r.task_completion_time for r in case_results
                    ),
                    'avg_satisfaction': statistics.mean(r.user_satisfaction_score for r in case_results),
                    'difficulty': test_case.difficulty,
                    'total_errors': sum(r.error_count for r in case_results)
                }

        return test_case_analysis

    def _analyze_accessibility(self) -> Dict[str, Any]:
        """Analyze accessibility-specific metrics."""
        accessibility_users = [
            r for r in self.results
            if r.user_persona in ['accessibility_user', 'senior_user']
        ]

        accessibility_tests = [
            r for r in self.results
            if any(test_case.name == r.test_case and test_case.accessibility_features
                   for test_case in self.test_cases)
        ]

        return {
            'accessibility_user_performance': {
                'test_count': len(accessibility_users),
                'avg_success_rate': statistics.mean(r.task_success for r in accessibility_users) if accessibility_users else 0,
                'avg_satisfaction': statistics.mean(r.user_satisfaction_score for r in accessibility_users) if accessibility_users else 0,
                'avg_accessibility_score': statistics.mean(r.accessibility_score for r in accessibility_users) if accessibility_users else 0
            },
            'accessibility_feature_effectiveness': {
                'tests_with_features': len(accessibility_tests),
                'avg_accessibility_score': statistics.mean(r.accessibility_score for r in accessibility_tests) if accessibility_tests else 0,
                'feature_satisfaction': statistics.mean(r.user_satisfaction_score for r in accessibility_tests) if accessibility_tests else 0
            },
            'accessibility_coverage': len([tc for tc in self.test_cases if tc.accessibility_features]) / len(self.test_cases)
        }

    def _summarize_user_feedback(self) -> Dict[str, Any]:
        """Summarize user feedback across all tests."""
        all_feedback = []
        for result in self.results:
            all_feedback.extend(result.feedback)

        # Count common feedback themes
        feedback_themes = {
            'positive': len([f for f in all_feedback if any(word in f.lower() for word in ['good', 'great', 'excellent', 'easy', 'intuitive'])]),
            'negative': len([f for f in all_feedback if any(word in f.lower() for word in ['difficult', 'hard', 'confusing', 'slow', 'frustrating'])]),
            'accessibility': len([f for f in all_feedback if 'accessibility' in f.lower()]),
            'performance': len([f for f in all_feedback if any(word in f.lower() for word in ['fast', 'slow', 'quick', 'lag'])]),
            'errors': len([f for f in all_feedback if any(word in f.lower() for word in ['error', 'mistake', 'wrong'])])
        }

        return {
            'total_feedback_items': len(all_feedback),
            'feedback_themes': feedback_themes,
            'sentiment_distribution': {
                'positive': feedback_themes['positive'] / len(all_feedback) if all_feedback else 0,
                'negative': feedback_themes['negative'] / len(all_feedback) if all_feedback else 0,
                'neutral': 1 - (feedback_themes['positive'] + feedback_themes['negative']) / len(all_feedback) if all_feedback else 1
            }
        }

    def _generate_ux_insights(self) -> List[str]:
        """Generate UX insights and recommendations."""
        insights = []

        if not self.results:
            return ["No test results available for analysis."]

        # Overall performance insights
        overall_metrics = self._calculate_overall_ux_metrics()

        if overall_metrics['average_user_satisfaction'] < 7.0:
            insights.append(f"User satisfaction is below target (current: {overall_metrics['average_user_satisfaction']:.1f}/10). Focus on improving usability.")

        if overall_metrics['average_task_success_rate'] < 0.8:
            insights.append(f"Task success rate is low ({overall_metrics['average_task_success_rate']*100:.1f}%). Simplify workflows and improve guidance.")

        if overall_metrics['average_cognitive_load'] > 6.0:
            insights.append(f"Cognitive load is high ({overall_metrics['average_cognitive_load']:.1f}/10). Reduce complexity and provide better visual cues.")

        # Persona-specific insights
        persona_analysis = self._analyze_by_persona()
        worst_persona = min(persona_analysis.items(), key=lambda x: x[1]['avg_satisfaction'], default=None)

        if worst_persona and worst_persona[1]['avg_satisfaction'] < 6.0:
            persona_name = self.user_personas[worst_persona[0]]['name']
            insights.append(f"Users like {persona_name} are struggling (satisfaction: {worst_persona[1]['avg_satisfaction']:.1f}/10). Tailor experience for this user type.")

        # Test case insights
        test_case_analysis = self._analyze_by_test_case()
        problematic_tests = [name for name, data in test_case_analysis.items() if data['success_rate'] < 0.7]

        if problematic_tests:
            insights.append(f"Tests with low success rates: {', '.join(problematic_tests)}. Review and simplify these workflows.")

        # Accessibility insights
        accessibility_analysis = self._analyze_accessibility()
        if accessibility_analysis['accessibility_user_performance']['avg_satisfaction'] < 7.0:
            insights.append("Accessibility users report lower satisfaction. Enhance accessibility features and testing.")

        # Performance insights
        slow_tests = [name for name, data in test_case_analysis.items() if data['time_efficiency'] < 0.8]
        if slow_tests:
            insights.append(f"Tests taking longer than expected: {', '.join(slow_tests)}. Optimize performance and reduce steps.")

        if not insights:
            insights.append("UX performance is meeting targets across all measured dimensions. Continue monitoring and incremental improvements.")

        return insights

    def _result_to_dict(self, result: UXTestResult) -> Dict[str, Any]:
        """Convert UX test result to dictionary."""
        return {
            'test_case': result.test_case,
            'user_persona': result.user_persona,
            'task_completion_time': result.task_completion_time,
            'task_success': result.task_success,
            'error_count': result.error_count,
            'user_satisfaction_score': result.user_satisfaction_score,
            'cognitive_load_score': result.cognitive_load_score,
            'accessibility_score': result.accessibility_score,
            'learning_attempts': result.learning_attempts,
            'feedback': result.feedback,
            'metrics': result.metrics
        }

    def save_ux_results(self, filename: str = "ux_test_results.json"):
        """Save UX test results to file."""
        report = self._generate_ux_report(
            len([r for r in self.results if r.task_success]),
            len([r for r in self.results if not r.task_success]),
            len(self.results)
        )

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"👥 UX test results saved to {filename}")


async def main():
    """Main UX testing function."""
    print("👥 Apple BCI-HID User Experience Testing Suite")
    print("=" * 60)

    ux_suite = UXTestingSuite()

    try:
        # Run all UX tests
        report = await ux_suite.run_all_ux_tests()

        # Print UX summary
        print("\n" + "=" * 60)
        print("👥 USER EXPERIENCE TESTING SUMMARY")
        print("=" * 60)

        summary = report['test_summary']
        overall = report['overall_metrics']
        accessibility = report['accessibility_analysis']

        print("\nTest Results:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Completed: {summary['completed_tests']}")
        print(f"  Failed: {summary['failed_tests']}")
        print(f"  Success Rate: {summary['success_rate']*100:.1f}%")
        print(f"  User Personas: {summary['unique_personas']}")
        print(f"  Test Cases: {summary['unique_test_cases']}")

        print("\nOverall UX Metrics:")
        print(f"  Avg User Satisfaction: {overall['average_user_satisfaction']:.1f}/10")
        print(f"  Avg Task Success Rate: {overall['average_task_success_rate']*100:.1f}%")
        print(f"  Avg Completion Time: {overall['average_completion_time']:.1f}s")
        print(f"  Avg Cognitive Load: {overall['average_cognitive_load']:.1f}/10 (lower is better)")
        print(f"  Avg Accessibility Score: {overall['average_accessibility_score']:.1f}/10")
        print(f"  Total Errors: {overall['total_errors']}")

        print("\nAccessibility Analysis:")
        acc_user_perf = accessibility['accessibility_user_performance']
        print(f"  Accessibility User Success Rate: {acc_user_perf['avg_success_rate']*100:.1f}%")
        print(f"  Accessibility User Satisfaction: {acc_user_perf['avg_satisfaction']:.1f}/10")
        print(f"  Accessibility Feature Coverage: {accessibility['accessibility_coverage']*100:.1f}%")

        print("\nTop Insights:")
        for i, insight in enumerate(report['insights_and_recommendations'][:5], 1):
            print(f"  {i}. {insight}")

        # Save results
        ux_suite.save_ux_results()

        print("\n✅ User experience testing completed successfully!")

        # Return success if overall satisfaction is good
        return overall['average_user_satisfaction'] >= 7.0

    except Exception as e:
        print(f"\n❌ User experience testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
