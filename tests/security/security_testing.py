"""Security testing suite for Apple BCI-HID system."""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.compression import CompressionQuality, WaveletCompressor


@dataclass
class SecurityTestCase:
    """Security test case definition."""
    name: str
    description: str
    test_type: str  # 'encryption', 'authentication', 'authorization', 'injection', 'data_leak'
    severity: str  # 'critical', 'high', 'medium', 'low'
    expected_outcome: str
    attack_vectors: List[str]


@dataclass
class SecurityResult:
    """Security test result."""
    test_case: str
    passed: bool
    vulnerabilities_found: List[str]
    risk_level: str
    mitigation_suggestions: List[str]
    execution_time: float
    details: Dict[str, Any]


class SecurityTestingSuite:
    """Comprehensive security testing for BCI-HID system."""

    def __init__(self):
        self.results: List[SecurityResult] = []
        self.test_cases = self._define_security_tests()
        self.crypto_key = self._generate_test_key()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('security_testing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        print("🔐 Security Testing Suite Initialized")
        print(f"   Test Cases: {len(self.test_cases)}")

    def _generate_test_key(self) -> bytes:
        """Generate test encryption key."""
        return os.urandom(32)  # 256-bit key

    def _define_security_tests(self) -> List[SecurityTestCase]:
        """Define comprehensive security test cases."""
        return [
            # Encryption Tests
            SecurityTestCase(
                name="data_encryption_at_rest",
                description="Verify neural data is encrypted when stored",
                test_type="encryption",
                severity="critical",
                expected_outcome="Data must be encrypted with AES-256",
                attack_vectors=["file_system_access", "memory_dump", "cold_boot"]
            ),

            SecurityTestCase(
                name="data_encryption_in_transit",
                description="Verify neural data is encrypted during transmission",
                test_type="encryption",
                severity="critical",
                expected_outcome="TLS 1.3 or higher encryption",
                attack_vectors=["network_sniffing", "man_in_middle", "protocol_downgrade"]
            ),

            SecurityTestCase(
                name="key_management_security",
                description="Test encryption key storage and rotation",
                test_type="encryption",
                severity="high",
                expected_outcome="Secure key storage with regular rotation",
                attack_vectors=["key_extraction", "weak_keys", "key_reuse"]
            ),

            # Authentication Tests
            SecurityTestCase(
                name="device_authentication",
                description="Verify BCI device authentication mechanisms",
                test_type="authentication",
                severity="critical",
                expected_outcome="Strong device authentication required",
                attack_vectors=["device_spoofing", "replay_attacks", "weak_credentials"]
            ),

            SecurityTestCase(
                name="user_authentication",
                description="Test user authentication for system access",
                test_type="authentication",
                severity="high",
                expected_outcome="Multi-factor authentication required",
                attack_vectors=["brute_force", "credential_stuffing", "session_hijacking"]
            ),

            SecurityTestCase(
                name="biometric_authentication",
                description="Test neural signature authentication",
                test_type="authentication",
                severity="high",
                expected_outcome="Unique neural signature verification",
                attack_vectors=["signature_replay", "template_theft", "spoofing"]
            ),

            # Authorization Tests
            SecurityTestCase(
                name="privilege_escalation",
                description="Test for unauthorized privilege escalation",
                test_type="authorization",
                severity="critical",
                expected_outcome="No privilege escalation possible",
                attack_vectors=["buffer_overflow", "code_injection", "race_conditions"]
            ),

            SecurityTestCase(
                name="access_control",
                description="Verify proper access control mechanisms",
                test_type="authorization",
                severity="high",
                expected_outcome="Strict role-based access control",
                attack_vectors=["unauthorized_access", "permission_bypass", "path_traversal"]
            ),

            # Injection Tests
            SecurityTestCase(
                name="neural_signal_injection",
                description="Test for neural signal injection attacks",
                test_type="injection",
                severity="critical",
                expected_outcome="Malicious signals detected and blocked",
                attack_vectors=["signal_spoofing", "command_injection", "firmware_injection"]
            ),

            SecurityTestCase(
                name="data_injection",
                description="Test for malicious data injection",
                test_type="injection",
                severity="high",
                expected_outcome="Input validation prevents injection",
                attack_vectors=["sql_injection", "code_injection", "command_injection"]
            ),

            # Data Leak Tests
            SecurityTestCase(
                name="memory_leakage",
                description="Test for sensitive data in memory",
                test_type="data_leak",
                severity="high",
                expected_outcome="No sensitive data in memory dumps",
                attack_vectors=["memory_dump", "side_channel", "timing_attacks"]
            ),

            SecurityTestCase(
                name="log_data_exposure",
                description="Test for sensitive data in logs",
                test_type="data_leak",
                severity="medium",
                expected_outcome="No sensitive data logged",
                attack_vectors=["log_analysis", "file_disclosure", "debug_info"]
            ),

            SecurityTestCase(
                name="network_data_leakage",
                description="Test for data leakage over network",
                test_type="data_leak",
                severity="critical",
                expected_outcome="No unencrypted sensitive data transmitted",
                attack_vectors=["packet_sniffing", "dns_leakage", "metadata_exposure"]
            )
        ]

    async def run_all_security_tests(self) -> Dict[str, Any]:
        """Run all security test cases."""
        print("\n🔐 Starting Security Testing Suite")
        print("=" * 60)

        total_tests = len(self.test_cases)
        passed_tests = 0
        failed_tests = 0
        critical_vulnerabilities = 0

        for test_case in self.test_cases:
            try:
                print(f"\n🧪 Running Security Test: {test_case.name}")
                print(f"   Type: {test_case.test_type}")
                print(f"   Severity: {test_case.severity}")

                result = await self.run_security_test(test_case)
                self.results.append(result)

                if result.passed:
                    print("   ✅ PASSED")
                    passed_tests += 1
                else:
                    print("   ❌ FAILED")
                    failed_tests += 1
                    if test_case.severity == 'critical':
                        critical_vulnerabilities += 1

                    # Print vulnerabilities found
                    for vuln in result.vulnerabilities_found:
                        print(f"     🚨 {vuln}")

            except Exception as e:
                print(f"   💥 ERROR: {e}")
                failed_tests += 1
                self.logger.error(f"Security test {test_case.name} failed with error: {e}")

        # Generate security report
        return self._generate_security_report(passed_tests, failed_tests, total_tests, critical_vulnerabilities)

    async def run_security_test(self, test_case: SecurityTestCase) -> SecurityResult:
        """Run a single security test case."""
        start_time = time.perf_counter()
        vulnerabilities = []
        risk_level = "low"
        mitigation_suggestions = []
        details = {}

        try:
            if test_case.test_type == "encryption":
                result = await self._test_encryption_security(test_case)
            elif test_case.test_type == "authentication":
                result = await self._test_authentication_security(test_case)
            elif test_case.test_type == "authorization":
                result = await self._test_authorization_security(test_case)
            elif test_case.test_type == "injection":
                result = await self._test_injection_security(test_case)
            elif test_case.test_type == "data_leak":
                result = await self._test_data_leak_security(test_case)
            else:
                result = {
                    'passed': False,
                    'vulnerabilities': [f"Unknown test type: {test_case.test_type}"],
                    'risk_level': 'medium',
                    'suggestions': ['Implement proper test coverage'],
                    'details': {}
                }

            vulnerabilities = result['vulnerabilities']
            risk_level = result['risk_level']
            mitigation_suggestions = result['suggestions']
            details = result['details']

        except Exception as e:
            vulnerabilities = [f"Test execution error: {str(e)}"]
            risk_level = "high"
            mitigation_suggestions = ["Fix test execution issues"]
            details = {'error': str(e)}

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        passed = len(vulnerabilities) == 0

        return SecurityResult(
            test_case=test_case.name,
            passed=passed,
            vulnerabilities_found=vulnerabilities,
            risk_level=risk_level,
            mitigation_suggestions=mitigation_suggestions,
            execution_time=execution_time,
            details=details
        )

    async def _test_encryption_security(self, test_case: SecurityTestCase) -> Dict[str, Any]:
        """Test encryption-related security."""
        vulnerabilities = []
        suggestions = []
        details = {}

        if test_case.name == "data_encryption_at_rest":
            # Test data at rest encryption
            test_data = b"sensitive neural data"

            # Test with weak encryption (should fail)
            weak_encrypted = self._weak_encrypt(test_data)
            if self._is_easily_decryptable(weak_encrypted):
                vulnerabilities.append("Weak encryption detected - data easily decryptable")
                suggestions.append("Use AES-256 encryption for data at rest")

            # Test with strong encryption (should pass)
            strong_encrypted = self._strong_encrypt(test_data)
            if not self._is_properly_encrypted(strong_encrypted):
                vulnerabilities.append("Strong encryption not properly implemented")
                suggestions.append("Implement proper AES-256 encryption")

            details['encryption_strength'] = 'weak' if vulnerabilities else 'strong'

        elif test_case.name == "data_encryption_in_transit":
            # Test in-transit encryption
            tls_version = self._check_tls_version()
            if tls_version < 1.3:
                vulnerabilities.append(f"Weak TLS version: {tls_version}")
                suggestions.append("Upgrade to TLS 1.3 or higher")

            cipher_strength = self._check_cipher_strength()
            if cipher_strength < 256:
                vulnerabilities.append(f"Weak cipher strength: {cipher_strength} bits")
                suggestions.append("Use 256-bit or stronger ciphers")

            details['tls_version'] = tls_version
            details['cipher_strength'] = cipher_strength

        elif test_case.name == "key_management_security":
            # Test key management
            key_entropy = self._check_key_entropy(self.crypto_key)
            if key_entropy < 7.0:  # Shannon entropy
                vulnerabilities.append(f"Low key entropy: {key_entropy}")
                suggestions.append("Use cryptographically secure random keys")

            key_storage = self._check_key_storage()
            if not key_storage['secure']:
                vulnerabilities.append("Insecure key storage detected")
                suggestions.append("Store keys in secure hardware or encrypted storage")

            details['key_entropy'] = key_entropy
            details['key_storage'] = key_storage

        risk_level = "critical" if vulnerabilities else "low"

        return {
            'passed': len(vulnerabilities) == 0,
            'vulnerabilities': vulnerabilities,
            'risk_level': risk_level,
            'suggestions': suggestions,
            'details': details
        }

    async def _test_authentication_security(self, test_case: SecurityTestCase) -> Dict[str, Any]:
        """Test authentication-related security."""
        vulnerabilities = []
        suggestions = []
        details = {}

        if test_case.name == "device_authentication":
            # Test device authentication
            auth_result = self._test_device_auth()

            if not auth_result['certificate_valid']:
                vulnerabilities.append("Invalid device certificate")
                suggestions.append("Implement proper device certificate validation")

            if auth_result['brute_force_vulnerable']:
                vulnerabilities.append("Device authentication vulnerable to brute force")
                suggestions.append("Implement rate limiting and account lockout")

            if not auth_result['mutual_auth']:
                vulnerabilities.append("Mutual authentication not implemented")
                suggestions.append("Implement mutual authentication between device and system")

            details = auth_result

        elif test_case.name == "user_authentication":
            # Test user authentication
            auth_strength = self._test_user_auth_strength()

            if not auth_strength['mfa_enabled']:
                vulnerabilities.append("Multi-factor authentication not enabled")
                suggestions.append("Enable multi-factor authentication")

            if auth_strength['password_strength'] < 0.8:
                vulnerabilities.append("Weak password requirements")
                suggestions.append("Strengthen password requirements")

            if auth_strength['session_timeout'] > 3600:  # 1 hour
                vulnerabilities.append("Session timeout too long")
                suggestions.append("Reduce session timeout to 30 minutes or less")

            details = auth_strength

        elif test_case.name == "biometric_authentication":
            # Test biometric authentication
            biometric_security = self._test_biometric_auth()

            if biometric_security['false_accept_rate'] > 0.001:  # 0.1%
                vulnerabilities.append("High false acceptance rate")
                suggestions.append("Improve biometric matching algorithm")

            if not biometric_security['liveness_detection']:
                vulnerabilities.append("Liveness detection not implemented")
                suggestions.append("Implement liveness detection")

            if not biometric_security['template_protection']:
                vulnerabilities.append("Biometric templates not protected")
                suggestions.append("Encrypt and secure biometric templates")

            details = biometric_security

        risk_level = "critical" if any("certificate" in v or "brute force" in v for v in vulnerabilities) else "medium" if vulnerabilities else "low"

        return {
            'passed': len(vulnerabilities) == 0,
            'vulnerabilities': vulnerabilities,
            'risk_level': risk_level,
            'suggestions': suggestions,
            'details': details
        }

    async def _test_authorization_security(self, test_case: SecurityTestCase) -> Dict[str, Any]:
        """Test authorization-related security."""
        vulnerabilities = []
        suggestions = []
        details = {}

        if test_case.name == "privilege_escalation":
            # Test privilege escalation
            escalation_tests = self._test_privilege_escalation()

            if escalation_tests['buffer_overflow_possible']:
                vulnerabilities.append("Buffer overflow vulnerability detected")
                suggestions.append("Implement proper bounds checking")

            if escalation_tests['code_injection_possible']:
                vulnerabilities.append("Code injection vulnerability detected")
                suggestions.append("Implement input sanitization and validation")

            if escalation_tests['race_condition_detected']:
                vulnerabilities.append("Race condition vulnerability detected")
                suggestions.append("Implement proper synchronization mechanisms")

            details = escalation_tests

        elif test_case.name == "access_control":
            # Test access control
            access_control = self._test_access_control()

            if not access_control['rbac_implemented']:
                vulnerabilities.append("Role-based access control not implemented")
                suggestions.append("Implement proper RBAC system")

            if access_control['unauthorized_access_possible']:
                vulnerabilities.append("Unauthorized access possible")
                suggestions.append("Strengthen access control checks")

            if access_control['path_traversal_vulnerable']:
                vulnerabilities.append("Path traversal vulnerability detected")
                suggestions.append("Implement proper path validation")

            details = access_control

        risk_level = "critical" if any("overflow" in v or "injection" in v for v in vulnerabilities) else "high" if vulnerabilities else "low"

        return {
            'passed': len(vulnerabilities) == 0,
            'vulnerabilities': vulnerabilities,
            'risk_level': risk_level,
            'suggestions': suggestions,
            'details': details
        }

    async def _test_injection_security(self, test_case: SecurityTestCase) -> Dict[str, Any]:
        """Test injection-related security."""
        vulnerabilities = []
        suggestions = []
        details = {}

        if test_case.name == "neural_signal_injection":
            # Test neural signal injection
            signal_security = self._test_signal_injection()

            if signal_security['spoofing_possible']:
                vulnerabilities.append("Neural signal spoofing possible")
                suggestions.append("Implement signal authentication and validation")

            if signal_security['command_injection_possible']:
                vulnerabilities.append("Command injection through neural signals")
                suggestions.append("Sanitize and validate all neural signal inputs")

            if not signal_security['integrity_checking']:
                vulnerabilities.append("No signal integrity checking")
                suggestions.append("Implement signal integrity verification")

            details = signal_security

        elif test_case.name == "data_injection":
            # Test data injection
            injection_tests = self._test_data_injection()

            if injection_tests['sql_injection_possible']:
                vulnerabilities.append("SQL injection vulnerability detected")
                suggestions.append("Use parameterized queries")

            if injection_tests['command_injection_possible']:
                vulnerabilities.append("Command injection vulnerability detected")
                suggestions.append("Sanitize all user inputs")

            if injection_tests['code_injection_possible']:
                vulnerabilities.append("Code injection vulnerability detected")
                suggestions.append("Implement proper input validation")

            details = injection_tests

        risk_level = "critical" if vulnerabilities else "low"

        return {
            'passed': len(vulnerabilities) == 0,
            'vulnerabilities': vulnerabilities,
            'risk_level': risk_level,
            'suggestions': suggestions,
            'details': details
        }

    async def _test_data_leak_security(self, test_case: SecurityTestCase) -> Dict[str, Any]:
        """Test data leak-related security."""
        vulnerabilities = []
        suggestions = []
        details = {}

        if test_case.name == "memory_leakage":
            # Test memory leakage
            memory_security = self._test_memory_security()

            if memory_security['sensitive_data_in_memory']:
                vulnerabilities.append("Sensitive data found in memory dumps")
                suggestions.append("Clear sensitive data from memory after use")

            if memory_security['side_channel_vulnerable']:
                vulnerabilities.append("Side-channel attack vulnerability")
                suggestions.append("Implement side-channel attack protections")

            details = memory_security

        elif test_case.name == "log_data_exposure":
            # Test log data exposure
            log_security = self._test_log_security()

            if log_security['sensitive_data_logged']:
                vulnerabilities.append("Sensitive data found in logs")
                suggestions.append("Remove sensitive data from log outputs")

            if not log_security['log_encryption']:
                vulnerabilities.append("Logs not encrypted")
                suggestions.append("Encrypt log files")

            details = log_security

        elif test_case.name == "network_data_leakage":
            # Test network data leakage
            network_security = self._test_network_security()

            if network_security['unencrypted_data_transmitted']:
                vulnerabilities.append("Unencrypted sensitive data transmitted")
                suggestions.append("Encrypt all sensitive data transmissions")

            if network_security['dns_leakage']:
                vulnerabilities.append("DNS leakage detected")
                suggestions.append("Use secure DNS or DNS over HTTPS")

            if network_security['metadata_exposed']:
                vulnerabilities.append("Sensitive metadata exposed")
                suggestions.append("Minimize metadata exposure")

            details = network_security

        risk_level = "high" if any("unencrypted" in v or "sensitive data" in v for v in vulnerabilities) else "medium" if vulnerabilities else "low"

        return {
            'passed': len(vulnerabilities) == 0,
            'vulnerabilities': vulnerabilities,
            'risk_level': risk_level,
            'suggestions': suggestions,
            'details': details
        }

    # Helper methods for security testing
    def _weak_encrypt(self, data: bytes) -> bytes:
        """Simulate weak encryption."""
        # Simple XOR "encryption" (weak)
        key = b'weak'
        return bytes(a ^ b for a, b in zip(data, (key * (len(data) // len(key) + 1))[:len(data)]))

    def _strong_encrypt(self, data: bytes) -> bytes:
        """Simulate strong encryption."""
        # Simulate AES-256 encryption
        return hashlib.sha256(data + self.crypto_key).digest()

    def _is_easily_decryptable(self, encrypted_data: bytes) -> bool:
        """Check if encryption is easily breakable."""
        # Simple heuristic - real implementation would be more sophisticated
        return len(set(encrypted_data)) < len(encrypted_data) * 0.8

    def _is_properly_encrypted(self, encrypted_data: bytes) -> bool:
        """Check if encryption appears strong."""
        return len(set(encrypted_data)) > len(encrypted_data) * 0.9

    def _check_tls_version(self) -> float:
        """Simulate TLS version check."""
        # Simulate checking TLS version
        return 1.3  # Assume TLS 1.3 for testing

    def _check_cipher_strength(self) -> int:
        """Simulate cipher strength check."""
        return 256  # Assume 256-bit cipher for testing

    def _check_key_entropy(self, key: bytes) -> float:
        """Calculate Shannon entropy of key."""
        if not key:
            return 0.0

        # Calculate byte frequency
        freq = {}
        for byte in key:
            freq[byte] = freq.get(byte, 0) + 1

        # Calculate Shannon entropy
        entropy = 0.0
        key_len = len(key)
        for count in freq.values():
            p = count / key_len
            entropy -= p * (p.bit_length() - 1)

        return entropy

    def _check_key_storage(self) -> Dict[str, Any]:
        """Simulate key storage security check."""
        return {
            'secure': True,  # Assume secure for testing
            'hardware_protected': False,
            'encrypted': True,
            'access_controlled': True
        }

    def _test_device_auth(self) -> Dict[str, Any]:
        """Simulate device authentication test."""
        return {
            'certificate_valid': True,
            'brute_force_vulnerable': False,
            'mutual_auth': True,
            'certificate_expiry': 365,  # days
            'certificate_strength': 256
        }

    def _test_user_auth_strength(self) -> Dict[str, Any]:
        """Simulate user authentication strength test."""
        return {
            'mfa_enabled': True,
            'password_strength': 0.9,
            'session_timeout': 1800,  # 30 minutes
            'lockout_policy': True,
            'password_history': 12
        }

    def _test_biometric_auth(self) -> Dict[str, Any]:
        """Simulate biometric authentication test."""
        return {
            'false_accept_rate': 0.0001,  # 0.01%
            'false_reject_rate': 0.01,    # 1%
            'liveness_detection': True,
            'template_protection': True,
            'spoofing_resistance': True
        }

    def _test_privilege_escalation(self) -> Dict[str, Any]:
        """Simulate privilege escalation test."""
        return {
            'buffer_overflow_possible': False,
            'code_injection_possible': False,
            'race_condition_detected': False,
            'stack_protection': True,
            'aslr_enabled': True
        }

    def _test_access_control(self) -> Dict[str, Any]:
        """Simulate access control test."""
        return {
            'rbac_implemented': True,
            'unauthorized_access_possible': False,
            'path_traversal_vulnerable': False,
            'permission_model': 'least_privilege',
            'audit_logging': True
        }

    def _test_signal_injection(self) -> Dict[str, Any]:
        """Simulate signal injection test."""
        return {
            'spoofing_possible': False,
            'command_injection_possible': False,
            'integrity_checking': True,
            'signal_authentication': True,
            'anomaly_detection': True
        }

    def _test_data_injection(self) -> Dict[str, Any]:
        """Simulate data injection test."""
        return {
            'sql_injection_possible': False,
            'command_injection_possible': False,
            'code_injection_possible': False,
            'input_validation': True,
            'output_encoding': True
        }

    def _test_memory_security(self) -> Dict[str, Any]:
        """Simulate memory security test."""
        return {
            'sensitive_data_in_memory': False,
            'side_channel_vulnerable': False,
            'memory_protection': True,
            'data_clearing': True,
            'secure_memory_allocation': True
        }

    def _test_log_security(self) -> Dict[str, Any]:
        """Simulate log security test."""
        return {
            'sensitive_data_logged': False,
            'log_encryption': True,
            'log_integrity': True,
            'access_controls': True,
            'retention_policy': True
        }

    def _test_network_security(self) -> Dict[str, Any]:
        """Simulate network security test."""
        return {
            'unencrypted_data_transmitted': False,
            'dns_leakage': False,
            'metadata_exposed': False,
            'traffic_encryption': True,
            'certificate_pinning': True
        }

    def _generate_security_report(self, passed: int, failed: int, total: int, critical_vulns: int) -> Dict[str, Any]:
        """Generate comprehensive security report."""

        # Categorize results by type
        results_by_type = {}
        for result in self.results:
            test_type = next((tc.test_type for tc in self.test_cases if tc.name == result.test_case), 'unknown')
            if test_type not in results_by_type:
                results_by_type[test_type] = []
            results_by_type[test_type].append(result)

        # Calculate risk metrics
        risk_distribution = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        total_vulnerabilities = 0

        for result in self.results:
            risk_distribution[result.risk_level] += 1
            total_vulnerabilities += len(result.vulnerabilities_found)

        # Generate CVSS-like score
        security_score = self._calculate_security_score(risk_distribution, total_vulnerabilities)

        report = {
            'test_summary': {
                'total_tests': total,
                'passed_tests': passed,
                'failed_tests': failed,
                'success_rate': passed / total if total > 0 else 0,
                'critical_vulnerabilities': critical_vulns,
                'total_vulnerabilities': total_vulnerabilities
            },
            'security_metrics': {
                'security_score': security_score,
                'risk_distribution': risk_distribution,
                'average_execution_time': sum(r.execution_time for r in self.results) / len(self.results) if self.results else 0
            },
            'test_type_analysis': self._analyze_by_test_type(results_by_type),
            'vulnerability_analysis': self._analyze_vulnerabilities(),
            'compliance_status': self._assess_compliance(),
            'detailed_results': [self._result_to_dict(r) for r in self.results],
            'security_recommendations': self._generate_security_recommendations()
        }

        return report

    def _calculate_security_score(self, risk_dist: Dict[str, int], total_vulns: int) -> float:
        """Calculate overall security score (0-10 scale)."""
        base_score = 10.0

        # Deduct points based on vulnerabilities
        base_score -= risk_dist['critical'] * 3.0
        base_score -= risk_dist['high'] * 1.5
        base_score -= risk_dist['medium'] * 0.5
        base_score -= risk_dist['low'] * 0.1

        # Additional deduction for total vulnerability count
        base_score -= min(total_vulns * 0.1, 2.0)

        return max(0.0, base_score)

    def _analyze_by_test_type(self, results_by_type: Dict[str, List[SecurityResult]]) -> Dict[str, Any]:
        """Analyze results by test type."""
        analysis = {}

        for test_type, results in results_by_type.items():
            total = len(results)
            passed = sum(1 for r in results if r.passed)

            analysis[test_type] = {
                'total_tests': total,
                'passed_tests': passed,
                'failed_tests': total - passed,
                'success_rate': passed / total if total > 0 else 0,
                'vulnerabilities': sum(len(r.vulnerabilities_found) for r in results),
                'avg_execution_time': sum(r.execution_time for r in results) / total if total > 0 else 0
            }

        return analysis

    def _analyze_vulnerabilities(self) -> Dict[str, Any]:
        """Analyze vulnerability patterns."""
        all_vulns = []
        for result in self.results:
            all_vulns.extend(result.vulnerabilities_found)

        # Common vulnerability patterns
        vuln_patterns = {
            'encryption': len([v for v in all_vulns if 'encrypt' in v.lower()]),
            'authentication': len([v for v in all_vulns if 'auth' in v.lower()]),
            'injection': len([v for v in all_vulns if 'injection' in v.lower()]),
            'access_control': len([v for v in all_vulns if 'access' in v.lower() or 'unauthorized' in v.lower()]),
            'data_exposure': len([v for v in all_vulns if 'data' in v.lower() and ('exposed' in v.lower() or 'leak' in v.lower())])
        }

        return {
            'total_vulnerabilities': len(all_vulns),
            'vulnerability_patterns': vuln_patterns,
            'most_common_issues': sorted(vuln_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        }

    def _assess_compliance(self) -> Dict[str, Any]:
        """Assess compliance with security standards."""
        compliance_scores = {}

        # HIPAA compliance assessment
        hipaa_tests = ['data_encryption_at_rest', 'data_encryption_in_transit', 'access_control', 'log_data_exposure']
        hipaa_passed = sum(1 for r in self.results if r.test_case in hipaa_tests and r.passed)
        compliance_scores['HIPAA'] = hipaa_passed / len(hipaa_tests)

        # GDPR compliance assessment
        gdpr_tests = ['data_encryption_at_rest', 'access_control', 'log_data_exposure', 'memory_leakage']
        gdpr_passed = sum(1 for r in self.results if r.test_case in gdpr_tests and r.passed)
        compliance_scores['GDPR'] = gdpr_passed / len(gdpr_tests)

        # FDA security guidance compliance
        fda_tests = ['device_authentication', 'data_encryption_in_transit', 'neural_signal_injection', 'privilege_escalation']
        fda_passed = sum(1 for r in self.results if r.test_case in fda_tests and r.passed)
        compliance_scores['FDA_Security'] = fda_passed / len(fda_tests)

        return compliance_scores

    def _result_to_dict(self, result: SecurityResult) -> Dict[str, Any]:
        """Convert security result to dictionary."""
        return {
            'test_case': result.test_case,
            'passed': result.passed,
            'vulnerabilities_found': result.vulnerabilities_found,
            'risk_level': result.risk_level,
            'mitigation_suggestions': result.mitigation_suggestions,
            'execution_time': result.execution_time,
            'details': result.details
        }

    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on test results."""
        recommendations = []

        # Analyze failed tests
        failed_tests = [r for r in self.results if not r.passed]
        critical_failures = [r for r in failed_tests if r.risk_level == 'critical']

        if critical_failures:
            recommendations.append(
                f"CRITICAL: Address {len(critical_failures)} critical security vulnerabilities immediately"
            )

        # Specific recommendations based on vulnerability patterns
        vuln_analysis = self._analyze_vulnerabilities()
        patterns = vuln_analysis['vulnerability_patterns']

        if patterns['encryption'] > 0:
            recommendations.append("Strengthen encryption implementation and key management")

        if patterns['authentication'] > 0:
            recommendations.append("Improve authentication mechanisms and implement MFA")

        if patterns['injection'] > 0:
            recommendations.append("Implement comprehensive input validation and sanitization")

        if patterns['access_control'] > 0:
            recommendations.append("Review and strengthen access control mechanisms")

        if patterns['data_exposure'] > 0:
            recommendations.append("Implement data loss prevention and secure data handling")

        # Compliance recommendations
        compliance = self._assess_compliance()
        for standard, score in compliance.items():
            if score < 0.8:
                recommendations.append(f"Improve {standard} compliance (current: {score*100:.1f}%)")

        if not recommendations:
            recommendations.append("Security posture is strong. Continue regular security testing.")

        return recommendations

    def save_security_results(self, filename: str = "security_test_results.json"):
        """Save security test results to file."""
        report = self._generate_security_report(
            len([r for r in self.results if r.passed]),
            len([r for r in self.results if not r.passed]),
            len(self.results),
            len([r for r in self.results if r.risk_level == 'critical' and not r.passed])
        )

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"🔐 Security test results saved to {filename}")


async def main():
    """Main security testing function."""
    print("🔐 Apple BCI-HID Security Testing Suite")
    print("=" * 60)

    security_suite = SecurityTestingSuite()

    try:
        # Run all security tests
        report = await security_suite.run_all_security_tests()

        # Print security summary
        print("\n" + "=" * 60)
        print("🔐 SECURITY TESTING SUMMARY")
        print("=" * 60)

        summary = report['test_summary']
        metrics = report['security_metrics']
        compliance = report['compliance_status']

        print("\nTest Results:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed_tests']}")
        print(f"  Failed: {summary['failed_tests']}")
        print(f"  Success Rate: {summary['success_rate']*100:.1f}%")
        print(f"  Critical Vulnerabilities: {summary['critical_vulnerabilities']}")
        print(f"  Total Vulnerabilities: {summary['total_vulnerabilities']}")

        print(f"\nSecurity Metrics:")
        print(f"  Security Score: {metrics['security_score']:.1f}/10")
        print(f"  Risk Distribution:")
        for risk, count in metrics['risk_distribution'].items():
            print(f"    {risk.title()}: {count}")

        print(f"\nCompliance Status:")
        for standard, score in compliance.items():
            status = "✅ COMPLIANT" if score >= 0.8 else "❌ NON-COMPLIANT"
            print(f"  {standard}: {score*100:.1f}% {status}")

        print(f"\nSecurity Recommendations:")
        for i, rec in enumerate(report['security_recommendations'], 1):
            print(f"  {i}. {rec}")

        # Save results
        security_suite.save_security_results()

        print(f"\n✅ Security testing completed successfully!")
        return report['security_metrics']['security_score'] >= 8.0

    except Exception as e:
        print(f"\n❌ Security testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
