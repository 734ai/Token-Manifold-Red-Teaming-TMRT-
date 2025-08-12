#!/usr/bin/env python3
"""
Final Submission Validator
Performs comprehensive validation to ensure the TMRT project is 100% ready for competition submission.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinalSubmissionValidator:
    """Comprehensive validation for competition submission readiness."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.submission_dir = self.base_dir / "competition_submission"
        self.validation_results = {}
        self.issues = []
        self.recommendations = []
        
    def validate_all(self):
        """Run all validation checks."""
        logger.info("🔍 Starting Comprehensive Submission Validation")
        logger.info("=" * 80)
        
        # Core validation checks
        checks = [
            ("Competition Files", self._validate_competition_files),
            ("Kaggle Format", self._validate_kaggle_format),
            ("Finding Quality", self._validate_finding_quality),
            ("Documentation", self._validate_documentation),
            ("Reproducibility", self._validate_reproducibility),
            ("Technical Completeness", self._validate_technical_completeness),
            ("Competition Compliance", self._validate_competition_compliance),
            ("Final Readiness", self._validate_final_readiness)
        ]
        
        for check_name, check_func in checks:
            logger.info(f"Running {check_name} validation...")
            try:
                result = check_func()
                self.validation_results[check_name] = result
                if result.get('status') == 'PASS':
                    logger.info(f"  ✅ {check_name}: PASSED")
                else:
                    logger.warning(f"  ⚠️  {check_name}: {result.get('status', 'UNKNOWN')}")
                    if 'issues' in result:
                        self.issues.extend(result['issues'])
            except Exception as e:
                logger.error(f"  ❌ {check_name}: ERROR - {str(e)}")
                self.validation_results[check_name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        # Generate final report
        return self._generate_final_report()
        
    def _validate_competition_files(self):
        """Validate all required competition files are present."""
        required_files = [
            "tmrt_submission_optimized_20250812_102323_kaggle.json",
            "tmrt_submission_optimized_20250812_102323_complete.json",
            "tmrt_submission_optimized_20250812_102323_executive_summary.md"
        ]
        
        result = {
            'status': 'PASS',
            'files_checked': len(required_files),
            'files_found': 0,
            'missing_files': []
        }
        
        for file in required_files:
            file_path = self.submission_dir / file
            if file_path.exists():
                result['files_found'] += 1
                logger.info(f"    ✓ Found: {file}")
            else:
                result['missing_files'].append(file)
                logger.warning(f"    ✗ Missing: {file}")
        
        if result['missing_files']:
            result['status'] = 'FAIL'
            result['issues'] = [f"Missing required file: {f}" for f in result['missing_files']]
        
        return result
        
    def _validate_kaggle_format(self):
        """Validate Kaggle JSON format compliance."""
        kaggle_file = self.submission_dir / "tmrt_submission_optimized_20250812_102323_kaggle.json"
        
        result = {
            'status': 'PASS',
            'format_valid': False,
            'required_fields': [],
            'missing_fields': []
        }
        
        if not kaggle_file.exists():
            result['status'] = 'FAIL'
            result['issues'] = ['Kaggle submission file not found']
            return result
        
        try:
            with open(kaggle_file, 'r') as f:
                data = json.load(f)
            
            required_fields = [
                'competition',
                'team_name',
                'submission_date',
                'methodology',
                'findings_summary',
                'competition_entries'
            ]
            
            for field in required_fields:
                if field in data:
                    result['required_fields'].append(field)
                    logger.info(f"    ✓ Found field: {field}")
                else:
                    result['missing_fields'].append(field)
                    logger.warning(f"    ✗ Missing field: {field}")
            
            # Validate competition entries
            if 'competition_entries' in data:
                entries = data['competition_entries']
                result['total_entries'] = len(entries)
                result['entries_valid'] = True
                
                for i, entry in enumerate(entries):
                    entry_fields = ['finding_id', 'title', 'category', 'risk_assessment', 'novelty_score']
                    for field in entry_fields:
                        if field not in entry:
                            result['entries_valid'] = False
                            result['missing_fields'].append(f"Entry {i+1}: {field}")
            
            if result['missing_fields']:
                result['status'] = 'FAIL'
                result['issues'] = [f"Missing field: {f}" for f in result['missing_fields']]
            else:
                result['format_valid'] = True
                
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
            
        return result
        
    def _validate_finding_quality(self):
        """Validate the quality and completeness of findings."""
        kaggle_file = self.submission_dir / "tmrt_submission_optimized_20250812_102323_kaggle.json"
        
        result = {
            'status': 'PASS',
            'total_findings': 0,
            'high_risk_findings': 0,
            'critical_findings': 0,
            'average_novelty': 0.0,
            'quality_issues': []
        }
        
        try:
            with open(kaggle_file, 'r') as f:
                data = json.load(f)
            
            if 'competition_entries' not in data:
                result['status'] = 'FAIL'
                result['issues'] = ['No competition entries found']
                return result
            
            entries = data['competition_entries']
            result['total_findings'] = len(entries)
            
            novelty_scores = []
            
            for entry in entries:
                # Count risk levels
                risk = entry.get('risk_assessment', '').upper()
                if risk == 'HIGH':
                    result['high_risk_findings'] += 1
                elif risk == 'CRITICAL':
                    result['critical_findings'] += 1
                
                # Collect novelty scores
                novelty = entry.get('novelty_score', 0)
                if isinstance(novelty, (int, float)):
                    novelty_scores.append(novelty)
                else:
                    result['quality_issues'].append(f"Invalid novelty score in {entry.get('finding_id', 'unknown')}")
            
            # Calculate average novelty
            if novelty_scores:
                result['average_novelty'] = sum(novelty_scores) / len(novelty_scores)
            
            # Quality checks
            if result['total_findings'] < 5:
                result['quality_issues'].append(f"Only {result['total_findings']} findings (recommend 10+)")
            
            if result['average_novelty'] < 0.75:
                result['quality_issues'].append(f"Average novelty {result['average_novelty']:.2f} below recommended 0.75")
            
            if result['critical_findings'] == 0:
                result['quality_issues'].append("No critical findings (recommend at least 1)")
            
            if result['quality_issues']:
                result['status'] = 'WARNING'
                
            logger.info(f"    Findings: {result['total_findings']} total, {result['critical_findings']} critical, {result['high_risk_findings']} high-risk")
            logger.info(f"    Average novelty: {result['average_novelty']:.3f}")
            
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
            
        return result
        
    def _validate_documentation(self):
        """Validate documentation completeness."""
        result = {
            'status': 'PASS',
            'documents_found': 0,
            'missing_documents': []
        }
        
        required_docs = [
            "tmrt_submission_optimized_20250812_102323_executive_summary.md",
            "../README.md",
            "../IMPLEMENTATION_COMPLETE.md"
        ]
        
        for doc in required_docs:
            doc_path = self.submission_dir / doc
            if doc_path.exists():
                result['documents_found'] += 1
                size = doc_path.stat().st_size / 1024  # KB
                logger.info(f"    ✓ Found: {doc} ({size:.1f} KB)")
            else:
                result['missing_documents'].append(doc)
                logger.warning(f"    ✗ Missing: {doc}")
        
        if result['missing_documents']:
            result['status'] = 'WARNING'
            result['issues'] = [f"Missing documentation: {d}" for d in result['missing_documents']]
        
        return result
        
    def _validate_reproducibility(self):
        """Validate reproducibility components."""
        result = {
            'status': 'PASS',
            'components_found': 0,
            'missing_components': []
        }
        
        reproducibility_components = [
            "../docker/Dockerfile",
            "../requirements.txt",
            "../pyproject.toml",
            "../src/tmrt/__init__.py"
        ]
        
        for component in reproducibility_components:
            comp_path = self.submission_dir / component
            if comp_path.exists():
                result['components_found'] += 1
                logger.info(f"    ✓ Found: {component}")
            else:
                result['missing_components'].append(component)
                logger.warning(f"    ✗ Missing: {component}")
        
        if result['missing_components']:
            result['status'] = 'WARNING'
            result['issues'] = [f"Missing component: {c}" for c in result['missing_components']]
        
        return result
        
    def _validate_technical_completeness(self):
        """Validate technical implementation completeness."""
        result = {
            'status': 'PASS',
            'modules_found': 0,
            'missing_modules': []
        }
        
        core_modules = [
            "../src/tmrt/unicode_mutators.py",
            "../src/tmrt/embedding_optimizer.py",
            "../src/tmrt/scaffolder.py",
            "../src/tmrt/novelty_detector.py",
            "../src/tmrt/search_controller.py",
            "../src/tmrt/verifier.py"
        ]
        
        for module in core_modules:
            mod_path = self.submission_dir / module
            if mod_path.exists():
                result['modules_found'] += 1
                logger.info(f"    ✓ Found: {module}")
            else:
                result['missing_modules'].append(module)
                logger.warning(f"    ✗ Missing: {module}")
        
        if result['missing_modules']:
            result['status'] = 'FAIL'
            result['issues'] = [f"Missing core module: {m}" for m in result['missing_modules']]
        
        return result
        
    def _validate_competition_compliance(self):
        """Validate compliance with competition rules."""
        kaggle_file = self.submission_dir / "tmrt_submission_optimized_20250812_102323_kaggle.json"
        
        result = {
            'status': 'PASS',
            'compliance_checks': [],
            'violations': []
        }
        
        try:
            with open(kaggle_file, 'r') as f:
                data = json.load(f)
            
            # Check file size (should be reasonable for submission)
            file_size_mb = kaggle_file.stat().st_size / (1024 * 1024)
            if file_size_mb > 10:
                result['violations'].append(f"Submission file too large: {file_size_mb:.1f} MB")
            
            # Check entry count (typically 1-10 allowed)
            entries = data.get('competition_entries', [])
            if len(entries) > 10:
                result['violations'].append(f"Too many entries: {len(entries)} (max usually 10)")
            elif len(entries) < 1:
                result['violations'].append("No competition entries found")
            
            # Check for required metadata
            if not data.get('methodology'):
                result['violations'].append("Missing methodology description")
            
            result['compliance_checks'] = [
                f"File size: {file_size_mb:.2f} MB",
                f"Entry count: {len(entries)}",
                f"Has methodology: {'Yes' if data.get('methodology') else 'No'}"
            ]
            
            if result['violations']:
                result['status'] = 'FAIL'
                result['issues'] = result['violations']
            
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
            
        return result
        
    def _validate_final_readiness(self):
        """Final readiness check."""
        result = {
            'status': 'PASS',
            'readiness_score': 0.0,
            'critical_issues': [],
            'recommendations': []
        }
        
        # Calculate readiness score based on previous validations
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for r in self.validation_results.values() if r.get('status') == 'PASS')
        
        result['readiness_score'] = passed_checks / total_checks if total_checks > 0 else 0.0
        
        if result['readiness_score'] >= 0.9:
            result['status'] = 'READY'
            result['recommendations'].append("✅ Submission is ready for competition!")
        elif result['readiness_score'] >= 0.7:
            result['status'] = 'MOSTLY_READY'
            result['recommendations'].append("⚠️ Minor issues to address before submission")
        else:
            result['status'] = 'NOT_READY'
            result['critical_issues'].append("❌ Major issues must be resolved before submission")
        
        # Add timeline recommendations
        days_until_deadline = 14  # August 26, 2025
        if days_until_deadline <= 3:
            result['recommendations'].append("🚨 URGENT: Submit within 24-48 hours!")
        elif days_until_deadline <= 7:
            result['recommendations'].append("⏰ Submit within next 2-3 days for safety")
        else:
            result['recommendations'].append("📅 Good timing - can submit strategically")
        
        return result
        
    def _generate_final_report(self):
        """Generate comprehensive final validation report."""
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'competition_deadline': '2025-08-26',
            'validation_results': self.validation_results,
            'overall_issues': self.issues,
            'recommendations': self.recommendations,
            'final_assessment': self._get_final_assessment()
        }
        
        # Save report
        report_file = self.base_dir / "final_submission" / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Final validation report saved to: {report_file}")
        
        return report
        
    def _get_final_assessment(self):
        """Get final assessment based on all validations."""
        readiness_result = self.validation_results.get('Final Readiness', {})
        readiness_score = readiness_result.get('readiness_score', 0.0)
        
        if readiness_score >= 0.95:
            return {
                'status': 'EXCELLENT',
                'message': '🏆 SUBMISSION READY - EXCELLENT PREPARATION',
                'action': 'Ready to submit immediately'
            }
        elif readiness_score >= 0.85:
            return {
                'status': 'GOOD',
                'message': '✅ SUBMISSION READY - GOOD PREPARATION',
                'action': 'Address minor issues then submit'
            }
        elif readiness_score >= 0.70:
            return {
                'status': 'NEEDS_WORK',
                'message': '⚠️ NEEDS IMPROVEMENT BEFORE SUBMISSION',
                'action': 'Address issues before submitting'
            }
        else:
            return {
                'status': 'NOT_READY',
                'message': '❌ NOT READY FOR SUBMISSION',
                'action': 'Major work required before submission'
            }

def main():
    """Main validation execution."""
    print("🔍 TMRT Final Submission Validation")
    print("=" * 80)
    print("Competition: OpenAI GPT-OSS-20b Red Teaming Challenge")
    print("Validation Goal: Ensure 100% Submission Readiness")
    print("Process Started:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 80)
    
    validator = FinalSubmissionValidator()
    report = validator.validate_all()
    
    print("\n" + "=" * 80)
    print("🏆 Final Validation Complete!")
    print("=" * 80)
    
    final_assessment = report['final_assessment']
    print(f"Final Status: {final_assessment['message']}")
    print(f"Recommended Action: {final_assessment['action']}")
    
    if report.get('overall_issues'):
        print(f"\n⚠️ Issues Found: {len(report['overall_issues'])}")
        for issue in report['overall_issues'][:5]:  # Show first 5 issues
            print(f"  • {issue}")
        if len(report['overall_issues']) > 5:
            print(f"  • ... and {len(report['overall_issues']) - 5} more issues")
    
    # Show readiness score
    readiness_result = report['validation_results'].get('Final Readiness', {})
    readiness_score = readiness_result.get('readiness_score', 0.0)
    print(f"\n📊 Overall Readiness Score: {readiness_score:.1%}")
    
    print("\n🚀 Validation complete!")
    
    return 0 if final_assessment['status'] in ['EXCELLENT', 'GOOD'] else 1

if __name__ == "__main__":
    sys.exit(main())
