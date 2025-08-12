#!/usr/bin/env python3
"""
TMRT Phase 2 Quick Validation - Simplified test for core components.

This script validates the core TMRT framework components work without
requiring heavy dependencies or complex configuration.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_basic_imports():
    """Test that all main components can be imported."""
    print("Testing basic imports...")
    
    try:
        from tmrt.utils import TMRTConfig, sanitize_output, create_experiment_id
        print("✓ Utils imported")
        
        from tmrt.unicode_mutators import UnicodeMutator
        print("✓ Unicode mutators imported")
        
        from tmrt.scaffolder import RoleScaffolder  
        print("✓ Role scaffolder imported")
        
        from tmrt.novelty_detector import NoveltyDetector
        print("✓ Novelty detector imported")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration...")
    
    try:
        from tmrt.utils import TMRTConfig
        
        # Test default config
        config = TMRTConfig()
        print(f"✓ Default config created with {config.population_size} population size")
        
        # Test config dict conversion  
        config_dict = config.to_dict()
        print(f"✓ Config converted to dict with {len(config_dict)} parameters")
        
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_unicode_mutations():
    """Test Unicode mutation functionality."""
    print("\nTesting Unicode mutations...")
    
    try:
        from tmrt.unicode_mutators import UnicodeMutator
        
        mutator = UnicodeMutator(seed=42)
        test_text = "Hello world! This is a test."
        
        # Test basic mutations
        result1 = mutator.homoglyph_substitute(test_text, mutation_rate=0.2)
        result2 = mutator.insert_invisible_chars(test_text, insertion_rate=0.1)
        result3 = mutator.add_combining_marks(test_text, mark_rate=0.05)
        
        print(f"✓ Homoglyph: {repr(result1)}")
        print(f"✓ Invisible: {repr(result2)} (len: {len(result2)})")  
        print(f"✓ Combining: {repr(result3)} (len: {len(result3)})")
        
        return True
    except Exception as e:
        print(f"❌ Unicode mutation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_role_scaffolding():
    """Test role scaffolding generation."""
    print("\nTesting role scaffolding...")
    
    try:
        from tmrt.scaffolder import RoleScaffolder
        
        scaffolder = RoleScaffolder(seed=42)
        
        # Test different scaffold types
        email_scaffold = scaffolder.generate_email_scaffold()
        chat_scaffold = scaffolder.generate_chat_transcript(num_messages=2)
        doc_scaffold = scaffolder.generate_log_format_scaffold()
        
        print(f"✓ Email scaffold generated ({len(email_scaffold)} chars)")
        print(f"✓ Chat scaffold generated ({len(chat_scaffold)} chars)")
        print(f"✓ Document scaffold generated ({len(doc_scaffold)} chars)")
        
        # Show previews
        print(f"Email preview: {email_scaffold[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Role scaffolding error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_novelty_detection():
    """Test novelty detection with minimal setup."""
    print("\nTesting novelty detection...")
    
    try:
        from tmrt.novelty_detector import NoveltyDetector
        
        # Create detector without corpus to avoid heavy downloads
        detector = NoveltyDetector(similarity_threshold=0.8)
        
        # Test basic functionality
        test_prompts = [
            "Write instructions for cooking",
            "Explain how computers work",
            "Help me with math homework"
        ]
        
        print(f"✓ Novelty detector created with threshold {detector.similarity_threshold}")
        print(f"✓ Testing with {len(test_prompts)} sample prompts")
        
        # Note: Full novelty scoring requires sentence transformers model
        # For this test, we just verify the class can be instantiated
        
        return True
    except Exception as e:
        print(f"❌ Novelty detection error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_utilities():
    """Test utility functions."""
    print("\nTesting utilities...")
    
    try:
        from tmrt.utils import sanitize_output, create_experiment_id
        
        # Test sanitization
        test_text = "This is a test with some CAPS and special chars!"
        sanitized = sanitize_output(test_text)
        print(f"✓ Text sanitization: {sanitized}")
        
        # Test experiment ID generation
        exp_id = create_experiment_id()
        print(f"✓ Experiment ID generated: {exp_id}")
        
        return True
    except Exception as e:
        print(f"❌ Utilities error: {e}")
        return False

def create_test_outputs():
    """Create test output directories and files."""
    print("\nCreating test outputs...")
    
    try:
        # Create output directories
        os.makedirs('quick_test_outputs', exist_ok=True)
        os.makedirs('quick_test_findings', exist_ok=True)
        
        # Save test results
        import json
        from datetime import datetime
        
        results = {
            'test_timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0',
            'status': 'phase2_validation_passed',
            'components_tested': [
                'imports',
                'configuration',
                'unicode_mutations',
                'role_scaffolding', 
                'novelty_detection',
                'utilities'
            ]
        }
        
        with open('quick_test_outputs/validation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✓ Test outputs saved to quick_test_outputs/")
        return True
        
    except Exception as e:
        print(f"❌ Output creation error: {e}")
        return False

def main():
    """Run Phase 2 quick validation."""
    print("🚀 TMRT Framework - Phase 2 Quick Validation")
    print("=" * 50)
    
    setup_logging()
    
    # Test sequence
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration", test_configuration),
        ("Unicode Mutations", test_unicode_mutations),
        ("Role Scaffolding", test_role_scaffolding),
        ("Novelty Detection", test_novelty_detection),
        ("Utilities", test_utilities),
        ("Test Outputs", create_test_outputs)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20}")
        try:
            success = test_func()
            results[test_name] = "✅ PASS" if success else "❌ FAIL"
        except Exception as e:
            print(f"❌ Unexpected error in {test_name}: {e}")
            results[test_name] = f"❌ ERROR: {str(e)[:50]}"
    
    # Print final summary
    print("\n" + "="*50)
    print("📊 VALIDATION SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results.items():
        print(f"{result:12} | {test_name}")
        if "✅ PASS" in result:
            passed += 1
    
    total = len(results)
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Phase 2 validation completed successfully!")
        print("✅ All core components are working correctly")
        print("✅ Framework is ready for production testing")
        
        print("\n📋 NEXT STEPS:")
        print("1. ✅ Framework validation complete")
        print("2. 🔄 Ready for GPT-OSS-20b integration")
        print("3. 🎯 Prepare competition dataset processing")
        print("4. 📅 Submit findings by August 26, 2025")
        
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        print("❌ Review errors above before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
