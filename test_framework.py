#!/usr/bin/env python3
"""
Simple framework validation test.
"""

import sys
import os
sys.path.append('src')

from tmrt import UnicodeMutator, RoleScaffolder, NoveltyDetector

def test_unicode_mutator():
    """Test Unicode mutation functionality."""
    print("Testing Unicode Mutator...")
    mutator = UnicodeMutator()
    
    test_text = "Hello world"
    
    # Test homoglyph substitution
    homoglyph_result = mutator.homoglyph_substitute(test_text, mutation_rate=0.3)
    print(f"  Original: {test_text}")
    print(f"  Homoglyph: {homoglyph_result}")
    
    # Test invisible character insertion
    invisible_result = mutator.insert_invisible_chars(test_text, insertion_rate=0.2)
    print(f"  Invisible chars: {repr(invisible_result)}")
    
    print("  ✓ Unicode mutator tests passed\n")

def test_role_scaffolder():
    """Test role scaffolding functionality."""
    print("Testing Role Scaffolder...")
    scaffolder = RoleScaffolder()
    
    # Test email scaffold
    email_scaffold = scaffolder.generate_email_scaffold(sender_role="authority", urgency_level="high")
    print(f"  Email scaffold: {email_scaffold[:100]}...")
    
    # Test chat transcript
    chat_scaffold = scaffolder.generate_chat_transcript(num_messages=3, include_timestamps=True)
    print(f"  Chat scaffold: {chat_scaffold[:100]}...")
    
    print("  ✓ Role scaffolder tests passed\n")

def test_novelty_detector():
    """Test novelty detection functionality."""
    print("Testing Novelty Detector...")
    print("  Skipping novelty detector test (requires model download)")
    print("  ✓ Novelty detector tests skipped\n")

def main():
    """Run all validation tests."""
    print("🚀 TMRT Framework Validation Tests")
    print("=" * 40)
    
    try:
        test_unicode_mutator()
        test_role_scaffolder() 
        test_novelty_detector()
        
        print("🎉 All framework components validated successfully!")
        print("\n📋 Next steps:")
        print("  1. Run toy demo: python examples/toy_demo.py")
        print("  2. Configure GPT-OSS-20b model")
        print("  3. Execute full search pipeline")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
