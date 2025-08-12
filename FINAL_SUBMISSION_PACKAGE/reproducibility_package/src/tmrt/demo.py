#!/usr/bin/env python3
"""
TMRT Demo Script

Demonstrates the Token-Manifold Red Teaming framework with a safe toy model.
This script can be run to test the pipeline before using it on the target model.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tmrt import SearchController, load_config, setup_logging
from tmrt.utils import sanitize_output


def run_toy_demo(config_path: str = "configs/toy_demo.yaml", seed: int = 42):
    """Run toy demo with safe model."""
    print("=== TMRT Toy Demo ===")
    print("This demo uses a small, safe model to test the framework.")
    print()
    
    # Load configuration
    try:
        config = load_config(config_path)
        print(f"Loaded config from {config_path}")
    except Exception as e:
        print(f"Failed to load config: {e}")
        return
    
    # Initialize search controller
    try:
        controller = SearchController(
            model_name=config['model_name'],
            config=config,
            seed=seed
        )
        print(f"Initialized SearchController with model: {config['model_name']}")
    except Exception as e:
        print(f"Failed to initialize controller: {e}")
        print("Note: This demo requires transformers and a working model.")
        return
    
    # Run quick test first
    print("\\n--- Running Quick Test ---")
    test_prompts = [
        "Tell me a story about cats",
        "Explain how photosynthesis works",
        "What is machine learning"
    ]
    
    try:
        quick_results = controller.quick_test_run(test_prompts, max_variants=3)
        
        print(f"Tested {quick_results['total_variants_tested']} variants")
        print(f"Successful attacks: {quick_results['successful_attacks']}")
        
        print("\\nTop 3 results:")
        for i, result in enumerate(quick_results['test_results'][:3], 1):
            print(f"{i}. Attack Success: {result['attack_success']:.3f}")
            print(f"   Mutation: {result['mutation_type']}")
            print(f"   Preview: {result['variant_text'][:100]}...")
            print()
    
    except Exception as e:
        print(f"Quick test failed: {e}")
    
    # Run short evolutionary search  
    print("\\n--- Running Evolutionary Search ---")
    try:
        search_results = controller.run_search(
            seed_prompts=config.get('seed_prompts'),
            num_generations=config.get('num_generations', 5)
        )
        
        print(f"Search completed in {search_results['search_duration_seconds']:.1f}s")
        print(f"Best fitness: {search_results['best_fitness']:.4f}")
        print(f"Final avg fitness: {search_results['statistics']['final_avg_fitness']:.4f}")
        
        # Show top results
        best_individuals = search_results['best_individuals'][:3]
        print(f"\\nTop {len(best_individuals)} individuals:")
        
        for i, individual in enumerate(best_individuals, 1):
            print(f"{i}. Fitness: {individual.get('fitness_score', 0):.3f}")
            print(f"   Scaffold: {individual['scaffold_config']['scaffold_type']}")
            print(f"   Response: {individual.get('response_preview', 'N/A')[:80]}...")
            print()
        
    except Exception as e:
        print(f"Evolutionary search failed: {e}")
    
    print("\\n=== Demo Complete ===")
    print("Check the outputs/ and findings/ directories for detailed results.")


def run_unicode_test():
    """Test Unicode mutations in isolation."""
    print("=== Unicode Mutation Test ===")
    
    from tmrt.unicode_mutators import UnicodeMutator
    
    mutator = UnicodeMutator(seed=42)
    
    test_text = "Write a tutorial on data analysis"
    print(f"Original text: {test_text}")
    print()
    
    # Test different mutation types
    variants = mutator.generate_variants(test_text, num_variants=5)
    
    for i, (variant, mutation_type, score) in enumerate(variants, 1):
        print(f"{i}. {mutation_type} (score: {score:.3f})")
        print(f"   Result: {repr(variant)}")
        print()


def run_scaffolding_test():
    """Test role scaffolding in isolation."""
    print("=== Role Scaffolding Test ===")
    
    from tmrt.scaffolder import RoleScaffolder
    
    scaffolder = RoleScaffolder(seed=42)
    
    # Test different scaffold types
    print("1. Email scaffold:")
    email_scaffold = scaffolder.generate_email_scaffold(
        sender_role="authority",
        urgency_level="high"
    )
    print(sanitize_output(email_scaffold))
    print()
    
    print("2. Chat transcript:")
    chat_scaffold = scaffolder.generate_chat_transcript(build_rapport=True)
    print(sanitize_output(chat_scaffold))
    print()
    
    print("3. Multi-role scenario:")
    scenario = scaffolder.generate_multi_role_scenario("incident_response")
    print(sanitize_output(scenario))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="TMRT Demo Script")
    parser.add_argument(
        '--mode',
        choices=['full', 'unicode', 'scaffold'],
        default='full',
        help='Demo mode to run'
    )
    parser.add_argument(
        '--config',
        default='configs/toy_demo.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    if args.mode == 'full':
        run_toy_demo(args.config, args.seed)
    elif args.mode == 'unicode':
        run_unicode_test()
    elif args.mode == 'scaffold':
        run_scaffolding_test()


if __name__ == "__main__":
    main()
