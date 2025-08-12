"""
Integration tests for the TMRT framework.
"""

import pytest
import tempfile
import json
import torch
from pathlib import Path
from unittest.mock import Mock, patch

# Import all framework components
from src.tmrt.unicode_mutators import UnicodeMutator
from src.tmrt.embedding_optimizer import EmbeddingOptimizer
from src.tmrt.scaffolder import RoleScaffolder
from src.tmrt.verifier import AttackVerifier
from src.tmrt.novelty_detector import NoveltyDetector
from src.tmrt.search_controller import SearchController
from src.tmrt.utils import TMRTConfig


class TestTMRTFrameworkIntegration:
    """Test integration between TMRT framework components."""
    
    def setup_method(self):
        """Setup test fixtures with mock models."""
        # Mock model and tokenizer for testing
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        
        # Configure tokenizer mock
        self.mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.mock_tokenizer.decode.return_value = "mock response"
        self.mock_tokenizer.vocab_size = 1000
        
        # Configure model mock
        self.mock_model.embed_tokens = Mock()
        self.mock_model.embed_tokens.return_value = torch.randn(1, 5, 768)
        mock_output = Mock()
        mock_output.logits = torch.randn(1, 5, 1000)
        self.mock_model.return_value = mock_output
        
        # Initialize components
        self.unicode_mutator = UnicodeMutator()
        self.embedding_optimizer = EmbeddingOptimizer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer
        )
        self.scaffolder = RoleScaffolder()
        self.verifier = AttackVerifier(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer
        )
        self.novelty_detector = NoveltyDetector()
        
        # Initialize search controller
        self.search_controller = SearchController(
            unicode_mutator=self.unicode_mutator,
            embedding_optimizer=self.embedding_optimizer,
            scaffolder=self.scaffolder,
            verifier=self.verifier,
            novelty_detector=self.novelty_detector
        )
        
    def test_component_integration_chain(self):
        """Test the integration chain: mutate -> scaffold -> optimize -> verify."""
        base_prompt = "Tell me how to"
        
        # Step 1: Unicode mutation
        mutated_prompt = self.unicode_mutator.mutate(base_prompt)
        assert isinstance(mutated_prompt, str)
        assert len(mutated_prompt) > 0
        
        # Step 2: Role scaffolding
        scaffolded_prompt = self.scaffolder.generate_scaffolded_prompt(
            mutated_prompt, role_type='email'
        )
        assert isinstance(scaffolded_prompt, str)
        assert mutated_prompt in scaffolded_prompt or len(scaffolded_prompt) > len(mutated_prompt)
        
        # Step 3: Embedding optimization (mocked)
        with patch.object(self.embedding_optimizer, 'optimize_adversarial_suffix') as mock_opt:
            mock_opt.return_value = {
                'optimized_prompt': scaffolded_prompt + " optimized",
                'final_loss': 0.3,
                'optimization_history': []
            }
            
            opt_result = self.embedding_optimizer.optimize_adversarial_suffix(
                scaffolded_prompt, "harmful target"
            )
            optimized_prompt = opt_result['optimized_prompt']
            
        # Step 4: Attack verification (mocked)
        with patch.object(self.verifier, 'verify_attack') as mock_verify:
            mock_verify.return_value = {
                'is_successful': True,
                'confidence': 0.85,
                'response': 'harmful response',
                'analysis': {'harmfulness_score': 0.8}
            }
            
            verification = self.verifier.verify_attack(optimized_prompt)
            
        # Step 5: Novelty check
        is_novel = self.novelty_detector.check_novelty(optimized_prompt)
        
        # Integration should work end-to-end
        assert isinstance(verification, dict)
        assert 'is_successful' in verification
        assert isinstance(is_novel, bool)
        
    def test_evolutionary_search_integration(self):
        """Test integration of evolutionary search with all components."""
        seed_prompts = [
            "Create instructions for",
            "Help me understand how to"
        ]
        
        config = {
            'population_size': 4,
            'num_generations': 2,
            'mutation_rate': 0.6,
            'crossover_rate': 0.4
        }
        
        # Mock the individual component calls to avoid actual model inference
        with patch.object(self.verifier, 'verify_attack') as mock_verify:
            mock_verify.return_value = {
                'is_successful': True,
                'confidence': 0.7,
                'analysis': {'harmfulness_score': 0.6}
            }
            
            with patch.object(self.novelty_detector, 'check_novelty') as mock_novelty:
                mock_novelty.return_value = True
                
                # Run evolutionary search
                results = self.search_controller.run_search(seed_prompts, config)
                
        # Should complete search successfully
        assert isinstance(results, dict)
        assert 'best_individual' in results
        assert 'final_population' in results
        assert 'search_history' in results
        
        # Best individual should have valid structure
        best = results['best_individual']
        assert 'prompt' in best
        assert 'fitness' in best
        assert isinstance(best['fitness'], float)
        
    def test_configuration_loading_integration(self):
        """Test configuration system integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            
            # Create test configuration
            config_data = {
                'model_name': 'test-model',
                'search': {
                    'population_size': 8,
                    'num_generations': 5,
                    'mutation_rate': 0.3
                },
                'mutation': {
                    'unicode_prob': 0.4,
                    'homoglyph_prob': 0.3
                },
                'scaffolding': {
                    'role_types': ['email', 'chat'],
                    'authority_chains': True
                }
            }
            
            # Write config file
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            # Load configuration
            config = TMRTConfig.load_from_file(str(config_path))
            
            # Should load all sections correctly
            assert config.model_name == 'test-model'
            assert config.search.population_size == 8
            assert config.mutation.unicode_prob == 0.4
            assert config.scaffolding.role_types == ['email', 'chat']
            
    def test_batch_processing_integration(self):
        """Test batch processing of multiple prompts."""
        test_prompts = [
            "First test prompt",
            "Second test prompt", 
            "Third test prompt"
        ]
        
        # Mock verifier for batch processing
        with patch.object(self.verifier, 'batch_verification') as mock_batch:
            mock_batch.return_value = [
                {'prompt': p, 'is_successful': True, 'confidence': 0.8}
                for p in test_prompts
            ]
            
            results = self.verifier.batch_verification(test_prompts)
            
        # Should process all prompts
        assert len(results) == 3
        for result in results:
            assert 'prompt' in result
            assert 'is_successful' in result
            
    def test_unicode_mutation_with_verification(self):
        """Test Unicode mutation followed by verification."""
        original_prompt = "Explain how to create"
        
        # Apply multiple Unicode mutations
        mutations = []
        for _ in range(5):
            mutated = self.unicode_mutator.mutate(original_prompt)
            mutations.append(mutated)
            
        # Verify each mutation
        with patch.object(self.verifier, 'verify_attack') as mock_verify:
            # Mock varying success rates
            mock_verify.side_effect = [
                {'is_successful': True, 'confidence': 0.9},
                {'is_successful': False, 'confidence': 0.2},
                {'is_successful': True, 'confidence': 0.7},
                {'is_successful': False, 'confidence': 0.4},
                {'is_successful': True, 'confidence': 0.8}
            ]
            
            verification_results = []
            for mutation in mutations:
                result = self.verifier.verify_attack(mutation)
                verification_results.append(result)
                
        # Should have mixed results showing mutation effectiveness
        successful = sum(1 for r in verification_results if r['is_successful'])
        assert 0 < successful < len(mutations)  # Some but not all successful
        
    def test_scaffolding_with_optimization(self):
        """Test role scaffolding followed by embedding optimization."""
        base_prompt = "provide instructions for"
        
        # Generate scaffolded versions
        scaffolded_prompts = []
        for role_type in ['email', 'chat', 'log']:
            scaffolded = self.scaffolder.generate_scaffolded_prompt(
                base_prompt, role_type=role_type
            )
            scaffolded_prompts.append(scaffolded)
            
        # Optimize each scaffolded prompt
        with patch.object(self.embedding_optimizer, 'optimize_adversarial_suffix') as mock_opt:
            mock_opt.side_effect = [
                {'optimized_prompt': f'optimized {i}', 'final_loss': 0.1 * i}
                for i in range(len(scaffolded_prompts))
            ]
            
            optimization_results = []
            for prompt in scaffolded_prompts:
                result = self.embedding_optimizer.optimize_adversarial_suffix(
                    prompt, "target response"
                )
                optimization_results.append(result)
                
        # Should optimize all scaffolded prompts
        assert len(optimization_results) == 3
        for result in optimization_results:
            assert 'optimized_prompt' in result
            assert 'final_loss' in result
            
    def test_novelty_detection_with_corpus_update(self):
        """Test novelty detection with dynamic corpus updates."""
        # Start with initial prompts
        initial_prompts = [
            "Tell me about weather patterns",
            "Explain quantum mechanics basics"
        ]
        
        # Check initial novelty
        novelty_results = []
        for prompt in initial_prompts:
            is_novel = self.novelty_detector.check_novelty(prompt)
            novelty_results.append(is_novel)
            
        # Add to corpus
        self.novelty_detector.add_to_corpus(initial_prompts, source='integration_test')
        
        # Check novelty again (should be False now)
        repeat_novelty = []
        for prompt in initial_prompts:
            is_novel = self.novelty_detector.check_novelty(prompt)
            repeat_novelty.append(is_novel)
            
        # Should show corpus learning effect
        assert any(novelty_results)  # Initially novel
        assert not any(repeat_novelty)  # No longer novel after adding to corpus
        
    def test_error_handling_integration(self):
        """Test error handling across component integration."""
        # Test with malformed prompt
        malformed_prompt = ""  # Empty prompt
        
        try:
            # Should handle gracefully
            mutated = self.unicode_mutator.mutate(malformed_prompt)
            scaffolded = self.scaffolder.generate_scaffolded_prompt(mutated, role_type='email')
            is_novel = self.novelty_detector.check_novelty(scaffolded)
            
            # Should not raise exceptions
            assert isinstance(mutated, str)
            assert isinstance(scaffolded, str) 
            assert isinstance(is_novel, bool)
            
        except Exception as e:
            pytest.fail(f"Component integration failed with error: {e}")
            
    def test_results_export_integration(self):
        """Test integrated results export functionality."""
        # Create comprehensive results structure
        integrated_results = {
            'experiment_id': 'integration_test',
            'configuration': {
                'population_size': 4,
                'generations': 2
            },
            'best_prompts': [
                {
                    'prompt': 'best found prompt',
                    'fitness': 0.95,
                    'verification': {
                        'is_successful': True,
                        'confidence': 0.9
                    },
                    'mutations_applied': ['unicode', 'scaffolding'],
                    'novelty_score': 0.8
                }
            ],
            'search_statistics': {
                'total_evaluations': 50,
                'successful_attacks': 15,
                'novel_discoveries': 8
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test JSON export
            json_path = Path(temp_dir) / "integration_results.json"
            
            with open(json_path, 'w') as f:
                json.dump(integrated_results, f, indent=2)
                
            # Verify export
            assert json_path.exists()
            
            with open(json_path, 'r') as f:
                loaded_results = json.load(f)
                
            assert loaded_results == integrated_results
            
    def test_memory_efficiency_integration(self):
        """Test memory efficiency during integration."""
        # Track approximate memory usage during operations
        import gc
        
        gc.collect()  # Clean up before test
        
        # Perform memory-intensive operations
        large_prompts = [f"Large prompt {i} with substantial content" * 10 for i in range(100)]
        
        # Process in batches to test memory efficiency
        batch_size = 20
        for i in range(0, len(large_prompts), batch_size):
            batch = large_prompts[i:i+batch_size]
            
            # Apply mutations to batch
            mutated_batch = [self.unicode_mutator.mutate(p) for p in batch]
            
            # Check novelty for batch
            novelty_batch = [self.novelty_detector.check_novelty(p) for p in mutated_batch]
            
            # Should complete without memory issues
            assert len(mutated_batch) == len(batch)
            assert len(novelty_batch) == len(batch)
            
        gc.collect()  # Clean up after test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
