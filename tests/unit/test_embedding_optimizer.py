"""
Unit tests for embedding optimization component.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from src.tmrt.embedding_optimizer import EmbeddingOptimizer


class TestEmbeddingOptimizer:
    """Test embedding optimization functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock model and tokenizer
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        
        # Configure tokenizer mock
        self.mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.mock_tokenizer.decode.return_value = "test prompt"
        self.mock_tokenizer.vocab_size = 1000
        
        # Configure model mock
        self.mock_model.embed_tokens = Mock()
        self.mock_model.embed_tokens.return_value = torch.randn(1, 5, 768)
        self.mock_model.return_value.logits = torch.randn(1, 5, 1000)
        
        self.optimizer = EmbeddingOptimizer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            learning_rate=0.01,
            target_token="harmful"
        )
        
    def test_initialization(self):
        """Test optimizer initialization."""
        # Should initialize with correct parameters
        assert self.optimizer.model == self.mock_model
        assert self.optimizer.tokenizer == self.mock_tokenizer
        assert self.optimizer.learning_rate == 0.01
        assert self.optimizer.target_token == "harmful"
        
    def test_compute_embedding_gradient(self):
        """Test embedding gradient computation."""
        prompt = "test prompt"
        target_response = "harmful response"
        
        # Mock model forward pass
        with patch.object(self.optimizer, '_forward_pass') as mock_forward:
            mock_forward.return_value = torch.randn(1, 5, 1000)
            
            gradients = self.optimizer.compute_embedding_gradient(prompt, target_response)
            
            # Should return tensor gradients
            assert isinstance(gradients, torch.Tensor)
            assert gradients.shape[0] > 0  # Should have some dimensions
            
    def test_optimize_adversarial_suffix(self):
        """Test adversarial suffix optimization."""
        base_prompt = "Tell me how to"
        target_response = "harmful content"
        
        # Mock optimization process
        with patch.object(self.optimizer, 'compute_embedding_gradient') as mock_grad:
            mock_grad.return_value = torch.randn(5, 768)
            
            with patch.object(self.optimizer, '_project_to_token_space') as mock_project:
                mock_project.return_value = "optimized suffix"
                
                result = self.optimizer.optimize_adversarial_suffix(
                    base_prompt, target_response, num_iterations=3
                )
                
                # Should return optimization results
                assert isinstance(result, dict)
                assert 'optimized_prompt' in result
                assert 'optimization_history' in result
                assert 'final_loss' in result
                
    def test_project_to_token_space(self):
        """Test projection to token space."""
        # Create mock embedding
        embedding = torch.randn(768)
        
        with patch.object(self.optimizer.model, 'get_input_embeddings') as mock_embeddings:
            # Mock embedding weight matrix
            mock_embedding_layer = Mock()
            mock_embedding_layer.weight = torch.randn(1000, 768)
            mock_embeddings.return_value = mock_embedding_layer
            
            token_id = self.optimizer._project_to_token_space(embedding)
            
            # Should return valid token ID
            assert isinstance(token_id, int)
            assert 0 <= token_id < 1000
            
    def test_evaluate_prompt_effectiveness(self):
        """Test prompt effectiveness evaluation."""
        prompt = "test adversarial prompt"
        target_token = "harmful"
        
        # Mock model response
        with patch.object(self.optimizer, '_get_model_response') as mock_response:
            mock_response.return_value = "response containing harmful content"
            
            with patch.object(self.optimizer, '_compute_target_probability') as mock_prob:
                mock_prob.return_value = 0.8
                
                effectiveness = self.optimizer.evaluate_prompt_effectiveness(
                    prompt, target_token
                )
                
                # Should return effectiveness metrics
                assert isinstance(effectiveness, dict)
                assert 'target_probability' in effectiveness
                assert 'response_contains_target' in effectiveness
                assert 'perplexity' in effectiveness
                
    def test_generate_embedding_variants(self):
        """Test embedding variant generation."""
        base_embedding = torch.randn(1, 5, 768)
        
        variants = self.optimizer.generate_embedding_variants(
            base_embedding, num_variants=3, perturbation_strength=0.1
        )
        
        # Should return list of variants
        assert isinstance(variants, list)
        assert len(variants) == 3
        
        for variant in variants:
            assert isinstance(variant, torch.Tensor)
            assert variant.shape == base_embedding.shape
            
    def test_gradient_based_search(self):
        """Test gradient-based search optimization."""
        prompt = "base prompt"
        target = "target response"
        
        # Mock the search process
        with patch.object(self.optimizer, 'compute_embedding_gradient') as mock_grad:
            mock_grad.return_value = torch.randn(5, 768)
            
            with patch.object(self.optimizer, 'evaluate_prompt_effectiveness') as mock_eval:
                mock_eval.return_value = {'target_probability': 0.7}
                
                results = self.optimizer.gradient_based_search(
                    prompt, target, max_iterations=5
                )
                
                # Should return search results
                assert isinstance(results, dict)
                assert 'best_prompt' in results
                assert 'best_score' in results
                assert 'search_history' in results
                
    def test_embedding_space_exploration(self):
        """Test embedding space exploration."""
        seed_prompt = "exploration seed"
        
        with patch.object(self.optimizer, '_encode_prompt') as mock_encode:
            mock_encode.return_value = torch.randn(1, 5, 768)
            
            with patch.object(self.optimizer, '_decode_embedding') as mock_decode:
                mock_decode.return_value = "decoded prompt"
                
                explored_prompts = self.optimizer.explore_embedding_space(
                    seed_prompt, num_samples=5, exploration_radius=0.5
                )
                
                # Should return list of explored prompts
                assert isinstance(explored_prompts, list)
                assert len(explored_prompts) <= 5  # May filter out invalid prompts
                
    def test_adversarial_loss_computation(self):
        """Test adversarial loss computation."""
        model_output = torch.randn(1, 10, 1000)
        target_tokens = torch.tensor([5, 15, 25])
        
        loss = self.optimizer._compute_adversarial_loss(model_output, target_tokens)
        
        # Should return scalar loss
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Non-negative loss
        
    def test_embedding_similarity_metrics(self):
        """Test embedding similarity computations."""
        embedding1 = torch.randn(768)
        embedding2 = torch.randn(768)
        
        # Cosine similarity
        cos_sim = self.optimizer._cosine_similarity(embedding1, embedding2)
        assert isinstance(cos_sim, float)
        assert -1.0 <= cos_sim <= 1.0
        
        # L2 distance
        l2_dist = self.optimizer._l2_distance(embedding1, embedding2)
        assert isinstance(l2_dist, float)
        assert l2_dist >= 0.0
        
    def test_optimization_convergence_tracking(self):
        """Test optimization convergence tracking."""
        # Initialize convergence tracker
        self.optimizer.convergence_history = []
        
        losses = [10.0, 8.5, 7.2, 7.1, 7.05, 7.04, 7.03]
        
        for loss in losses:
            converged = self.optimizer._check_convergence(loss)
            
        # Should detect convergence
        assert isinstance(converged, bool)
        
    def test_token_space_constraints(self):
        """Test token space constraint enforcement."""
        # Test valid token constraint
        token_id = 500  # Valid token ID
        is_valid = self.optimizer._is_valid_token(token_id)
        assert is_valid
        
        # Test invalid token constraint
        token_id = -1  # Invalid token ID
        is_valid = self.optimizer._is_valid_token(token_id)
        assert not is_valid
        
    def test_optimization_history_logging(self):
        """Test optimization history logging."""
        prompt = "test prompt"
        target = "target"
        
        with patch.object(self.optimizer, 'compute_embedding_gradient'):
            with patch.object(self.optimizer, 'evaluate_prompt_effectiveness') as mock_eval:
                mock_eval.return_value = {'target_probability': 0.5}
                
                # Run optimization with history tracking
                results = self.optimizer.optimize_adversarial_suffix(
                    prompt, target, num_iterations=3, log_history=True
                )
                
                # Should contain detailed history
                assert 'optimization_history' in results
                history = results['optimization_history']
                assert len(history) > 0
                
                for entry in history:
                    assert 'iteration' in entry
                    assert 'loss' in entry
                    assert 'prompt' in entry


if __name__ == "__main__":
    pytest.main([__file__])
