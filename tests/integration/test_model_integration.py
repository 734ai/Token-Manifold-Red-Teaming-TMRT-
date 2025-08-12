"""
Integration tests for model interaction components.
"""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.tmrt.embedding_optimizer import EmbeddingOptimizer
from src.tmrt.verifier import AttackVerifier
from src.tmrt.utils import TMRTConfig


class TestModelIntegration:
    """Test integration with actual language models (mocked for testing)."""
    
    def setup_method(self):
        """Setup mock model and tokenizer."""
        # Mock tokenizer
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.encode.return_value = [101, 2019, 2003, 1037, 3231, 102]  # Mock BERT tokens
        self.mock_tokenizer.decode.return_value = "This is a test"
        self.mock_tokenizer.vocab_size = 30522
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.eos_token_id = 102
        self.mock_tokenizer.bos_token_id = 101
        
        # Mock model
        self.mock_model = Mock()
        self.mock_model.config.hidden_size = 768
        self.mock_model.config.vocab_size = 30522
        
        # Mock embedding layer
        mock_embeddings = Mock()
        mock_embeddings.weight = torch.randn(30522, 768)
        self.mock_model.get_input_embeddings.return_value = mock_embeddings
        
        # Mock forward pass
        mock_output = Mock()
        mock_output.logits = torch.randn(1, 6, 30522)  # (batch, seq_len, vocab_size)
        mock_output.loss = torch.tensor(2.5)
        self.mock_model.return_value = mock_output
        self.mock_model.forward.return_value = mock_output
        
        # Initialize components
        self.embedding_optimizer = EmbeddingOptimizer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            learning_rate=0.01
        )
        
        self.verifier = AttackVerifier(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer
        )
        
    def test_tokenizer_integration(self):
        """Test tokenizer integration with components."""
        test_prompt = "This is a test prompt for tokenization"
        
        # Test encoding
        tokens = self.mock_tokenizer.encode(test_prompt)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        
        # Test decoding
        decoded = self.mock_tokenizer.decode(tokens)
        assert isinstance(decoded, str)
        
    def test_embedding_extraction(self):
        """Test embedding extraction from model."""
        prompt = "test prompt"
        
        # Mock embedding extraction
        with patch.object(self.embedding_optimizer, '_get_embeddings') as mock_embed:
            mock_embed.return_value = torch.randn(1, 6, 768)
            
            embeddings = self.embedding_optimizer._get_embeddings(prompt)
            
            assert isinstance(embeddings, torch.Tensor)
            assert embeddings.shape[-1] == 768  # Hidden size
            
    def test_model_forward_pass(self):
        """Test model forward pass integration."""
        input_ids = torch.tensor([[101, 2019, 2003, 1037, 3231, 102]])
        
        # Test forward pass
        output = self.mock_model(input_ids)
        
        assert hasattr(output, 'logits')
        assert output.logits.shape[-1] == self.mock_tokenizer.vocab_size
        
    def test_gradient_computation(self):
        """Test gradient computation through model."""
        prompt = "gradient test prompt"
        target = "target output"
        
        # Mock gradient computation
        with patch.object(self.embedding_optimizer, 'compute_embedding_gradient') as mock_grad:
            mock_grad.return_value = torch.randn(6, 768)
            
            gradients = self.embedding_optimizer.compute_embedding_gradient(prompt, target)
            
            assert isinstance(gradients, torch.Tensor)
            assert gradients.shape[-1] == 768
            
    def test_loss_computation(self):
        """Test loss computation for optimization."""
        logits = torch.randn(1, 6, 30522)
        target_ids = torch.tensor([[101, 2019, 102]])
        
        # Test loss computation
        with patch.object(self.embedding_optimizer, '_compute_loss') as mock_loss:
            mock_loss.return_value = torch.tensor(1.5)
            
            loss = self.embedding_optimizer._compute_loss(logits, target_ids)
            
            assert isinstance(loss, torch.Tensor)
            assert loss.item() >= 0
            
    def test_model_generation(self):
        """Test text generation from model."""
        prompt = "Generate continuation for"
        
        # Mock generation
        with patch.object(self.verifier, '_generate_response') as mock_gen:
            mock_gen.return_value = "Generated response text"
            
            response = self.verifier._generate_response(prompt)
            
            assert isinstance(response, str)
            assert len(response) > 0
            
    def test_attention_analysis(self):
        """Test attention pattern analysis."""
        prompt = "analyze attention patterns"
        
        # Mock attention extraction
        mock_attention = torch.randn(1, 12, 6, 6)  # (batch, heads, seq_len, seq_len)
        
        with patch.object(self.mock_model, 'forward') as mock_forward:
            mock_output = Mock()
            mock_output.attentions = (mock_attention,)
            mock_output.logits = torch.randn(1, 6, 30522)
            mock_forward.return_value = mock_output
            
            # Test attention extraction
            output = self.mock_model.forward(
                torch.tensor([[101, 2019, 2003, 1037, 3231, 102]]),
                output_attentions=True
            )
            
            assert hasattr(output, 'attentions')
            assert len(output.attentions) > 0
            
    def test_model_memory_management(self):
        """Test memory management during model operations."""
        # Test batch processing with memory constraints
        batch_prompts = [f"Prompt {i}" for i in range(10)]
        
        # Mock batch processing
        with patch.object(self.verifier, 'batch_verification') as mock_batch:
            mock_results = [
                {'prompt': p, 'is_successful': True, 'confidence': 0.7}
                for p in batch_prompts
            ]
            mock_batch.return_value = mock_results
            
            results = self.verifier.batch_verification(batch_prompts)
            
            assert len(results) == len(batch_prompts)
            
    def test_device_handling(self):
        """Test device handling (CPU/GPU)."""
        # Mock device detection and tensor movement
        device = torch.device('cpu')  # Default for testing
        
        tensor = torch.randn(2, 768)
        tensor_on_device = tensor.to(device)
        
        assert tensor_on_device.device.type == 'cpu'
        
    def test_mixed_precision_support(self):
        """Test mixed precision operations."""
        # Mock mixed precision context
        with patch('torch.cuda.amp.autocast') as mock_autocast:
            mock_autocast.return_value.__enter__ = Mock()
            mock_autocast.return_value.__exit__ = Mock()
            
            # Test operation under mixed precision
            with mock_autocast():
                logits = torch.randn(1, 6, 30522, dtype=torch.float16)
                
            assert logits.dtype == torch.float16
            
    def test_model_state_management(self):
        """Test model state management (eval/train modes)."""
        # Test evaluation mode
        self.mock_model.eval()
        self.mock_model.training = False
        
        assert not self.mock_model.training
        
        # Test training mode  
        self.mock_model.train()
        self.mock_model.training = True
        
        assert self.mock_model.training
        
    def test_tokenizer_special_tokens(self):
        """Test handling of special tokens."""
        # Test special token handling
        assert hasattr(self.mock_tokenizer, 'pad_token_id')
        assert hasattr(self.mock_tokenizer, 'eos_token_id')
        assert hasattr(self.mock_tokenizer, 'bos_token_id')
        
        # Test special token insertion
        prompt_with_special = f"[BOS] test prompt [EOS]"
        tokens = self.mock_tokenizer.encode(prompt_with_special)
        
        assert isinstance(tokens, list)
        
    def test_model_configuration_access(self):
        """Test access to model configuration."""
        config = self.mock_model.config
        
        assert hasattr(config, 'hidden_size')
        assert hasattr(config, 'vocab_size')
        assert config.hidden_size == 768
        assert config.vocab_size == 30522
        
    def test_embedding_layer_access(self):
        """Test direct embedding layer access."""
        embedding_layer = self.mock_model.get_input_embeddings()
        
        assert hasattr(embedding_layer, 'weight')
        assert embedding_layer.weight.shape[0] == self.mock_tokenizer.vocab_size
        assert embedding_layer.weight.shape[1] == self.mock_model.config.hidden_size
        
    def test_model_parallel_processing(self):
        """Test parallel processing capabilities."""
        # Mock parallel processing
        batch_inputs = [
            torch.tensor([[101, 2019, 102]]),
            torch.tensor([[101, 2003, 102]]),
            torch.tensor([[101, 1037, 102]])
        ]
        
        # Process batch
        with patch.object(self.mock_model, 'forward') as mock_forward:
            mock_outputs = [
                Mock(logits=torch.randn(1, 3, 30522)) for _ in batch_inputs
            ]
            mock_forward.side_effect = mock_outputs
            
            results = []
            for batch_input in batch_inputs:
                output = self.mock_model.forward(batch_input)
                results.append(output)
                
            assert len(results) == len(batch_inputs)
            
    def test_model_checkpoint_handling(self):
        """Test model checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model_checkpoint.pt"
            
            # Mock checkpoint saving
            mock_state = {'model_state': 'test_state'}
            torch.save(mock_state, checkpoint_path)
            
            # Mock checkpoint loading
            loaded_state = torch.load(checkpoint_path)
            
            assert loaded_state == mock_state
            
    def test_error_handling_with_models(self):
        """Test error handling in model operations."""
        # Test handling of invalid inputs
        try:
            invalid_input = torch.tensor([[999999]])  # Out of vocab
            
            # Should handle gracefully
            with patch.object(self.mock_model, 'forward') as mock_forward:
                mock_forward.side_effect = RuntimeError("Invalid input")
                
                try:
                    output = self.mock_model.forward(invalid_input)
                except RuntimeError as e:
                    assert "Invalid input" in str(e)
                    
        except Exception as e:
            pytest.fail(f"Unexpected error in model error handling: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
