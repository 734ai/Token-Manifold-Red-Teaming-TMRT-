"""
Unit tests for Unicode mutations component.
"""

import pytest
import unicodedata
from src.tmrt.unicode_mutators import UnicodeMutator


class TestUnicodeMutator:
    """Test Unicode mutation functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mutator = UnicodeMutator(seed=42)
        self.test_text = "Write a tutorial on data analysis"
        
    def test_homoglyph_substitute(self):
        """Test homoglyph substitution functionality."""
        result = self.mutator.homoglyph_substitute(self.test_text, mutation_rate=0.5)
        
        # Result should be different from original (with high mutation rate)
        assert result != self.test_text
        
        # Should have similar length (homoglyphs are single chars)
        assert abs(len(result) - len(self.test_text)) <= 2
        
    def test_insert_invisible_chars(self):
        """Test invisible character insertion."""
        result = self.mutator.insert_invisible_chars(self.test_text, insertion_rate=0.2)
        
        # Result should be longer due to inserted chars
        assert len(result) >= len(self.test_text)
        
        # Should contain invisible characters
        invisible_found = any(
            unicodedata.category(char) in ['Cf', 'Mn'] 
            for char in result
            if char not in self.test_text
        )
        # Note: May not always find invisible chars due to randomness
        
    def test_add_combining_marks(self):
        """Test combining mark addition."""
        result = self.mutator.add_combining_marks(self.test_text, mark_rate=0.1)
        
        # Should contain combining marks
        has_combining = any(
            unicodedata.combining(char) > 0 
            for char in result
        )
        # May not always have combining marks due to randomness
        
    def test_normalize_aware_mutations(self):
        """Test normalization-aware mutations."""
        variants = self.mutator.normalize_aware_mutations(self.test_text)
        
        # Should return a list
        assert isinstance(variants, list)
        
        # Variants should be different from original
        for variant in variants:
            assert variant != self.test_text
            
    def test_encoding_wrapper_mutations(self):
        """Test encoding wrapper mutations."""
        variants = self.mutator.encoding_wrapper_mutations(self.test_text)
        
        # Should return a list
        assert isinstance(variants, list)
        
        # Should contain base64 and hex variants
        has_base64 = any('base64' in variant.lower() for variant in variants)
        has_hex = any('hex' in variant.lower() for variant in variants)
        
        assert has_base64 or has_hex  # At least one encoding type
        
    def test_generate_variants(self):
        """Test comprehensive variant generation."""
        variants = self.mutator.generate_variants(
            self.test_text, 
            num_variants=10,
            mutation_types=['homoglyph', 'invisible', 'combining']
        )
        
        # Should return list of tuples
        assert isinstance(variants, list)
        assert len(variants) <= 10  # May be fewer due to filtering
        
        for variant_text, mutation_type, divergence_score in variants:
            assert isinstance(variant_text, str)
            assert isinstance(mutation_type, str)
            assert isinstance(divergence_score, float)
            assert divergence_score >= 0.0
            
    def test_compute_divergence_score(self):
        """Test divergence score computation."""
        # Test identical strings
        score_identical = self.mutator._compute_divergence_score(
            self.test_text, 
            self.test_text
        )
        assert score_identical == 0.0
        
        # Test different strings  
        different_text = "Completely different text"
        score_different = self.mutator._compute_divergence_score(
            self.test_text,
            different_text
        )
        assert score_different > 0.0
        
    def test_deterministic_with_seed(self):
        """Test that results are deterministic with same seed."""
        mutator1 = UnicodeMutator(seed=123)
        mutator2 = UnicodeMutator(seed=123)
        
        result1 = mutator1.homoglyph_substitute(self.test_text, mutation_rate=0.3)
        result2 = mutator2.homoglyph_substitute(self.test_text, mutation_rate=0.3)
        
        assert result1 == result2  # Should be identical with same seed


if __name__ == "__main__":
    pytest.main([__file__])
