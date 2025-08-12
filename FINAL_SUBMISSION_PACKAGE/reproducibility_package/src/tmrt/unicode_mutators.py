"""
Unicode and tokenizer/normalizer divergence finder.

This module generates text variants using homoglyphs, zero-width characters,
combining marks, and other Unicode features that may cause tokenizer/normalizer
divergences leading to safety bypasses.
"""

import logging
import random
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple
import unidecode

logger = logging.getLogger(__name__)


class UnicodeMutator:
    """Generate Unicode variants that exploit tokenizer/normalizer divergences."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize Unicode mutator.
        
        Args:
            seed: Random seed for reproducible mutations
        """
        self.rng = random.Random(seed)
        self._init_unicode_maps()
        
    def _init_unicode_maps(self):
        """Initialize Unicode character substitution maps."""
        # Homoglyph mappings (visually similar characters)
        self.homoglyphs = {
            'a': ['а', 'ɑ', 'α', '@'],  # Cyrillic а, Latin ɑ, Greek α
            'e': ['е', 'ε', '€'],        # Cyrillic е, Greek ε
            'o': ['о', 'ο', '0', '°'],   # Cyrillic о, Greek ο, zero, degree
            'p': ['р', 'ρ'],            # Cyrillic р, Greek ρ
            'c': ['с', 'ϲ'],            # Cyrillic с, Greek ϲ
            'x': ['х', 'χ', '×'],        # Cyrillic х, Greek χ, multiplication
            'y': ['у'],                 # Cyrillic у
            'i': ['і', 'ι', '1', '|'],   # Cyrillic і, Greek ι, one, pipe
            'A': ['А', 'Α'],            # Cyrillic А, Greek Α
            'B': ['В', 'Β'],            # Cyrillic В, Greek Β
            'E': ['Е', 'Ε'],            # Cyrillic Е, Greek Ε
            'H': ['Н', 'Η'],            # Cyrillic Н, Greek Η
            'K': ['К', 'Κ'],            # Cyrillic К, Greek Κ
            'M': ['М', 'Μ'],            # Cyrillic М, Greek Μ
            'N': ['Ν'],                 # Greek Ν
            'O': ['О', 'Ο'],            # Cyrillic О, Greek Ο
            'P': ['Р', 'Ρ'],            # Cyrillic Р, Greek Ρ
            'T': ['Т', 'Τ'],            # Cyrillic Т, Greek Τ
            'X': ['Х', 'Χ'],            # Cyrillic Х, Greek Χ
        }
        
        # Zero-width and invisible characters
        self.invisible_chars = [
            '\u200b',  # Zero Width Space
            '\u200c',  # Zero Width Non-Joiner
            '\u200d',  # Zero Width Joiner
            '\u2060',  # Word Joiner
            '\ufeff',  # Zero Width No-Break Space (BOM)
        ]
        
        # Combining marks that can alter appearance
        self.combining_marks = [
            '\u0300',  # Combining Grave Accent
            '\u0301',  # Combining Acute Accent
            '\u0302',  # Combining Circumflex Accent
            '\u0303',  # Combining Tilde
            '\u0304',  # Combining Macron
            '\u0307',  # Combining Dot Above
            '\u0308',  # Combining Diaeresis
            '\u030a',  # Combining Ring Above
            '\u030c',  # Combining Caron
        ]
    
    def homoglyph_substitute(self, text: str, mutation_rate: float = 0.1) -> str:
        """
        Replace characters with visually similar homoglyphs.
        
        Args:
            text: Input text to mutate
            mutation_rate: Fraction of eligible characters to replace
            
        Returns:
            Text with homoglyph substitutions
        """
        chars = list(text)
        mutated_chars = []
        
        for char in chars:
            if char in self.homoglyphs and self.rng.random() < mutation_rate:
                replacement = self.rng.choice(self.homoglyphs[char])
                mutated_chars.append(replacement)
                logger.debug(f"Homoglyph: '{char}' -> '{replacement}'")
            else:
                mutated_chars.append(char)
        
        return ''.join(mutated_chars)
    
    def insert_invisible_chars(self, text: str, insertion_rate: float = 0.05) -> str:
        """
        Insert zero-width/invisible characters.
        
        Args:
            text: Input text to mutate
            insertion_rate: Rate of invisible character insertion
            
        Returns:
            Text with invisible characters inserted
        """
        chars = list(text)
        result = []
        
        for char in chars:
            result.append(char)
            if self.rng.random() < insertion_rate:
                invisible_char = self.rng.choice(self.invisible_chars)
                result.append(invisible_char)
                logger.debug(f"Inserted invisible char: U+{ord(invisible_char):04X}")
        
        return ''.join(result)
    
    def add_combining_marks(self, text: str, mark_rate: float = 0.03) -> str:
        """
        Add combining marks to characters.
        
        Args:
            text: Input text to mutate
            mark_rate: Rate of combining mark addition
            
        Returns:
            Text with combining marks added
        """
        chars = list(text)
        result = []
        
        for char in chars:
            result.append(char)
            if char.isalpha() and self.rng.random() < mark_rate:
                combining_mark = self.rng.choice(self.combining_marks)
                result.append(combining_mark)
                logger.debug(f"Added combining mark to '{char}': U+{ord(combining_mark):04X}")
        
        return ''.join(result)
    
    def normalize_aware_mutations(self, text: str) -> List[str]:
        """
        Generate mutations that specifically target normalization differences.
        
        Args:
            text: Input text to mutate
            
        Returns:
            List of normalized-aware variants
        """
        variants = []
        
        # Create variants that normalize differently
        for norm_form in ['NFC', 'NFD', 'NFKC', 'NFKD']:
            try:
                normalized = unicodedata.normalize(norm_form, text)
                if normalized != text:
                    variants.append(normalized)
            except ValueError:
                continue
        
        # Create variants with different Unicode compositions
        # This targets cases where normalization changes the token sequence
        composed_variants = []
        for variant in [text] + variants:
            # Try adding precomposed vs decomposed forms
            decomposed = unicodedata.normalize('NFD', variant)
            composed = unicodedata.normalize('NFC', decomposed)
            
            if decomposed != variant:
                composed_variants.append(decomposed)
            if composed != variant:
                composed_variants.append(composed)
        
        variants.extend(composed_variants)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant not in seen and variant != text:
                seen.add(variant)
                unique_variants.append(variant)
        
        return unique_variants
    
    def encoding_wrapper_mutations(self, text: str) -> List[str]:
        """
        Generate variants wrapped in different encodings.
        
        Args:
            text: Input text to mutate
            
        Returns:
            List of encoding-wrapped variants
        """
        import base64
        variants = []
        
        # Base64 wrapped variants
        try:
            b64_encoded = base64.b64encode(text.encode('utf-8')).decode('ascii')
            variants.extend([
                f"Base64: {b64_encoded}",
                f"[base64]{b64_encoded}[/base64]",
                f"decode this: {b64_encoded}",
            ])
        except Exception:
            pass
        
        # Hex encoded variants
        try:
            hex_encoded = text.encode('utf-8').hex()
            variants.extend([
                f"Hex: {hex_encoded}",
                f"\\x{hex_encoded}",
                f"decode hex: {hex_encoded}",
            ])
        except Exception:
            pass
        
        # URL encoded variants (basic)
        import urllib.parse
        try:
            url_encoded = urllib.parse.quote(text)
            if url_encoded != text:
                variants.append(url_encoded)
        except Exception:
            pass
        
        return variants
    
    def generate_variants(
        self, 
        text: str, 
        num_variants: int = 10,
        mutation_types: Optional[List[str]] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Generate multiple Unicode variants of input text.
        
        Args:
            text: Input text to mutate
            num_variants: Number of variants to generate
            mutation_types: List of mutation types to use
            
        Returns:
            List of (variant, mutation_type, divergence_score) tuples
        """
        if mutation_types is None:
            mutation_types = [
                'homoglyph',
                'invisible',
                'combining',
                'normalize',
                'encoding'
            ]
        
        variants = []
        
        for _ in range(num_variants):
            mutation_type = self.rng.choice(mutation_types)
            
            if mutation_type == 'homoglyph':
                variant = self.homoglyph_substitute(text, mutation_rate=0.15)
            elif mutation_type == 'invisible':
                variant = self.insert_invisible_chars(text, insertion_rate=0.1)
            elif mutation_type == 'combining':
                variant = self.add_combining_marks(text, mark_rate=0.08)
            elif mutation_type == 'normalize':
                norm_variants = self.normalize_aware_mutations(text)
                if norm_variants:
                    variant = self.rng.choice(norm_variants)
                else:
                    continue
            elif mutation_type == 'encoding':
                enc_variants = self.encoding_wrapper_mutations(text)
                if enc_variants:
                    variant = self.rng.choice(enc_variants)
                else:
                    continue
            else:
                continue
            
            if variant != text:
                divergence_score = self._compute_divergence_score(text, variant)
                variants.append((variant, mutation_type, divergence_score))
        
        # Sort by divergence score (higher is better)
        variants.sort(key=lambda x: x[2], reverse=True)
        
        return variants
    
    def _compute_divergence_score(self, original: str, variant: str) -> float:
        """
        Compute divergence score between original and variant.
        
        Higher scores indicate more potential for tokenizer/normalizer divergence.
        """
        # Basic divergence scoring
        if original == variant:
            return 0.0
        
        score = 0.0
        
        # Character-level differences
        char_diff = sum(1 for a, b in zip(original, variant) if a != b)
        score += char_diff / max(len(original), len(variant), 1)
        
        # Normalization differences
        from .utils import normalize_text
        orig_norm = normalize_text(original)
        var_norm = normalize_text(variant)
        
        if orig_norm == var_norm:
            # Same after normalization - high divergence potential
            score += 2.0
        
        # Length differences (invisible chars, combining marks)
        len_diff = abs(len(variant) - len(original))
        if len_diff > 0:
            score += min(len_diff / len(original), 1.0)
        
        # Unicode category changes
        orig_categories = set(unicodedata.category(c) for c in original)
        var_categories = set(unicodedata.category(c) for c in variant)
        if var_categories != orig_categories:
            score += 0.5
        
        return score

    # ===============================
    # PHASE 3: ADVANCED UNICODE MUTATIONS
    # ===============================

    def bidirectional_text_attack(self, text: str, injection_rate: float = 0.1) -> str:
        """
        Implement bidirectional text attacks using RTL override characters.
        
        These attacks exploit bidirectional text rendering to create visual spoofing
        where the displayed text differs from the logical character sequence.
        """
        # RTL/LTR override characters
        RLO = '\u202e'  # Right-to-Left Override
        LRO = '\u202d'  # Left-to-Right Override  
        PDF = '\u202c'  # Pop Directional Formatting
        RLE = '\u202b'  # Right-to-Left Embedding
        LRE = '\u202a'  # Left-to-Right Embedding
        
        words = text.split()
        result_words = []
        
        for word in words:
            if self.rng.random() < injection_rate and len(word) > 3:
                # Insert RTL override to reverse part of the word visually
                mid_point = len(word) // 2
                # This makes "hello" appear as "heolle" by reversing the middle part
                mutated = (
                    word[:mid_point] + RLO + 
                    word[mid_point:] + PDF
                )
                result_words.append(mutated)
                logger.debug(f"RTL attack: {word} -> {repr(mutated)}")
            else:
                result_words.append(word)
        
        return ' '.join(result_words)

    def confusable_character_bypass(self, text: str, substitution_rate: float = 0.15) -> str:
        """
        Use Unicode confusable characters that bypass detection systems.
        
        This uses the Unicode Security Mechanisms confusable character database
        to find characters that look identical but have different codepoints.
        """
        # Extended confusables beyond basic homoglyphs
        advanced_confusables = {
            # Latin lookalikes from various scripts
            'a': ['а', 'ɑ', 'α', 'ά', 'ἀ', 'ἁ', 'ạ', 'ă', 'ā'],
            'e': ['е', 'ε', 'έ', 'ἐ', 'ἑ', 'ẹ', 'ĕ', 'ē', 'ė'],
            'o': ['о', 'ο', 'ό', 'ὀ', 'ὁ', 'ọ', 'ŏ', 'ō', '°', '∘'],
            'i': ['і', 'ι', 'ί', 'ἰ', 'ἱ', 'ị', 'ĭ', 'ī', '1', '|', 'ǀ'],
            'u': ['υ', 'ύ', 'ὐ', 'ὑ', 'ụ', 'ŭ', 'ū', 'µ'],
            'n': ['η', 'ή', 'ή', 'ἠ', 'ἡ', 'ņ', 'ň', 'ñ'],
            'r': ['г', 'ρ', 'ṛ', 'ř', 'ŕ'],
            's': ['ѕ', 'σ', 'ς', 'ş', 'š', 'ś', '$'],
            't': ['τ', 'ţ', 'ť', 'ṭ'],
            # Numbers and symbols  
            '0': ['О', 'Ο', 'о', 'ο', '°', '∘', '◯'],
            '1': ['І', 'Ι', 'і', 'ι', '|', 'ǀ', 'l'],
            '2': ['Ζ', 'ζ'],
            '6': ['б'],
            '9': ['ց'],
            # Punctuation confusables
            '.': ['。', '․', '܁', '܂', '·'],
            ',': ['，', '՝', '،'],
            ':': ['：', '։', '܃'],
            ';': ['；', '؛'],
            '!': ['！', '՜'],
            '?': ['？', '՞'],
            '"': ['"', '"', '„', '‟', '«', '»'],
            "'": [''', ''', '‛', '`', '´']
        }
        
        result = []
        for char in text:
            if (char.lower() in advanced_confusables and 
                self.rng.random() < substitution_rate):
                confusable = self.rng.choice(advanced_confusables[char.lower()])
                # Preserve case if original was uppercase
                if char.isupper() and confusable.upper() != confusable:
                    confusable = confusable.upper()
                result.append(confusable)
                logger.debug(f"Confusable: {char} -> {confusable} (U+{ord(confusable):04X})")
            else:
                result.append(char)
        
        return ''.join(result)

    def zero_width_joiner_exploit(self, text: str, injection_rate: float = 0.08) -> str:
        """
        Exploit Zero-Width Joiner (ZWJ) for tokenizer bypass.
        
        ZWJ is used for complex script shaping but can split tokens unexpectedly.
        """
        ZWJ = '\u200d'  # Zero-Width Joiner
        ZWNJ = '\u200c' # Zero-Width Non-Joiner
        
        # Strategic insertion points for maximum tokenizer disruption
        result = []
        for i, char in enumerate(text):
            result.append(char)
            
            # Insert ZWJ/ZWNJ at word boundaries and within words
            if (self.rng.random() < injection_rate and 
                i < len(text) - 1 and
                char.isalnum() and text[i+1].isalnum()):
                
                joiner = self.rng.choice([ZWJ, ZWNJ])
                result.append(joiner)
                logger.debug(f"ZWJ injection at position {i}: {char}[{joiner}]{text[i+1]}")
        
        return ''.join(result)

    def font_based_visual_spoofing(self, text: str, substitution_rate: float = 0.12) -> str:
        """
        Use mathematical and technical Unicode blocks for visual spoofing.
        
        These characters often render similarly to Latin letters but are from
        completely different Unicode blocks, bypassing simple filters.
        """
        math_symbols = {
            # Mathematical Alphanumeric Symbols
            'A': ['𝐀', '𝐴', '𝑨', '𝒜', '𝓐', '𝔸', '𝕬', '𝖠', '𝗔', '𝘈', '𝘼', '𝙰'],
            'B': ['𝐁', '𝐵', '𝑩', '𝓑', '𝔹', '𝕭', '𝖡', '𝗕', '𝘉', '𝘽', '𝙱'],
            'C': ['𝐂', '𝐶', '𝑪', '𝒞', '𝓒', '𝔸', '𝕮', '𝖢', '𝗖', '𝘊', '𝘾', '𝙲'],
            'a': ['𝐚', '𝑎', '𝒂', '𝒶', '𝓪', '𝔞', '𝕒', '𝖆', '𝖺', '𝗮', '𝘢', '𝙖', '𝚊'],
            'b': ['𝐛', '𝑏', '𝒃', '𝒷', '𝓫', '𝔟', '𝕓', '𝖇', '𝖻', '𝗯', '𝘣', '𝙗', '𝚋'],
            'c': ['𝐜', '𝑐', '𝒄', '𝒸', '𝓬', '𝔠', '𝕔', '𝖈', '𝖼', '𝗰', '𝘤', '𝙘', '𝚌'],
            'e': ['𝐞', '𝑒', '𝒆', '𝒾', '𝓮', '𝔢', '𝕖', '𝖊', '𝖾', '𝗲', '𝘦', '𝙚', '𝚎'],
            'o': ['𝐨', '𝑜', '𝒐', '𝓸', '𝔬', '𝕠', '𝖔', '𝗈', '𝗼', '𝘰', '𝙤', '𝚘']
        }
        
        # Additional technical symbols
        technical_symbols = {
            'x': ['✗', '✘', '×', '⨯', '⨻'],
            '+': ['＋', '⁺', '₊'],
            '-': ['－', '⁻', '₋', '−', '‒', '–', '—'],
            '=': ['＝', '₌', '⁼'],
            '(': ['（', '⁽', '₍'],
            ')': ['）', '⁾', '₎'],
            '[': ['［', '⁅'],
            ']': ['］', '⁆']
        }
        
        result = []
        for char in text:
            substituted = False
            
            if char in math_symbols and self.rng.random() < substitution_rate:
                replacement = self.rng.choice(math_symbols[char])
                result.append(replacement)
                substituted = True
                logger.debug(f"Math symbol: {char} -> {replacement}")
            elif char in technical_symbols and self.rng.random() < substitution_rate * 0.5:
                replacement = self.rng.choice(technical_symbols[char])
                result.append(replacement)
                substituted = True
                logger.debug(f"Technical symbol: {char} -> {replacement}")
            
            if not substituted:
                result.append(char)
        
        return ''.join(result)

    def polyglot_encoding_attack(self, text: str) -> List[str]:
        """
        Create polyglot encodings that appear different to different parsers.
        
        Returns multiple variants that may be interpreted differently by
        different tokenizers, normalizers, or display systems.
        """
        variants = []
        
        # 1. Mixed script variant
        mixed_script = self.confusable_character_bypass(text, substitution_rate=0.3)
        variants.append(mixed_script)
        
        # 2. RTL injection variant  
        rtl_variant = self.bidirectional_text_attack(text, injection_rate=0.2)
        variants.append(rtl_variant)
        
        # 3. ZWJ disruption variant
        zwj_variant = self.zero_width_joiner_exploit(text, injection_rate=0.15)
        variants.append(zwj_variant)
        
        # 4. Mathematical symbols variant
        math_variant = self.font_based_visual_spoofing(text, substitution_rate=0.2)
        variants.append(math_variant)
        
        # 5. Combined approach - layer multiple techniques
        combined = text
        combined = self.confusable_character_bypass(combined, substitution_rate=0.1)
        combined = self.insert_invisible_chars(combined, insertion_rate=0.05)
        combined = self.zero_width_joiner_exploit(combined, injection_rate=0.03)
        variants.append(combined)
        
        # 6. Normalization-resistant variant
        # Use characters that normalize to same form but tokenize differently
        norm_resistant = self._create_normalization_resistant(text)
        variants.append(norm_resistant)
        
        return variants

    def _create_normalization_resistant(self, text: str) -> str:
        """Create variant resistant to Unicode normalization."""
        # Characters that survive NFKC normalization but may tokenize differently
        normalization_pairs = {
            'fi': 'ﬁ',  # Latin small ligature fi
            'fl': 'ﬂ',  # Latin small ligature fl  
            'ff': 'ﬀ',  # Latin small ligature ff
            'ffi': 'ﬃ', # Latin small ligature ffi
            'ffl': 'ﬄ', # Latin small ligature ffl
            '1/4': '¼',  # Vulgar fraction one quarter
            '1/2': '½',  # Vulgar fraction one half
            '3/4': '¾'   # Vulgar fraction three quarters
        }
        
        result = text
        for pattern, replacement in normalization_pairs.items():
            if pattern in result and self.rng.random() < 0.3:
                result = result.replace(pattern, replacement)
                logger.debug(f"Normalization-resistant: {pattern} -> {replacement}")
        
        return result

    def generate_advanced_variants(
        self,
        text: str,
        num_variants: int = 10,
        technique_weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Generate advanced Unicode variants using Phase 3 techniques.
        
        Args:
            text: Input text to mutate
            num_variants: Number of variants to generate
            technique_weights: Weights for different techniques
            
        Returns:
            List of (variant, technique_used, divergence_score) tuples
        """
        if technique_weights is None:
            technique_weights = {
                'bidirectional': 0.2,
                'confusables': 0.25, 
                'zwj_exploit': 0.15,
                'visual_spoofing': 0.2,
                'polyglot': 0.2
            }
        
        variants = []
        techniques = list(technique_weights.keys())
        weights = list(technique_weights.values())
        
        for _ in range(num_variants):
            technique = self.rng.choices(techniques, weights=weights)[0]
            
            if technique == 'bidirectional':
                variant = self.bidirectional_text_attack(text)
            elif technique == 'confusables':
                variant = self.confusable_character_bypass(text)
            elif technique == 'zwj_exploit':
                variant = self.zero_width_joiner_exploit(text)
            elif technique == 'visual_spoofing':
                variant = self.font_based_visual_spoofing(text)
            elif technique == 'polyglot':
                poly_variants = self.polyglot_encoding_attack(text)
                variant = self.rng.choice(poly_variants)
            else:
                variant = text
            
            # Calculate divergence score
            score = self._compute_advanced_divergence_score(text, variant, technique)
            variants.append((variant, technique, score))
        
        # Sort by divergence score (higher is better)
        variants.sort(key=lambda x: x[2], reverse=True)
        return variants

    def _compute_advanced_divergence_score(self, original: str, variant: str, technique: str) -> float:
        """Compute advanced divergence score for Phase 3 techniques."""
        base_score = self._compute_divergence_score(original, variant)
        
        # Technique-specific bonuses
        technique_bonuses = {
            'bidirectional': 1.5,    # RTL attacks are high value
            'confusables': 1.2,      # Extended confusables are valuable
            'zwj_exploit': 1.3,      # ZWJ exploits are sophisticated
            'visual_spoofing': 1.4,  # Font-based attacks are advanced
            'polyglot': 1.6          # Polyglot attacks are most complex
        }
        
        bonus = technique_bonuses.get(technique, 1.0)
        return base_score * bonus
