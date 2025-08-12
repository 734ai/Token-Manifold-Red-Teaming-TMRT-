"""
Token-Manifold Red Teaming (TMRT) Package

A hybrid white-box/black-box pipeline for discovering minimal, natural-language
perturbations that cause safety failures by exploiting tokenizer/normalizer
divergences, embedding-space adversarial directions, and contextual role scaffolding.
"""

__version__ = "0.1.0"
__author__ = "Muzan Sano"
__email__ = "northerntribe@example.com"

from .search_controller import SearchController
from .unicode_mutators import UnicodeMutator
from .embedding_optimizer import EmbeddingOptimizer
from .scaffolder import RoleScaffolder
from .verifier import AttackVerifier
from .novelty_detector import NoveltyDetector
from .utils import load_config, setup_logging, sanitize_output

__all__ = [
    "SearchController",
    "UnicodeMutator",
    "EmbeddingOptimizer", 
    "RoleScaffolder",
    "AttackVerifier",
    "NoveltyDetector",
    "load_config",
    "setup_logging",
    "sanitize_output",
]
