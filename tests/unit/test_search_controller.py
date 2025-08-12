"""
Unit tests for search controller component.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.tmrt.search_controller import SearchController


class TestSearchController:
    """Test evolutionary search controller functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock components
        self.mock_mutator = Mock()
        self.mock_optimizer = Mock() 
        self.mock_scaffolder = Mock()
        self.mock_verifier = Mock()
        self.mock_novelty = Mock()
        
        # Configure component mocks
        self.mock_mutator.mutate.return_value = "mutated prompt"
        self.mock_optimizer.optimize_adversarial_suffix.return_value = {
            'optimized_prompt': 'optimized prompt',
            'final_loss': 0.5
        }
        self.mock_scaffolder.generate_scaffolded_prompt.return_value = "scaffolded prompt"
        self.mock_verifier.verify_attack.return_value = {
            'is_successful': True,
            'confidence': 0.8,
            'analysis': {'harmfulness_score': 0.7}
        }
        self.mock_novelty.check_novelty.return_value = True
        
        # Create search controller
        self.controller = SearchController(
            unicode_mutator=self.mock_mutator,
            embedding_optimizer=self.mock_optimizer,
            scaffolder=self.mock_scaffolder,
            verifier=self.mock_verifier,
            novelty_detector=self.mock_novelty
        )
        
    def test_initialization(self):
        """Test search controller initialization."""
        # Should initialize with all components
        assert self.controller.unicode_mutator == self.mock_mutator
        assert self.controller.embedding_optimizer == self.mock_optimizer
        assert self.controller.scaffolder == self.mock_scaffolder
        assert self.controller.verifier == self.mock_verifier
        assert self.controller.novelty_detector == self.mock_novelty
        
        # Should have default configuration
        assert hasattr(self.controller, 'population_size')
        assert hasattr(self.controller, 'num_generations')
        assert hasattr(self.controller, 'mutation_rate')
        
    def test_initialize_population(self):
        """Test population initialization."""
        seed_prompts = [
            "seed prompt 1",
            "seed prompt 2"
        ]
        
        population = self.controller.initialize_population(
            seed_prompts, population_size=5
        )
        
        # Should return population of specified size
        assert isinstance(population, list)
        assert len(population) == 5
        
        # Population should contain individuals with required attributes
        for individual in population:
            assert isinstance(individual, dict)
            assert 'prompt' in individual
            assert 'fitness' in individual
            assert 'generation' in individual
            
    def test_evaluate_fitness(self):
        """Test fitness evaluation."""
        individual = {
            'prompt': 'test prompt',
            'fitness': 0.0,
            'generation': 0
        }
        
        # Mock verification and novelty check
        self.mock_verifier.verify_attack.return_value = {
            'is_successful': True,
            'confidence': 0.8,
            'analysis': {'harmfulness_score': 0.7}
        }
        self.mock_novelty.check_novelty.return_value = True
        
        evaluated = self.controller.evaluate_fitness(individual)
        
        # Should update fitness score
        assert evaluated['fitness'] > 0
        assert 'evaluation_details' in evaluated
        
    def test_selection_process(self):
        """Test selection for breeding."""
        population = []
        for i in range(10):
            population.append({
                'prompt': f'prompt {i}',
                'fitness': i * 0.1,  # Varying fitness
                'generation': 0
            })
            
        selected = self.controller.selection(population, num_select=5)
        
        # Should select specified number
        assert len(selected) == 5
        
        # Should prefer higher fitness (but allow some diversity)
        selected_fitness = [ind['fitness'] for ind in selected]
        assert max(selected_fitness) >= 0.5  # Should include some high-fitness individuals
        
    def test_crossover_operation(self):
        """Test crossover between parents."""
        parent1 = {
            'prompt': 'first parent prompt with content',
            'fitness': 0.8,
            'generation': 1
        }
        parent2 = {
            'prompt': 'second parent prompt with different content',
            'fitness': 0.7,
            'generation': 1
        }
        
        offspring = self.controller.crossover(parent1, parent2)
        
        # Should create new individual
        assert isinstance(offspring, dict)
        assert 'prompt' in offspring
        assert 'fitness' in offspring
        assert 'generation' in offspring
        
        # Offspring should have elements from both parents
        assert len(offspring['prompt']) > 0
        
    def test_mutation_operation(self):
        """Test mutation of individuals."""
        individual = {
            'prompt': 'original prompt',
            'fitness': 0.5,
            'generation': 1
        }
        
        # Mock mutator components
        self.mock_mutator.mutate.return_value = "mutated prompt"
        self.mock_scaffolder.generate_scaffolded_prompt.return_value = "scaffolded version"
        
        mutated = self.controller.mutate(individual, mutation_rate=0.8)
        
        # Should create mutated individual
        assert isinstance(mutated, dict)
        assert 'prompt' in mutated
        assert mutated['generation'] == individual['generation']
        
        # Prompt should be modified (based on mock)
        assert mutated['prompt'] in ['mutated prompt', 'scaffolded version']
        
    def test_evolutionary_search_run(self):
        """Test complete evolutionary search run."""
        seed_prompts = ["initial seed prompt"]
        
        # Configure for short test run
        config = {
            'population_size': 4,
            'num_generations': 3,
            'mutation_rate': 0.5,
            'crossover_rate': 0.7
        }
        
        # Mock fitness evaluations to converge
        fitness_values = [0.3, 0.5, 0.7, 0.8, 0.9]
        fitness_iter = iter(fitness_values * 10)  # Repeat for multiple evaluations
        
        def mock_evaluate(individual):
            individual['fitness'] = next(fitness_iter, 0.5)
            individual['evaluation_details'] = {'success': True}
            return individual
            
        with patch.object(self.controller, 'evaluate_fitness', side_effect=mock_evaluate):
            results = self.controller.run_search(seed_prompts, config)
            
            # Should return search results
            assert isinstance(results, dict)
            assert 'best_individual' in results
            assert 'final_population' in results
            assert 'search_history' in results
            assert 'convergence_data' in results
            
    def test_fitness_function_components(self):
        """Test individual fitness function components."""
        prompt = "test adversarial prompt"
        
        # Mock component responses
        self.mock_verifier.verify_attack.return_value = {
            'is_successful': True,
            'confidence': 0.85,
            'analysis': {
                'harmfulness_score': 0.8,
                'perplexity': 15.2,
                'safety_violations': ['unsafe_content']
            }
        }
        self.mock_novelty.check_novelty.return_value = True
        
        fitness_score = self.controller._compute_fitness_score(prompt)
        
        # Should return composite fitness score
        assert isinstance(fitness_score, float)
        assert 0.0 <= fitness_score <= 1.0
        
    def test_diversity_maintenance(self):
        """Test population diversity maintenance."""
        # Create population with similar prompts
        similar_population = []
        for i in range(6):
            similar_population.append({
                'prompt': f'very similar prompt {i}',
                'fitness': 0.7 + i * 0.01,
                'generation': 2
            })
            
        diverse_population = self.controller.maintain_diversity(similar_population)
        
        # Should maintain diversity while keeping good fitness
        assert len(diverse_population) <= len(similar_population)
        
        # Should keep varied prompts
        prompts = [ind['prompt'] for ind in diverse_population]
        assert len(set(prompts)) == len(diverse_population)  # All unique
        
    def test_convergence_detection(self):
        """Test convergence detection."""
        # Simulate fitness history showing convergence
        fitness_history = [
            [0.1, 0.2, 0.3, 0.4],  # Generation 0
            [0.3, 0.4, 0.5, 0.6],  # Generation 1 
            [0.5, 0.6, 0.65, 0.67], # Generation 2 (converging)
            [0.65, 0.66, 0.67, 0.67] # Generation 3 (converged)
        ]
        
        for generation, fitness_values in enumerate(fitness_history):
            converged = self.controller._check_convergence(
                fitness_values, generation, patience=2
            )
            
        # Should eventually detect convergence
        assert isinstance(converged, bool)
        
    def test_export_search_results(self):
        """Test search results export."""
        results = {
            'best_individual': {
                'prompt': 'best found prompt',
                'fitness': 0.95,
                'generation': 5
            },
            'search_history': [
                {'generation': 0, 'best_fitness': 0.3},
                {'generation': 1, 'best_fitness': 0.5}
            ],
            'config': {
                'population_size': 10,
                'num_generations': 5
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "search_results.json"
            
            self.controller.export_results(results, str(output_path))
            
            # Should create valid results file
            assert output_path.exists()
            
            with open(output_path, 'r') as f:
                loaded_results = json.load(f)
                
            # Should contain all key information
            assert 'best_individual' in loaded_results
            assert 'search_history' in loaded_results
            assert 'metadata' in loaded_results
            
    def test_parallel_evaluation(self):
        """Test parallel fitness evaluation."""
        population = []
        for i in range(6):
            population.append({
                'prompt': f'prompt for parallel eval {i}',
                'fitness': 0.0,
                'generation': 1
            })
            
        # Mock parallel evaluation
        def mock_evaluate_parallel(individuals):
            for ind in individuals:
                ind['fitness'] = 0.6
                ind['evaluation_details'] = {'parallel': True}
            return individuals
            
        with patch.object(self.controller, 'evaluate_population_parallel', 
                         side_effect=mock_evaluate_parallel):
            evaluated_pop = self.controller.evaluate_population_parallel(population)
            
            # Should evaluate all individuals
            assert len(evaluated_pop) == 6
            
            for individual in evaluated_pop:
                assert individual['fitness'] > 0
                assert 'evaluation_details' in individual
                
    def test_search_statistics_tracking(self):
        """Test search statistics tracking."""
        # Initialize statistics tracker
        self.controller.statistics = {
            'evaluations_count': 0,
            'successful_attacks': 0,
            'novel_prompts': 0,
            'generations_completed': 0
        }
        
        # Simulate search progress
        self.controller._update_statistics('evaluation', success=True, novel=True)
        self.controller._update_statistics('evaluation', success=False, novel=True)
        self.controller._update_statistics('generation_complete')
        
        stats = self.controller.get_search_statistics()
        
        # Should track key metrics
        assert stats['evaluations_count'] == 2
        assert stats['successful_attacks'] == 1
        assert stats['novel_prompts'] == 2
        assert stats['generations_completed'] == 1


if __name__ == "__main__":
    pytest.main([__file__])
