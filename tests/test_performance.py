"""
Performance benchmarking for TMRT framework components.
"""

import pytest
import time
import psutil
import gc
from statistics import mean, stdev
from contextlib import contextmanager
from unittest.mock import Mock, patch

from src.tmrt.unicode_mutators import UnicodeMutator
from src.tmrt.scaffolder import RoleScaffolder
from src.tmrt.novelty_detector import NoveltyDetector
from src.tmrt.search_controller import SearchController


@contextmanager
def measure_performance():
    """Context manager to measure execution time and memory usage."""
    process = psutil.Process()
    
    # Initial measurements
    start_time = time.perf_counter()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    try:
        yield
    finally:
        # Final measurements
        end_time = time.perf_counter()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        print(f"Execution time: {execution_time:.4f} seconds")
        print(f"Memory change: {memory_delta:.2f} MB")


class TestPerformanceBenchmarks:
    """Performance benchmarking tests for TMRT components."""
    
    def setup_method(self):
        """Setup components for benchmarking."""
        self.unicode_mutator = UnicodeMutator()
        self.scaffolder = RoleScaffolder()
        self.novelty_detector = NoveltyDetector()
        
        # Mock model components for performance testing
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "mock response"
        
        self.search_controller = SearchController(
            unicode_mutator=self.unicode_mutator,
            embedding_optimizer=Mock(),
            scaffolder=self.scaffolder,
            verifier=Mock(),
            novelty_detector=self.novelty_detector
        )
        
    def test_unicode_mutation_performance(self):
        """Benchmark Unicode mutation performance."""
        test_prompts = [
            "Short prompt",
            "This is a medium length prompt with several words",
            "This is a much longer prompt that contains many words and should test the performance of Unicode mutations across larger text inputs with various character types and encoding scenarios"
        ]
        
        execution_times = []
        
        for prompt in test_prompts:
            iterations = 100
            start_time = time.perf_counter()
            
            for _ in range(iterations):
                mutated = self.unicode_mutator.mutate(prompt)
                assert isinstance(mutated, str)
                
            end_time = time.perf_counter()
            avg_time = (end_time - start_time) / iterations
            execution_times.append(avg_time)
            
        # Performance assertions
        assert all(t < 0.01 for t in execution_times), f"Unicode mutation too slow: {execution_times}"
        print(f"Unicode mutation average times: {[f'{t:.6f}s' for t in execution_times]}")
        
    def test_scaffolding_performance(self):
        """Benchmark role scaffolding performance."""
        base_prompt = "Generate instructions for"
        role_types = ['email', 'chat', 'log', 'json', 'academic']
        
        with measure_performance():
            scaffolding_times = []
            
            for role_type in role_types:
                iterations = 50
                start_time = time.perf_counter()
                
                for _ in range(iterations):
                    scaffolded = self.scaffolder.generate_scaffolded_prompt(
                        base_prompt, role_type=role_type
                    )
                    assert isinstance(scaffolded, str)
                    assert len(scaffolded) > len(base_prompt)
                    
                end_time = time.perf_counter()
                avg_time = (end_time - start_time) / iterations
                scaffolding_times.append(avg_time)
                
            # Performance assertions
            assert all(t < 0.05 for t in scaffolding_times), f"Scaffolding too slow: {scaffolding_times}"
            print(f"Scaffolding average times by role: {dict(zip(role_types, scaffolding_times))}")
            
    def test_novelty_detection_performance(self):
        """Benchmark novelty detection performance."""
        # Test with varying prompt lengths and corpus sizes
        test_prompts = [
            "Short novel prompt",
            "Medium length prompt that should be tested for novelty detection performance",
            "Very long prompt that contains extensive text to test how novelty detection scales with input length and whether the similarity computation algorithms maintain reasonable performance"
        ]
        
        with measure_performance():
            detection_times = []
            
            for prompt in test_prompts:
                iterations = 20
                start_time = time.perf_counter()
                
                for _ in range(iterations):
                    is_novel = self.novelty_detector.check_novelty(prompt)
                    assert isinstance(is_novel, bool)
                    
                end_time = time.perf_counter()
                avg_time = (end_time - start_time) / iterations
                detection_times.append(avg_time)
                
            # Performance assertions
            assert all(t < 0.1 for t in detection_times), f"Novelty detection too slow: {detection_times}"
            print(f"Novelty detection times: {[f'{t:.4f}s' for t in detection_times]}")
            
    def test_corpus_scaling_performance(self):
        """Test novelty detection performance with growing corpus."""
        base_corpus_sizes = [10, 50, 100, 200]
        test_prompt = "Test prompt for corpus scaling"
        
        scaling_times = []
        
        for corpus_size in base_corpus_sizes:
            # Add prompts to corpus
            additional_prompts = [f"Corpus prompt {i}" for i in range(corpus_size)]
            self.novelty_detector.add_to_corpus(additional_prompts, source=f"scale_test_{corpus_size}")
            
            # Measure detection time
            iterations = 10
            start_time = time.perf_counter()
            
            for _ in range(iterations):
                is_novel = self.novelty_detector.check_novelty(test_prompt)
                
            end_time = time.perf_counter()
            avg_time = (end_time - start_time) / iterations
            scaling_times.append(avg_time)
            
        print(f"Corpus scaling times: {dict(zip(base_corpus_sizes, scaling_times))}")
        
        # Should scale reasonably (not exponentially)
        if len(scaling_times) > 1:
            time_ratios = [scaling_times[i] / scaling_times[0] for i in range(1, len(scaling_times))]
            corpus_ratios = [base_corpus_sizes[i] / base_corpus_sizes[0] for i in range(1, len(base_corpus_sizes))]
            
            # Time scaling should be better than quadratic
            for i, (time_ratio, corpus_ratio) in enumerate(zip(time_ratios, corpus_ratios)):
                assert time_ratio < corpus_ratio ** 1.5, f"Poor scaling at size {base_corpus_sizes[i+1]}: {time_ratio} vs {corpus_ratio}"
                
    def test_memory_efficiency(self):
        """Test memory efficiency of framework components."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large datasets for testing
        large_prompt_set = [f"Large test prompt {i} with substantial content" * 20 for i in range(100)]
        
        # Test Unicode mutations
        with measure_performance():
            mutated_prompts = []
            for prompt in large_prompt_set:
                mutated = self.unicode_mutator.mutate(prompt)
                mutated_prompts.append(mutated)
                
            # Clear references
            del mutated_prompts
            gc.collect()
            
        # Test scaffolding  
        with measure_performance():
            scaffolded_prompts = []
            for prompt in large_prompt_set[:20]:  # Smaller set for scaffolding
                scaffolded = self.scaffolder.generate_scaffolded_prompt(prompt, role_type='email')
                scaffolded_prompts.append(scaffolded)
                
            del scaffolded_prompts
            gc.collect()
            
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable
        assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.2f} MB"
        print(f"Total memory growth: {memory_growth:.2f} MB")
        
    def test_batch_processing_performance(self):
        """Test performance of batch processing operations."""
        batch_sizes = [1, 5, 10, 20]
        base_prompt = "Test prompt for batch processing"
        
        batch_times = []
        
        for batch_size in batch_sizes:
            batch_prompts = [f"{base_prompt} {i}" for i in range(batch_size)]
            
            start_time = time.perf_counter()
            
            # Process batch
            results = []
            for prompt in batch_prompts:
                mutated = self.unicode_mutator.mutate(prompt)
                is_novel = self.novelty_detector.check_novelty(mutated)
                results.append((mutated, is_novel))
                
            end_time = time.perf_counter()
            total_time = end_time - start_time
            per_item_time = total_time / batch_size
            
            batch_times.append(per_item_time)
            
        print(f"Batch processing per-item times: {dict(zip(batch_sizes, batch_times))}")
        
        # Should show some efficiency gains with batching
        if len(batch_times) > 1:
            # Later batches shouldn't be much slower per item
            assert batch_times[-1] < batch_times[0] * 1.5, "Poor batch scaling"
            
    def test_concurrent_access_performance(self):
        """Test performance under concurrent access scenarios."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        num_threads = 4
        prompts_per_thread = 10
        
        def worker_thread(thread_id):
            """Worker thread for concurrent testing."""
            thread_results = []
            for i in range(prompts_per_thread):
                prompt = f"Thread {thread_id} prompt {i}"
                
                # Perform operations
                mutated = self.unicode_mutator.mutate(prompt)
                is_novel = self.novelty_detector.check_novelty(mutated)
                
                thread_results.append((mutated, is_novel))
                
            results_queue.put((thread_id, thread_results))
            
        # Run concurrent threads
        start_time = time.perf_counter()
        
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Collect results
        all_results = {}
        while not results_queue.empty():
            thread_id, results = results_queue.get()
            all_results[thread_id] = results
            
        print(f"Concurrent processing time: {total_time:.4f}s")
        print(f"Threads completed: {len(all_results)}/{num_threads}")
        
        # Should complete all threads successfully
        assert len(all_results) == num_threads
        for thread_id, results in all_results.items():
            assert len(results) == prompts_per_thread
            
    def test_evolutionary_search_performance(self):
        """Test evolutionary search performance scaling."""
        population_sizes = [4, 8, 12]
        generations = [2, 3]
        
        # Mock verification for performance testing
        with patch.object(self.search_controller.verifier, 'verify_attack') as mock_verify:
            mock_verify.return_value = {
                'is_successful': True,
                'confidence': 0.7,
                'analysis': {'harmfulness_score': 0.6}
            }
            
            with patch.object(self.search_controller.novelty_detector, 'check_novelty') as mock_novelty:
                mock_novelty.return_value = True
                
                search_times = {}
                
                for pop_size in population_sizes:
                    for gen_count in generations:
                        config = {
                            'population_size': pop_size,
                            'num_generations': gen_count,
                            'mutation_rate': 0.5
                        }
                        
                        start_time = time.perf_counter()
                        
                        results = self.search_controller.run_search(
                            ["seed prompt"], config
                        )
                        
                        end_time = time.perf_counter()
                        search_time = end_time - start_time
                        
                        search_times[(pop_size, gen_count)] = search_time
                        
                        # Should complete successfully
                        assert 'best_individual' in results
                        
        print(f"Search performance times: {search_times}")
        
        # Performance should scale reasonably
        for (pop_size, gen_count), search_time in search_times.items():
            expected_operations = pop_size * gen_count
            time_per_operation = search_time / expected_operations
            assert time_per_operation < 0.1, f"Search too slow: {time_per_operation:.4f}s per operation"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
