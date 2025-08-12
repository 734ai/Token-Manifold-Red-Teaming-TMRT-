"""
Embedding-space adversarial optimizer.

This module uses white-box access to compute embedding perturbations that increase
the probability of unsafe outputs, then maps perturbations back to fluent natural
text via constrained decoding.
"""

import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import math

logger = logging.getLogger(__name__)


class EmbeddingOptimizer:
    """White-box embedding-space adversarial optimizer."""
    
    def __init__(
        self,
        model_name: str = "gpt-oss-20b",
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        max_length: int = 512
    ):
        """
        Initialize embedding optimizer.
        
        Args:
            model_name: Name of the model to optimize against
            device: Device to run on ("auto", "cuda", "cpu")
            torch_dtype: Torch dtype for model weights
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Setup device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing EmbeddingOptimizer on {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=True
        )
        self.model.eval()
        
        # Cache embedding layer for optimization
        self.embedding_layer = self.model.get_input_embeddings()
        self.vocab_size = self.embedding_layer.num_embeddings
        self.embed_dim = self.embedding_layer.embedding_dim
        
        logger.info(f"Loaded model with vocab_size={self.vocab_size}, embed_dim={self.embed_dim}")
    
    def get_embedding_gradients(
        self,
        input_ids: torch.Tensor,
        target_tokens: Union[List[int], torch.Tensor],
        target_positions: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Compute gradients of target token log-probabilities w.r.t. input embeddings.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            target_tokens: Target token IDs to maximize probability for
            target_positions: Positions where to evaluate target tokens (default: next position)
            
        Returns:
            Gradients w.r.t. input embeddings [batch_size, seq_len, embed_dim]
        """
        input_ids = input_ids.to(self.device)
        
        if isinstance(target_tokens, list):
            target_tokens = torch.tensor(target_tokens, device=self.device)
        target_tokens = target_tokens.to(self.device)
        
        # Get input embeddings
        inputs_embeds = self.embedding_layer(input_ids)
        inputs_embeds.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(inputs_embeds=inputs_embeds)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # Compute target loss
        if target_positions is None:
            # Use next token position for each sequence
            target_positions = [input_ids.size(1) - 1] * input_ids.size(0)
        
        losses = []
        for batch_idx in range(input_ids.size(0)):
            pos = min(target_positions[batch_idx], logits.size(1) - 1)
            target_logits = logits[batch_idx, pos]  # [vocab_size]
            
            # Compute loss for target tokens
            target_probs = F.softmax(target_logits, dim=-1)
            target_loss = -torch.log(target_probs[target_tokens]).sum()
            losses.append(target_loss)
        
        total_loss = torch.stack(losses).sum()
        
        # Backward pass
        total_loss.backward()
        
        return inputs_embeds.grad.detach()
    
    def optimize_embeddings(
        self,
        input_text: str,
        target_tokens: Union[List[str], List[int]],
        num_steps: int = 50,
        learning_rate: float = 0.1,
        perturbation_budget: float = 0.1,
        norm_constraint: str = "l2"
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Optimize input embeddings to increase target token probabilities.
        
        Args:
            input_text: Input text to perturb
            target_tokens: Target tokens to maximize (strings or token IDs)
            num_steps: Number of optimization steps
            learning_rate: Learning rate for gradient ascent
            perturbation_budget: Maximum perturbation norm
            norm_constraint: Norm constraint type ("l2", "linf")
            
        Returns:
            Optimized embeddings and loss history
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids[:, :self.max_length]  # Truncate if necessary
        
        # Convert target tokens to IDs if needed
        if isinstance(target_tokens[0], str):
            target_token_ids = []
            for token in target_tokens:
                token_id = self.tokenizer.encode(token, add_special_tokens=False)
                if token_id:
                    target_token_ids.append(token_id[0])
            target_tokens = target_token_ids
        
        logger.info(f"Optimizing for target tokens: {target_tokens}")
        
        # Initialize embeddings
        original_embeds = self.embedding_layer(input_ids.to(self.device))
        perturbed_embeds = original_embeds.clone().detach().requires_grad_(True)
        
        # Optimization loop
        loss_history = []
        optimizer = torch.optim.Adam([perturbed_embeds], lr=learning_rate)
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs_embeds=perturbed_embeds)
            logits = outputs.logits[:, -1, :]  # Last token logits
            
            # Compute target loss (negative for maximization)
            target_probs = F.softmax(logits, dim=-1)
            target_loss = -torch.log(target_probs[0, target_tokens]).sum()
            
            # Add perturbation constraint
            perturbation = perturbed_embeds - original_embeds
            if norm_constraint == "l2":
                perturbation_norm = torch.norm(perturbation, p=2)
                constraint_loss = F.relu(perturbation_norm - perturbation_budget)
            elif norm_constraint == "linf":
                perturbation_norm = torch.norm(perturbation, p=float('inf'))
                constraint_loss = F.relu(perturbation_norm - perturbation_budget)
            else:
                constraint_loss = 0.0
            
            total_loss = target_loss + 10.0 * constraint_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            loss_history.append(total_loss.item())
            
            if step % 10 == 0:
                logger.debug(f"Step {step}: loss={total_loss.item():.4f}, "
                           f"target_loss={target_loss.item():.4f}, "
                           f"constraint_loss={constraint_loss:.4f}")
        
        return perturbed_embeds.detach(), loss_history
    
    def project_to_tokens(
        self,
        perturbed_embeds: torch.Tensor,
        original_input_ids: torch.Tensor,
        beam_size: int = 5,
        preserve_positions: Optional[List[int]] = None
    ) -> List[Tuple[str, float]]:
        """
        Project perturbed embeddings back to valid token sequences.
        
        Args:
            perturbed_embeds: Optimized embeddings [1, seq_len, embed_dim]
            original_input_ids: Original input token IDs
            beam_size: Beam size for search
            preserve_positions: Positions to keep unchanged
            
        Returns:
            List of (decoded_text, score) tuples
        """
        seq_len = perturbed_embeds.size(1)
        candidates = []
        
        # For each position, find closest tokens in embedding space
        for pos in range(seq_len):
            if preserve_positions and pos in preserve_positions:
                # Keep original token
                candidates.append([original_input_ids[0, pos].item()])
                continue
                
            embed_vec = perturbed_embeds[0, pos]  # [embed_dim]
            
            # Compute similarities to all vocabulary embeddings
            vocab_embeds = self.embedding_layer.weight  # [vocab_size, embed_dim]
            similarities = F.cosine_similarity(
                embed_vec.unsqueeze(0), 
                vocab_embeds, 
                dim=-1
            )
            
            # Get top-k most similar tokens
            top_k = min(beam_size, self.vocab_size)
            top_token_ids = similarities.argsort(descending=True)[:top_k].tolist()
            
            candidates.append(top_token_ids)
        
        # Generate sequences using beam search over candidate tokens
        sequences = self._beam_search_candidates(candidates, beam_size)
        
        # Decode and score sequences
        results = []
        for token_ids, score in sequences:
            try:
                decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                results.append((decoded, score))
            except Exception as e:
                logger.warning(f"Failed to decode sequence: {e}")
                continue
        
        return results[:beam_size]
    
    def _beam_search_candidates(
        self,
        candidates: List[List[int]],
        beam_size: int
    ) -> List[Tuple[List[int], float]]:
        """
        Perform beam search over candidate token lists.
        
        Args:
            candidates: List of candidate token lists for each position
            beam_size: Beam size
            
        Returns:
            List of (token_sequence, score) tuples
        """
        # Initialize beam with first position candidates
        beam = [([], 0.0)]
        
        for pos_candidates in candidates:
            new_beam = []
            
            for sequence, score in beam:
                for token_id in pos_candidates[:beam_size]:
                    new_sequence = sequence + [token_id]
                    # Simple scoring: uniform probability for now
                    # In practice, would use language model scoring
                    new_score = score - math.log(len(pos_candidates))
                    new_beam.append((new_sequence, new_score))
            
            # Keep top beam_size sequences
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_size]
        
        return beam
    
    def adversarial_suffix_search(
        self,
        input_text: str,
        target_response: str,
        suffix_length: int = 10,
        num_iterations: int = 100,
        batch_size: int = 8
    ) -> List[str]:
        """
        Search for adversarial suffixes that elicit target responses.
        
        Args:
            input_text: Base input text
            target_response: Target response to elicit
            suffix_length: Length of suffix to optimize
            num_iterations: Number of search iterations
            batch_size: Batch size for parallel search
            
        Returns:
            List of adversarial suffixes
        """
        # Tokenize inputs
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        target_ids = self.tokenizer.encode(target_response, return_tensors="pt")
        
        # Initialize random suffixes
        suffixes = []
        for _ in range(batch_size):
            suffix_tokens = torch.randint(
                0, self.vocab_size, 
                (suffix_length,), 
                device=self.device
            )
            suffixes.append(suffix_tokens)
        
        best_suffixes = []
        
        for iteration in range(num_iterations):
            # Evaluate current suffixes
            for suffix in suffixes:
                # Combine input with suffix
                combined_ids = torch.cat([input_ids.squeeze(0), suffix])
                combined_ids = combined_ids.unsqueeze(0).to(self.device)
                
                # Compute loss for target response
                with torch.no_grad():
                    outputs = self.model(combined_ids)
                    logits = outputs.logits[0, -suffix_length:]
                    
                    # Simple scoring: how well suffix leads to target
                    target_logprobs = F.log_softmax(logits, dim=-1)
                    score = target_logprobs[torch.arange(suffix_length), suffix].sum().item()
                    
                    decoded_suffix = self.tokenizer.decode(suffix, skip_special_tokens=True)
                    best_suffixes.append((decoded_suffix, score))
            
            # Mutate suffixes for next iteration
            for i, suffix in enumerate(suffixes):
                # Random mutation
                if torch.rand(1) < 0.1:  # 10% mutation rate
                    pos = torch.randint(0, suffix_length, (1,)).item()
                    suffix[pos] = torch.randint(0, self.vocab_size, (1,)).item()
        
        # Return top suffixes
        best_suffixes.sort(key=lambda x: x[1], reverse=True)
        return [suffix for suffix, _ in best_suffixes[:10]]
    
    def compute_embedding_perturbation_budget(
        self, 
        input_text: str, 
        target_perplexity_ratio: float = 1.5
    ) -> float:
        """
        Compute appropriate perturbation budget to maintain text fluency.
        
        Args:
            input_text: Input text
            target_perplexity_ratio: Maximum allowed perplexity increase
            
        Returns:
            Recommended perturbation budget
        """
        # Tokenize and get embeddings
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        
        with torch.no_grad():
            # Original perplexity
            outputs = self.model(input_ids)
            orig_loss = F.cross_entropy(
                outputs.logits[0, :-1], 
                input_ids[0, 1:], 
                reduction='mean'
            )
            orig_perplexity = torch.exp(orig_loss).item()
            
            # Estimate perturbation budget through sampling
            embeds = self.embedding_layer(input_ids)
            embed_std = embeds.std().item()
            
            # Heuristic: budget should be small fraction of embedding std
            budget = embed_std * 0.1
            
            logger.info(f"Original perplexity: {orig_perplexity:.2f}")
            logger.info(f"Recommended perturbation budget: {budget:.4f}")
            
            return budget

    # ===============================
    # PHASE 3: ENHANCED EMBEDDING OPTIMIZATION
    # ===============================

    def adversarial_suffix_search(
        self,
        prompt: str,
        target_response: str,
        suffix_length: int = 20,
        num_iterations: int = 500,
        learning_rate: float = 0.01,
        temperature: float = 1.0
    ) -> List[str]:
        """
        Advanced adversarial suffix search using gradient-based optimization.
        
        This implements the GCG (Greedy Coordinate Gradient) attack for finding
        adversarial suffixes that maximize the probability of target responses.
        """
        # Tokenize inputs
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        target_ids = self.tokenizer.encode(target_response, return_tensors="pt").to(self.device)
        
        # Initialize suffix with random tokens
        suffix_ids = torch.randint(
            0, self.vocab_size, (1, suffix_length), 
            dtype=prompt_ids.dtype, device=self.device
        )
        
        best_suffixes = []
        
        for iteration in range(num_iterations):
            # Create full sequence: prompt + suffix + target
            full_ids = torch.cat([prompt_ids, suffix_ids], dim=1)
            
            # Get embeddings and make them require gradients
            with torch.no_grad():
                full_embeds = self.embedding_layer(full_ids)
            full_embeds.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(inputs_embeds=full_embeds)
            logits = outputs.logits[0]
            
            # Compute loss: maximize probability of target tokens
            suffix_end = prompt_ids.shape[1] + suffix_length
            target_start = suffix_end
            
            if target_start < logits.shape[0]:
                target_logits = logits[target_start-1:target_start-1+target_ids.shape[1]]
                loss = F.cross_entropy(target_logits, target_ids[0])
                
                # Backward pass
                loss.backward()
                
                # Get gradients for suffix embeddings only
                suffix_start = prompt_ids.shape[1]
                suffix_grads = full_embeds.grad[0, suffix_start:suffix_end]
                
                # Convert embedding gradients to token gradients
                embedding_matrix = self.embedding_layer.weight  # [vocab_size, embed_dim]
                token_grads = torch.matmul(suffix_grads, embedding_matrix.T)  # [suffix_len, vocab_size]
                
                # Greedy coordinate gradient update
                for pos in range(suffix_length):
                    # Find token with highest gradient
                    best_token = token_grads[pos].argmax().item()
                    suffix_ids[0, pos] = best_token
                
                # Evaluate current suffix
                if iteration % 50 == 0:
                    suffix_text = self.tokenizer.decode(suffix_ids[0], skip_special_tokens=True)
                    best_suffixes.append((suffix_text, -loss.item()))
                    logger.debug(f"Iteration {iteration}, Loss: {loss.item():.4f}, Suffix: {suffix_text}")
        
        # Return top suffixes
        best_suffixes.sort(key=lambda x: x[1], reverse=True)
        return [suffix for suffix, _ in best_suffixes[:5]]

    def genetic_embedding_optimization(
        self,
        input_text: str,
        objective_fn,
        population_size: int = 50,
        num_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Gradient-free optimization using genetic algorithms on embeddings.
        
        This explores the embedding space without requiring gradients,
        useful for black-box optimization scenarios.
        """
        from deap import base, creator, tools, algorithms
        import copy
        
        # Setup DEAP if not already done
        if not hasattr(creator, 'FitnessMax'):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, 'Individual'):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Get original embedding
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            original_embeds = self.embedding_layer(input_ids)[0].cpu().numpy()
        
        def create_individual():
            # Start with original embedding and add small random perturbations
            individual = original_embeds.copy()
            noise = np.random.normal(0, 0.01, individual.shape)
            individual += noise
            return creator.Individual(individual)
        
        def mutate_embedding(individual):
            """Mutate embedding by adding Gaussian noise."""
            for i in range(len(individual)):
                if np.random.random() < mutation_rate:
                    individual[i] += np.random.normal(0, 0.005, individual[i].shape)
            return individual,
        
        def crossover_embedding(ind1, ind2):
            """Crossover between two embedding individuals."""
            for i in range(len(ind1)):
                if np.random.random() < crossover_rate:
                    # Blend crossover
                    alpha = np.random.random()
                    temp1 = alpha * ind1[i] + (1 - alpha) * ind2[i]
                    temp2 = alpha * ind2[i] + (1 - alpha) * ind1[i]
                    ind1[i] = temp1
                    ind2[i] = temp2
            return ind1, ind2
        
        def evaluate_embedding(individual):
            """Evaluate embedding using objective function."""
            # Convert numpy array back to tensor
            embed_tensor = torch.FloatTensor(individual).unsqueeze(0).to(self.device)
            
            # Project back to token space for evaluation
            projected_text = self.project_embeddings_to_tokens(embed_tensor)[0]
            
            # Evaluate using objective function
            try:
                score = objective_fn(projected_text)
                return (float(score),)
            except:
                return (-1000.0,)  # Heavy penalty for invalid outputs
        
        # Register genetic operators
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate_embedding)
        toolbox.register("mate", crossover_embedding)
        toolbox.register("mutate", mutate_embedding)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run genetic algorithm
        population = toolbox.population(n=population_size)
        
        # Evaluate initial population
        fitnesses = map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Evolution loop
        for generation in range(num_generations):
            # Select and clone offspring
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            
            # Apply crossover and mutation
            for i in range(0, len(offspring) - 1, 2):
                if np.random.random() < crossover_rate:
                    toolbox.mate(offspring[i], offspring[i + 1])
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values
            
            for individual in offspring:
                if np.random.random() < mutation_rate:
                    toolbox.mutate(individual)
                    del individual.fitness.values
            
            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            population[:] = offspring
            
            if generation % 20 == 0:
                best_fitness = max(ind.fitness.values[0] for ind in population)
                logger.debug(f"Generation {generation}, Best fitness: {best_fitness:.4f}")
        
        # Return best individuals
        population.sort(key=lambda x: x.fitness.values[0], reverse=True)
        results = []
        
        for i, individual in enumerate(population[:10]):
            embed_tensor = torch.FloatTensor(individual).unsqueeze(0).to(self.device)
            projected_text = self.project_embeddings_to_tokens(embed_tensor)[0]
            fitness = individual.fitness.values[0]
            results.append((projected_text, fitness))
        
        return results

    def multi_objective_embedding_optimization(
        self,
        input_text: str,
        objectives: Dict[str, callable],
        weights: Optional[Dict[str, float]] = None,
        num_iterations: int = 200
    ) -> List[Tuple[str, Dict[str, float]]]:
        """
        Multi-objective optimization balancing attack success and stealth.
        
        Optimizes embeddings to maximize attack success while minimizing
        detection by maintaining semantic similarity and fluency.
        """
        if weights is None:
            weights = {obj: 1.0 for obj in objectives.keys()}
        
        # Get original embedding
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        original_embeds = self.embedding_layer(input_ids)
        
        # Initialize population of embeddings
        population_size = 20
        population = []
        
        for _ in range(population_size):
            # Add small random perturbations to original
            noise = torch.randn_like(original_embeds) * 0.01
            perturbed = original_embeds + noise
            population.append(perturbed.clone().detach().requires_grad_(True))
        
        optimizer = torch.optim.Adam(population, lr=0.001)
        
        pareto_front = []
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            total_loss = 0
            for embed in population:
                # Project to text
                projected_texts = self.project_embeddings_to_tokens(embed)
                if projected_texts:
                    text = projected_texts[0]
                    
                    # Evaluate all objectives
                    objective_scores = {}
                    for obj_name, obj_func in objectives.items():
                        try:
                            score = obj_func(text)
                            objective_scores[obj_name] = score
                        except:
                            objective_scores[obj_name] = -1000  # Heavy penalty
                    
                    # Weighted combination
                    weighted_score = sum(
                        weights.get(obj, 1.0) * score 
                        for obj, score in objective_scores.items()
                    )
                    
                    # Convert to loss (minimize negative reward)
                    loss = -weighted_score
                    total_loss += loss
                    
                    # Track Pareto-optimal solutions
                    if iteration % 20 == 0:
                        pareto_front.append((text, objective_scores.copy()))
            
            # Backward pass
            if total_loss != 0:
                total_loss.backward()
                optimizer.step()
            
            if iteration % 50 == 0:
                logger.debug(f"Multi-objective iteration {iteration}, Loss: {total_loss.item():.4f}")
        
        # Return Pareto-optimal solutions
        # Simple Pareto filtering (could be improved with proper dominance sorting)
        pareto_filtered = []
        for text, scores in pareto_front:
            is_dominated = False
            for other_text, other_scores in pareto_front:
                if (other_text != text and 
                    all(other_scores[obj] >= scores[obj] for obj in scores) and
                    any(other_scores[obj] > scores[obj] for obj in scores)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_filtered.append((text, scores))
        
        return pareto_filtered[:10]  # Return top 10

    def embedding_interpolation_attack(
        self,
        benign_text: str,
        malicious_text: str,
        num_interpolations: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Create interpolated embeddings between benign and malicious texts.
        
        This technique creates a gradient of texts that gradually transition
        from benign to malicious, potentially bypassing detection.
        """
        # Get embeddings for both texts
        benign_ids = self.tokenizer.encode(benign_text, return_tensors="pt").to(self.device)
        malicious_ids = self.tokenizer.encode(malicious_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            benign_embeds = self.embedding_layer(benign_ids)
            malicious_embeds = self.embedding_layer(malicious_ids)
        
        # Handle different sequence lengths by padding/truncating
        min_len = min(benign_embeds.shape[1], malicious_embeds.shape[1])
        benign_embeds = benign_embeds[:, :min_len]
        malicious_embeds = malicious_embeds[:, :min_len]
        
        interpolations = []
        
        # Create interpolations
        for i in range(num_interpolations + 1):
            alpha = i / num_interpolations  # 0 to 1
            
            # Linear interpolation
            interpolated = (1 - alpha) * benign_embeds + alpha * malicious_embeds
            
            # Project back to text
            projected_texts = self.project_embeddings_to_tokens(interpolated)
            if projected_texts:
                text = projected_texts[0]
                
                # Compute maliciousness score (how close to malicious end)
                maliciousness = alpha
                
                interpolations.append((text, maliciousness))
                logger.debug(f"Interpolation {alpha:.2f}: {text[:50]}...")
        
        return interpolations

    def attention_based_perturbation(
        self,
        input_text: str,
        target_layer: int = -1,
        attention_threshold: float = 0.1
    ) -> List[str]:
        """
        Target perturbations at high-attention tokens.
        
        Uses attention weights to identify important tokens and focuses
        perturbations on those positions for maximum impact.
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Get attention weights
            outputs = self.model(input_ids, output_attentions=True)
            
            # Extract attention from target layer (default: last layer)
            attention_weights = outputs.attentions[target_layer][0]  # [num_heads, seq_len, seq_len]
            
            # Average across heads and get self-attention weights
            avg_attention = attention_weights.mean(dim=0)  # [seq_len, seq_len]
            self_attention = avg_attention.diag()  # Self-attention scores
            
            # Identify high-attention positions
            high_attention_mask = self_attention > attention_threshold
            high_attention_positions = high_attention_mask.nonzero().squeeze(-1)
            
            logger.debug(f"High attention positions: {high_attention_positions.tolist()}")
        
        # Create perturbations focused on high-attention positions
        variants = []
        original_embeds = self.embedding_layer(input_ids)
        
        for _ in range(5):
            perturbed_embeds = original_embeds.clone()
            
            # Add stronger perturbations at high-attention positions
            for pos in high_attention_positions:
                if pos < perturbed_embeds.shape[1]:
                    # Add targeted noise
                    noise = torch.randn_like(perturbed_embeds[0, pos]) * 0.02
                    perturbed_embeds[0, pos] += noise
            
            # Project back to text
            projected_texts = self.project_embeddings_to_tokens(perturbed_embeds)
            if projected_texts:
                variants.append(projected_texts[0])
        
        return variants
