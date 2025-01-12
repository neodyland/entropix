# from https://github.com/EdwardDali/EntropixLab/blob/main/main.py
from llama_cpp import LogitsProcessorList, LogitsProcessor, Llama
from typing import List, Dict
from enum import Enum

try:
    import cupy as np
    import cupyx.scipy.special as sp

    use_cupy = True
except ImportError:
    import numpy as np
    import scipy.special as sp

    use_cupy = False
from collections import Counter, deque

LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E


class SamplerState(Enum):
    ARGMAX = 0
    SAMPLE = 1
    INSERT_COT = 2
    RESAMPLE = 3
    ADAPTIVE = 4  # New adaptive sampling strategy


class SamplerConfig:
    def __init__(self):
        self.entropy_threshold = 1.0
        self.varentropy_threshold = 1.5
        self.cot_token = "[COT]"
        self.resample_count = 5
        self.strategy_params: Dict[SamplerState, Dict[str, float]] = {
            SamplerState.ARGMAX: {
                "temperature": 0.1,
                "top_p": 1.0,
                "top_k": 1,
                "min_p": 0.0,
            },
            SamplerState.SAMPLE: {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "min_p": 0.02,
            },
            SamplerState.INSERT_COT: {
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 100,
                "min_p": 0.01,
            },
            SamplerState.RESAMPLE: {
                "temperature": 1.0,
                "top_p": 0.98,
                "top_k": 200,
                "min_p": 0.005,
            },
            SamplerState.ADAPTIVE: {
                "temperature": 0.666,
                "top_p": 0.90,
                "top_k": 27,
                "min_p": 0.03,
            },
        }
        self.repetition_penalty = 1.2
        self.max_ngram_size = 5
        self.max_ngram_repeat = 3
        self.strategy_change_batch_size = 5
        self.window_size = 50  # Size of the sliding window for weighted average
        self.decay_factor = 0.95  # Exponential decay factor for weighting

        # Adaptive sampling parameters
        self.n_adaptive_samples = 5
        self.ada_temp_logits = 0.3
        self.ada_temp_attn = 0.2
        self.ada_temp_agree = 0.2
        self.ada_top_p = 0.1
        self.ada_top_k_int = 0.3
        self.ada_top_k_agree = 0.2
        self.ada_min_p = 0.5
        self.ada_score_logits_ent = 0.1
        self.ada_score_attn_ent = 0.2
        self.ada_score_logits_vent = 0.3
        self.ada_score_attn_vent = 0.4
        self.ada_score_agree = 0.5
        self.ada_score_int = 0.6


class VarentropyLogitsProcessor(LogitsProcessor):
    def __init__(self, config: SamplerConfig):
        self.config = config
        self.strategy_counter = Counter()
        self.recent_tokens = deque(maxlen=100)
        self.current_batch = []
        self.current_strategy = SamplerState.SAMPLE
        self.tokens_since_last_change = 0
        self.entropy_window = deque(maxlen=self.config.window_size)
        self.varentropy_window = deque(maxlen=self.config.window_size)

    def __call__(self, input_ids: List[int], logits: np.ndarray) -> List[float]:
        if use_cupy:
            logits = np.asarray(logits)
        # Calculate entropy and varentropy for the current token
        entropy, varentropy = self.calculate_varentropy_logsoftmax(logits)
        self.entropy_window.append(entropy)
        self.varentropy_window.append(varentropy)

        # Check if it's time to recalculate the strategy
        if self.tokens_since_last_change % self.config.strategy_change_batch_size == 0:
            avg_entropy = self.weighted_average(self.entropy_window)
            avg_varentropy = self.weighted_average(self.varentropy_window)

            self.current_strategy = self.determine_strategy(avg_entropy, avg_varentropy)
            self.tokens_since_last_change = 0

        # Use the current strategy to sample
        if self.current_strategy == SamplerState.ADAPTIVE:
            sampled_token = self._adaptive_sample(logits)
        else:
            params = self.config.strategy_params[self.current_strategy]
            sampled_token = self._sample(logits, **params)

        # Update counters and lists
        self.strategy_counter[self.current_strategy.name] += 1
        self.tokens_since_last_change += 1
        self.current_batch.append(sampled_token)
        self.recent_tokens.append(sampled_token)

        # Check for n-gram repetition in the current batch
        if self.check_ngram_repetition(self.current_batch):
            # Increase temperature and top_k to encourage diversity
            temp_config = SamplerConfig()
            temp_config.strategy_params[SamplerState.SAMPLE]["temperature"] = 1.2
            temp_config.strategy_params[SamplerState.SAMPLE]["top_k"] = 100
            sampled_token = self._sample(
                logits, **temp_config.strategy_params[SamplerState.SAMPLE]
            )

        # Reset batch if it reaches the configured batch size
        if len(self.current_batch) == self.config.strategy_change_batch_size:
            self.current_batch = []

        # Set all logits to negative infinity except the sampled token
        new_scores = [-float("inf")] * len(logits)
        new_scores[sampled_token] = 0

        return new_scores

    def weighted_average(self, values):
        if not values:
            return 0
        weights = [self.config.decay_factor**i for i in range(len(values) - 1, -1, -1)]
        return sum(w * v for w, v in zip(weights, values)) / sum(weights)

    def determine_strategy(self, entropy: float, varentropy: float) -> SamplerState:
        if entropy < self.config.entropy_threshold:
            if varentropy < self.config.varentropy_threshold:
                return SamplerState.ARGMAX
            else:
                return SamplerState.SAMPLE
        else:
            if varentropy < self.config.varentropy_threshold:
                return SamplerState.INSERT_COT
            elif (
                varentropy > self.config.varentropy_threshold * 1.5
            ):  # Adjust this threshold as needed
                return SamplerState.RESAMPLE
            else:
                return SamplerState.ADAPTIVE

    def calculate_varentropy_logsoftmax(
        self, logits: np.ndarray, axis: int = -1
    ) -> tuple[float, float]:
        log_probs = sp.log_softmax(logits, axis=axis)
        probs = np.exp(log_probs)
        entropy = -np.sum(probs * log_probs, axis=axis) / np.log(2)
        entropy_expanded = np.expand_dims(entropy, axis=axis)
        varentropy = np.sum(
            probs * (log_probs / np.log(2) + entropy_expanded) ** 2, axis=axis
        )
        return float(entropy), float(varentropy)

    def _sample(
        self,
        logits: np.ndarray,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
    ) -> int:
        # Apply temperature and convert to probabilities
        logits = logits / temperature
        # Subtract max for numerical stability
        logits = logits - np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))

        # Apply min_p sampling
        if min_p > 0.0:
            p_max = np.max(probs)
            probs[probs < (min_p * p_max)] = 0
            # Renormalize
            probs_sum = np.sum(probs)
            if probs_sum > 0:
                probs = probs / probs_sum

        # Apply top-k sampling
        if top_k > 0:
            top_k = min(top_k, len(probs))
            indices = np.argpartition(probs, -top_k)[-top_k:]
            top_k_probs = probs[indices]
            sorted_idx = np.argsort(-top_k_probs)  # Sort in descending order
            top_k_probs = top_k_probs[sorted_idx]
            indices = indices[sorted_idx]
        else:
            top_k_probs = probs
            indices = np.arange(len(probs))

        # Apply top-p (nucleus) sampling
        if 0.0 < top_p < 1.0:
            cumulative_probs = np.cumsum(top_k_probs)
            cutoff_idx = np.searchsorted(
                cumulative_probs, np.array(top_p), side="right"
            )
            if cutoff_idx == 0:
                cutoff_idx = 1
            top_k_probs = top_k_probs[:cutoff_idx]
            indices = indices[:cutoff_idx]

            # Renormalize
            top_k_probs = top_k_probs / np.sum(top_k_probs)

        # If all probabilities are zero, return the highest probability token
        if np.sum(top_k_probs) <= 0:
            return np.argmax(probs).tolist()[0]

        # Sample from the filtered distribution
        try:
            sample_idx = np.random.choice(len(top_k_probs), p=top_k_probs, size=1)
            return indices[sample_idx].tolist()[0]
        except ValueError:
            # If sampling fails, fall back to argmax
            return np.argmax(probs).tolist()

    def _adaptive_sample(self, logits: np.ndarray) -> int:
        # Calculate metrics (simplified version as we don't have access to attention scores)
        entropy, varentropy = self.calculate_varentropy_logsoftmax(logits)

        # Adaptive sampling parameters (using fixed values from config)
        temperature = self.config.strategy_params[SamplerState.ADAPTIVE]["temperature"]
        top_p = self.config.strategy_params[SamplerState.ADAPTIVE]["top_p"]
        top_k = self.config.strategy_params[SamplerState.ADAPTIVE]["top_k"]
        min_p = self.config.strategy_params[SamplerState.ADAPTIVE]["min_p"]

        # Sample multiple times
        samples = []
        for _ in range(self.config.n_adaptive_samples):
            sample = self._sample(logits, temperature, top_p, top_k, min_p)
            samples.append(sample)

        # Score samples (simplified version)
        def score_sample(sample):
            log_prob = np.log(sp.softmax(logits, axis=-1)[sample])
            confidence_score = (1 - entropy) * self.config.ada_score_logits_ent + (
                1 - varentropy
            ) * self.config.ada_score_logits_vent
            return log_prob + confidence_score

        sample_scores = [score_sample(sample) for sample in samples]
        best_sample_idx = np.argmax(np.array(sample_scores)).tolist()
        return samples[best_sample_idx]

    def check_ngram_repetition(self, tokens: List[int]) -> bool:
        for n in range(2, self.config.max_ngram_size + 1):
            ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
            for ngram in set(ngrams):
                if ngrams.count(ngram) > self.config.max_ngram_repeat:
                    return True
        return False


def generate_response(
    model: Llama,
    prompt: str,
    max_new_tokens=4096,
    batch_size=10,
    stop: List[str] = [],
    **kwargs
):
    cfg = SamplerConfig()
    cfg.strategy_change_batch_size = batch_size
    logits_processor = VarentropyLogitsProcessor(cfg)
    logits_processors = LogitsProcessorList([logits_processor])
    default_params = cfg.strategy_params[SamplerState.SAMPLE]
    generation_params = {
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "logits_processor": logits_processors,
        "echo": False,
        "temperature": default_params["temperature"],
        "top_p": default_params["top_p"],
        "top_k": default_params["top_k"],
        "stream": True,
    }
    for k, v in kwargs.items():
        generation_params[k] = v
    generated_text = ""
    for output in model(**generation_params):
        token: str = output["choices"][0]["text"]
        generated_text += token
        yield token
        if any([x in generated_text for x in stop]):
            break
