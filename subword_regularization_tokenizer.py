import re
import math
import random
from typing import Dict, List
from collections import defaultdict


class SubwordRegularizationTokenizer:
    
    def __init__(self, vocab_size: int = 1000, alpha: float = 0.2):
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}
        
        self.pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)|"""
            r"""\w+|"""
            r"""[^\s\w]+|"""
            r"""\s+""",
            re.IGNORECASE | re.UNICODE
        )
        
    def _get_initial_vocab(self, text: str) -> Dict[str, float]:
        vocab = {}
        chars = set()
        
        for chunk in self.pattern.findall(text):
            chars.update(list(chunk))
        
        log_prob = -math.log(len(chars)) if chars else 0
        for char in chars:
            vocab[char] = log_prob
            
        return vocab
    
    def _get_subword_candidates(self, text: str, max_length: int = 10) -> Dict[str, int]:
        candidates = defaultdict(int)
        chunks = self.pattern.findall(text)
        
        for chunk in chunks:
            chunk_len = len(chunk)
            for i in range(chunk_len):
                for j in range(i + 1, min(i + max_length + 1, chunk_len + 1)):
                    subword = chunk[i:j]
                    candidates[subword] += 1
                    
        return dict(candidates)
    
    def _forward_algorithm(self, text: str, vocab: Dict[str, float]) -> List[float]:
        text_len = len(text)
        forward = [float('-inf')] * (text_len + 1)
        forward[0] = 0.0
        
        for i in range(text_len):
            if forward[i] == float('-inf'):
                continue
                
            for j in range(i + 1, text_len + 1):
                token = text[i:j]
                if token in vocab:
                    score = vocab[token] * (1.0 / (1.0 + self.alpha))
                    forward[j] = self._log_sum_exp(forward[j], forward[i] + score)
        
        return forward
    
    def _log_sum_exp(self, a: float, b: float) -> float:
        if a == float('-inf'):
            return b
        if b == float('-inf'):
            return a
        if a > b:
            return a + math.log(1 + math.exp(b - a))
        else:
            return b + math.log(1 + math.exp(a - b))
    
    def _sample_segmentation(self, text: str, vocab: Dict[str, float]) -> List[str]:
        text_len = len(text)
        
        forward = self._forward_algorithm(text, vocab)
        
        tokens = []
        pos = text_len
        
        while pos > 0:
            candidates = []
            
            for start in range(pos):
                token = text[start:pos]
                if token in vocab and forward[start] != float('-inf'):
                    score = vocab[token] * (1.0 / (1.0 + self.alpha))
                    prob = forward[start] + score - forward[pos]
                    candidates.append((start, token, prob))
            
            if not candidates:
                if pos > 0:
                    char = text[pos-1]
                    tokens.append(char)
                    pos -= 1
                continue
            
            log_probs = [c[2] for c in candidates]
            
            max_log_prob = max(log_probs)
            probs = [math.exp(lp - max_log_prob) for lp in log_probs]
            total = sum(probs)
            probs = [p / total for p in probs]
            
            idx = self._sample_from_probs(probs)
            start, token, _ = candidates[idx]
            
            tokens.append(token)
            pos = start
        
        tokens.reverse()
        return tokens
    
    def _sample_from_probs(self, probs: List[float]) -> int:
        r = random.random()
        cumsum = 0.0
        
        for i, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                return i
        
        return len(probs) - 1
    
    def _viterbi_decode(self, text: str, vocab: Dict[str, float]) -> List[str]:
        text_len = len(text)
        
        best_score = [float('-inf')] * (text_len + 1)
        best_score[0] = 0.0
        best_token_end = [None] * (text_len + 1)
        
        for i in range(text_len):
            if best_score[i] == float('-inf'):
                continue
                
            for j in range(i + 1, text_len + 1):
                token = text[i:j]
                if token in vocab:
                    score = best_score[i] + vocab[token]
                    if score > best_score[j]:
                        best_score[j] = score
                        best_token_end[j] = i
        
        tokens = []
        pos = text_len
        
        while pos > 0 and best_token_end[pos] is not None:
            start = best_token_end[pos]
            tokens.append(text[start:pos])
            pos = start
        
        if pos > 0:
            for i in range(pos):
                tokens.append(text[i])
        
        tokens.reverse()
        return tokens
    
    def _update_vocab(self, text: str, vocab: Dict[str, float], 
                     num_samples: int = 10) -> Dict[str, float]:
        expected_counts = defaultdict(float)
        chunks = self.pattern.findall(text)
        
        for chunk in chunks:
            for _ in range(num_samples):
                tokens = self._sample_segmentation(chunk, vocab)
                for token in tokens:
                    expected_counts[token] += 1.0 / num_samples
        
        total_count = sum(expected_counts.values())
        new_vocab = {}
        
        for token, count in expected_counts.items():
            if count > 0:
                new_vocab[token] = math.log(count / total_count)
        
        return new_vocab
    
    def _prune_vocab(self, vocab: Dict[str, float], target_size: int) -> Dict[str, float]:
        protected = {token for token in vocab if len(token) == 1}
        regular = {k: v for k, v in vocab.items() if k not in protected}
        
        sorted_regular = sorted(regular.items(), key=lambda x: x[1], reverse=True)
        keep_count = max(0, target_size - len(protected))
        
        kept = dict(sorted_regular[:keep_count])
        pruned = {**{k: vocab[k] for k in protected}, **kept}
        
        if pruned:
            total_prob = sum(math.exp(p) for p in pruned.values())
            log_total = math.log(total_prob) if total_prob > 0 else 0
            pruned = {k: v - log_total for k, v in pruned.items()}
        
        return pruned
    
    def train(self, text: str, num_iterations: int = 10, 
              num_samples: int = 10, verbose: bool = False):
        if verbose:
            print("Initializing vocabulary...")
        
        vocab = self._get_initial_vocab(text)
        
        if verbose:
            print("Extracting subword candidates...")
        
        candidates = self._get_subword_candidates(text, max_length=10)
        
        initial_size = min(len(candidates), self.vocab_size * 3)
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        
        total_freq = sum(candidates.values())
        for token, count in sorted_candidates[:initial_size]:
            if token not in vocab:
                vocab[token] = math.log(count / total_freq)
        
        if verbose:
            print(f"Initial vocabulary size: {len(vocab)}")
        
        for iteration in range(num_iterations):
            if verbose:
                print(f"\nIteration {iteration + 1}/{num_iterations}")
            
            vocab = self._update_vocab(text, vocab, num_samples)
            
            target_size = max(
                self.vocab_size,
                int(len(vocab) * (1 - 0.1 * (iteration + 1) / num_iterations))
            )
            vocab = self._prune_vocab(vocab, target_size)
            
            if verbose:
                print(f"  Vocabulary size: {len(vocab)}")
        
        self.vocab = self._prune_vocab(vocab, self.vocab_size)
        
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)
        self.token_to_id = {token: idx for idx, (token, _) in enumerate(sorted_vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        if verbose:
            print(f"\nTraining complete. Final vocabulary size: {len(self.vocab)}")
    
    def encode(self, text: str, use_sampling: bool = False, 
               num_samples: int = 1) -> List[int]:
        if not self.vocab:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        chunks = self.pattern.findall(text)
        
        if not use_sampling:
            all_token_ids = []
            for chunk in chunks:
                tokens = self._viterbi_decode(chunk, self.vocab)
                for token in tokens:
                    if token in self.token_to_id:
                        all_token_ids.append(self.token_to_id[token])
            return all_token_ids
        else:
            if num_samples == 1:
                all_token_ids = []
                for chunk in chunks:
                    tokens = self._sample_segmentation(chunk, self.vocab)
                    for token in tokens:
                        if token in self.token_to_id:
                            all_token_ids.append(self.token_to_id[token])
                return all_token_ids
            else:
                samples = []
                for _ in range(num_samples):
                    all_token_ids = []
                    for chunk in chunks:
                        tokens = self._sample_segmentation(chunk, self.vocab)
                        for token in tokens:
                            if token in self.token_to_id:
                                all_token_ids.append(self.token_to_id[token])
                    samples.append(all_token_ids)
                return samples
    
    def decode(self, token_ids: List[int]) -> str:
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append('<unk>')
        return ''.join(tokens)
    
    def tokenize(self, text: str, use_sampling: bool = False) -> List[str]:
        chunks = self.pattern.findall(text)
        all_tokens = []
        
        for chunk in chunks:
            if use_sampling:
                tokens = self._sample_segmentation(chunk, self.vocab)
            else:
                tokens = self._viterbi_decode(chunk, self.vocab)
            all_tokens.extend(tokens)
        
        return all_tokens
    
    def get_vocab(self) -> List[str]:
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)
        return [token for token, _ in sorted_vocab]
