import re
import math
from typing import Dict, List
from collections import defaultdict, Counter


class UnigramTokenizer:
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {}
        
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
    
    def _get_all_subwords(self, text: str, max_length: int = 10) -> Counter:
        subwords = Counter()
        chunks = self.pattern.findall(text)
        
        for chunk in chunks:
            chunk_len = len(chunk)
            for i in range(chunk_len):
                for j in range(i + 1, min(i + max_length + 1, chunk_len + 1)):
                    subword = chunk[i:j]
                    subwords[subword] += 1
                    
        return subwords
    
    def _e_step(self, text: str, vocab: Dict[str, float]) -> Dict[str, float]:
        expected_counts = defaultdict(float)
        chunks = self.pattern.findall(text)
        
        for chunk in chunks:
            chunk_len = len(chunk)
            
            forward = [float('-inf')] * (chunk_len + 1)
            forward[0] = 0.0
            best_token = [None] * (chunk_len + 1)
            
            for i in range(chunk_len):
                if forward[i] == float('-inf'):
                    continue
                    
                for j in range(i + 1, chunk_len + 1):
                    token = chunk[i:j]
                    if token in vocab:
                        score = forward[i] + vocab[token]
                        if score > forward[j]:
                            forward[j] = score
                            best_token[j] = (i, token)
            
            if forward[chunk_len] != float('-inf'):
                pos = chunk_len
                while pos > 0 and best_token[pos] is not None:
                    start, token = best_token[pos]
                    expected_counts[token] += 1
                    pos = start
        
        return expected_counts
    
    def _m_step(self, expected_counts: Dict[str, float], total_count: float) -> Dict[str, float]:
        vocab = {}
        for token, count in expected_counts.items():
            if count > 0:
                vocab[token] = math.log(count / total_count)
            else:
                vocab[token] = float('-inf')
        return vocab
    
    def _prune_vocab(self, vocab: Dict[str, float], target_size: int, 
                     protected_tokens: set = None) -> Dict[str, float]:
        if protected_tokens is None:
            protected_tokens = set()
            
        protected = {k: v for k, v in vocab.items() if k in protected_tokens}
        regular = {k: v for k, v in vocab.items() if k not in protected_tokens}
        
        sorted_regular = sorted(regular.items(), key=lambda x: x[1], reverse=True)
        keep_count = target_size - len(protected)
        
        if keep_count > 0:
            kept_regular = dict(sorted_regular[:keep_count])
        else:
            kept_regular = {}
            
        pruned = {**protected, **kept_regular}
        
        if pruned:
            total_prob = sum(math.exp(p) for p in pruned.values())
            log_total = math.log(total_prob) if total_prob > 0 else 0
            pruned = {k: v - log_total for k, v in pruned.items()}
            
        return pruned
    
    def train(self, text: str, num_iterations: int = 10, verbose: bool = False):
        if verbose:
            print("Initializing vocabulary from characters...")
            
        vocab = self._get_initial_vocab(text)
        
        if verbose:
            print("Extracting subword candidates...")
            
        subwords = self._get_all_subwords(text, max_length=10)
        
        initial_vocab_size = min(len(subwords), self.vocab_size * 3)
        most_frequent = subwords.most_common(initial_vocab_size)
        
        total_freq = sum(subwords.values())
        for token, count in most_frequent:
            if token not in vocab:
                vocab[token] = math.log(count / total_freq)
        
        protected = {token for token in vocab if len(token) == 1}
        
        if verbose:
            print(f"Initial vocabulary size: {len(vocab)}")
        
        for iteration in range(num_iterations):
            if verbose:
                print(f"\nIteration {iteration + 1}/{num_iterations}")
                
            expected_counts = self._e_step(text, vocab)
            total_count = sum(expected_counts.values())
            
            if verbose:
                print(f"  Total expected count: {total_count:.2f}")
            
            vocab = self._m_step(expected_counts, total_count)
            
            target_size = max(
                self.vocab_size,
                int(len(vocab) * (1 - 0.1 * (iteration + 1) / num_iterations))
            )
            vocab = self._prune_vocab(vocab, target_size, protected)
            
            if verbose:
                print(f"  Vocabulary size after pruning: {len(vocab)}")
        
        self.vocab = self._prune_vocab(vocab, self.vocab_size, protected)
        
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)
        self.token_to_id = {token: idx for idx, (token, _) in enumerate(sorted_vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        if verbose:
            print(f"\nTraining complete. Final vocabulary size: {len(self.vocab)}")
            print(f"Top 10 tokens by probability:")
            for i, (token, log_prob) in enumerate(sorted_vocab[:10]):
                print(f"  {i+1}. '{token}' (prob: {math.exp(log_prob):.6f})")
    
    def _viterbi_decode(self, text: str) -> List[str]:
        text_len = len(text)
        
        best_score = [float('-inf')] * (text_len + 1)
        best_score[0] = 0.0
        best_token_end = [None] * (text_len + 1)
        
        for i in range(text_len):
            if best_score[i] == float('-inf'):
                continue
                
            for j in range(i + 1, text_len + 1):
                token = text[i:j]
                if token in self.vocab:
                    score = best_score[i] + self.vocab[token]
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
                char = text[i]
                if char in self.vocab:
                    tokens.append(char)
                else:
                    tokens.append(char)
        
        tokens.reverse()
        return tokens
    
    def encode(self, text: str) -> List[int]:
        if not self.vocab:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        chunks = self.pattern.findall(text)
        
        all_token_ids = []
        for chunk in chunks:
            tokens = self._viterbi_decode(chunk)
            
            for token in tokens:
                if token in self.token_to_id:
                    all_token_ids.append(self.token_to_id[token])
                else:
                    for char in token:
                        if char in self.token_to_id:
                            all_token_ids.append(self.token_to_id[char])
        
        return all_token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append('<unk>')
        
        return ''.join(tokens)
    
    def tokenize(self, text: str) -> List[str]:
        chunks = self.pattern.findall(text)
        
        all_tokens = []
        for chunk in chunks:
            tokens = self._viterbi_decode(chunk)
            all_tokens.extend(tokens)
        
        return all_tokens
    
    def get_vocab(self) -> List[str]:
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)
        return [token for token, _ in sorted_vocab]
    
    def add_special_tokens(self, special_tokens: List[str]) -> None:
        max_log_prob = max(self.vocab.values()) if self.vocab else 0
        
        for token in special_tokens:
            if token not in self.vocab:
                self.vocab[token] = max_log_prob
                token_id = len(self.token_to_id)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
                self.special_tokens[token] = token_id
