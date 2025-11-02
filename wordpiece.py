import re
import math
from typing import Dict, List
from collections import defaultdict, Counter


class WordPieceTokenizer:
    
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
        
    def _get_initial_vocab(self, text: str) -> Dict[str, int]:
        vocab = {}
        
        chars = set()
        for chunk in self.pattern.findall(text):
            chars.update(list(chunk))
        
        for char in chars:
            vocab[char] = 0
            
        return vocab
    
    def _get_pair_frequencies(self, words: Dict[str, int]) -> Counter:
        pairs = Counter()
        
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
                
        return pairs
    
    def _merge_pair(self, pair: tuple, words: Dict[str, int]) -> Dict[str, int]:
        new_words = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in words.items():
            new_word = word.replace(bigram, replacement)
            new_words[new_word] = freq
            
        return new_words
    
    def _get_word_frequencies(self, text: str) -> Dict[str, int]:
        words = defaultdict(int)
        chunks = self.pattern.findall(text)
        
        for chunk in chunks:
            word = ' '.join(list(chunk))
            words[word] += 1
            
        return words
    
    def train(self, text: str, num_iterations: int = 10, verbose: bool = False):
        if verbose:
            print("Initializing vocabulary from characters...")
            
        vocab = self._get_initial_vocab(text)
        
        if verbose:
            print("Extracting word frequencies...")
            
        words = self._get_word_frequencies(text)
        
        if verbose:
            print(f"Initial vocabulary size: {len(vocab)}")
        
        num_merges = self.vocab_size - len(vocab)
        
        for iteration in range(min(num_merges, num_iterations * 100)):
            if verbose and iteration % 100 == 0:
                print(f"\nIteration {iteration + 1}")
                
            pairs = self._get_pair_frequencies(words)
            
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            
            if verbose and iteration % 100 == 0:
                print(f"  Merging: {best_pair} (frequency: {pairs[best_pair]})")
            
            words = self._merge_pair(best_pair, words)
            
            new_token = ''.join(best_pair)
            vocab[new_token] = iteration + 1
            
            if len(vocab) >= self.vocab_size:
                break
            
            if verbose and iteration % 100 == 0:
                print(f"  Vocabulary size: {len(vocab)}")
        
        self.vocab = vocab
        
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: (x[1], x[0]))
        self.token_to_id = {token: idx for idx, (token, _) in enumerate(sorted_vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        if verbose:
            print(f"\nTraining complete. Final vocabulary size: {len(self.vocab)}")
            print(f"Top 10 tokens by merge order:")
            for i, (token, order) in enumerate(sorted_vocab[-10:]):
                print(f"  {i+1}. '{token}' (order: {order})")
    
    def _greedy_decode(self, text: str) -> List[str]:
        tokens = []
        
        while text:
            found = False
            for i in range(len(text), 0, -1):
                substr = text[:i]
                if substr in self.vocab:
                    tokens.append(substr)
                    text = text[i:]
                    found = True
                    break
            
            if not found:
                if text[0] in self.vocab:
                    tokens.append(text[0])
                else:
                    tokens.append(text[0])
                text = text[1:]
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        if not self.vocab:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        chunks = self.pattern.findall(text)
        
        all_token_ids = []
        for chunk in chunks:
            tokens = self._greedy_decode(chunk)
            
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
            tokens = self._greedy_decode(chunk)
            all_tokens.extend(tokens)
        
        return all_tokens
    
    def get_vocab(self) -> List[str]:
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: (x[1], x[0]))
        return [token for token, _ in sorted_vocab]
    
    def add_special_tokens(self, special_tokens: List[str]) -> None:
        max_order = max(self.vocab.values()) if self.vocab else 0
        
        for token in special_tokens:
            if token not in self.vocab:
                self.vocab[token] = max_order
                token_id = len(self.token_to_id)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
                self.special_tokens[token] = token_id