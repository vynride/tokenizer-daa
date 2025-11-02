import re
from typing import Dict, List, Any
from bpe_service import BPEService
from wordpiece import UnigramTokenizer
from subword_regularization_tokenizer import SubwordRegularizationTokenizer


class TokenizerService:
    
    TOKENIZER_TYPES = {
        'bpe': 'Byte Pair Encoding',
        'wordpiece': 'Wordpiece Language Model',
        'subword_regularization': 'Subword Regularization'
    }
    
    def __init__(self):
        self.bpe_service = BPEService()
        self.unigram_tokenizer = None
        self.subword_tokenizer = None
        self.END_MARKER = '</w>'
        
    def get_available_tokenizers(self) -> Dict[str, str]:
        return self.TOKENIZER_TYPES.copy()
    
    def _preprocess_text(self, text: str, lowercase: bool = False) -> str:
        if lowercase:
            text = text.lower()
        return text
    
    def encode_text(
        self,
        text: str,
        tokenizer_type: str = 'bpe',
        max_merges: int = 50,
        vocab_size: int = 500,
        lowercase: bool = False,
        show_word_end: bool = True,
        use_sampling: bool = False
    ) -> Dict[str, Any]:
        processed_text = self._preprocess_text(text, lowercase)
        
        if tokenizer_type == 'bpe':
            return self._encode_with_bpe(
                processed_text, max_merges, vocab_size, show_word_end
            )
        elif tokenizer_type == 'unigram':
            return self._encode_with_unigram(
                processed_text, vocab_size
            )
        elif tokenizer_type == 'subword_regularization':
            return self._encode_with_subword_reg(
                processed_text, vocab_size, use_sampling
            )
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    def _encode_with_bpe(
        self,
        text: str,
        max_merges: int,
        vocab_size: int,
        show_word_end: bool
    ) -> Dict[str, Any]:
        result = self.bpe_service.encode_text(
            text=text,
            max_merges=max_merges,
            vocab_size=vocab_size,
            lowercase=False,
            show_word_end=show_word_end
        )
        
        result['tokenizer_type'] = 'bpe'
        result['tokenizer_name'] = 'Byte Pair Encoding'
        
        return result
    
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
    def _encode_with_subword_reg(
        self,
        text: str,
        vocab_size: int,
        use_sampling: bool = False
    ) -> Dict[str, Any]:
        self.subword_tokenizer = SubwordRegularizationTokenizer(
            vocab_size=vocab_size,
            alpha=0.2
        )
        self.subword_tokenizer.train(
            text, 
            num_iterations=8, 
            num_samples=5, 
            verbose=False
        )
        
        tokens = self.subword_tokenizer.tokenize(text, use_sampling=use_sampling)
        token_ids = self.subword_tokenizer.encode(text, use_sampling=use_sampling)
        vocab = self.subword_tokenizer.get_vocab()
        
        non_whitespace_tokens = [t for t in tokens if not re.match(r'^\s+$', t)]
        unique_tokens = set(non_whitespace_tokens)
        
        original_chars = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
        compression_ratio = len(non_whitespace_tokens) / original_chars if original_chars > 0 else 0
        
        return {
            'tokens': tokens,
            'token_ids': token_ids,
            'vocab': vocab,
            'merges': [],
            'stats': {
                'token_count': len(non_whitespace_tokens),
                'vocab_size': len(vocab),
                'unique_tokens': len(unique_tokens),
                'compression_ratio': round(compression_ratio, 3),
                'merge_count': 0
            },
            'tokenizer_type': 'subword_regularization',
            'tokenizer_name': 'Subword Regularization',
            'sampling_enabled': use_sampling
        }
    
    def train_tokenizer(
        self,
        text: str,
        tokenizer_type: str = 'bpe',
        vocab_size: int = 500,
        max_merges: int = None
    ) -> Dict[str, Any]:
        if tokenizer_type == 'bpe':
            result = self.bpe_service.train_tokenizer(text, vocab_size)
            result['tokenizer_type'] = 'bpe'
            return result
            
        elif tokenizer_type == 'unigram':
            tokenizer = UnigramTokenizer(vocab_size=vocab_size)
            tokenizer.train(text, num_iterations=10, verbose=False)
            
            return {
                'vocab_size': len(tokenizer.vocab),
                'num_merges': 0,
                'message': 'Unigram tokenizer trained successfully',
                'tokenizer_type': 'unigram'
            }
            
        elif tokenizer_type == 'subword_regularization':
            tokenizer = SubwordRegularizationTokenizer(vocab_size=vocab_size)
            tokenizer.train(text, num_iterations=10, num_samples=5, verbose=False)
            
            return {
                'vocab_size': len(tokenizer.vocab),
                'num_merges': 0,
                'message': 'Subword Regularization tokenizer trained successfully',
                'tokenizer_type': 'subword_regularization'
            }
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    def decode_tokens(
        self,
        token_ids: List[int],
        vocab_mapping: Dict[str, str],
        tokenizer_type: str = 'bpe'
    ) -> str:
        if tokenizer_type == 'bpe':
            return self.bpe_service.decode_tokens(token_ids, vocab_mapping)
        else:
            tokens = []
            for token_id in token_ids:
                token_str = str(token_id)
                if token_str in vocab_mapping:
                    token = vocab_mapping[token_str]
                    tokens.append(token)
            return ''.join(tokens)
    
    def get_tokenizer_info(self, tokenizer_type: str) -> Dict[str, Any]:
        info = {
            'type': tokenizer_type,
            'name': self.TOKENIZER_TYPES.get(tokenizer_type, 'Unknown'),
        }
        
        if tokenizer_type == 'bpe':
            info['description'] = 'Iteratively merges the most frequent pairs of bytes/characters'
            info['features'] = [
                'Deterministic segmentation',
                'Explicit merge operations',
                'Good for compression',
                'Used in GPT models'
            ]
        elif tokenizer_type == 'unigram':
            info['description'] = 'Probabilistic segmentation based on likelihood maximization'
            info['features'] = [
                'Probabilistic segmentation',
                'Likelihood-based vocabulary',
                'Better for multilingual data',
                'Used in SentencePiece'
            ]
        elif tokenizer_type == 'subword_regularization':
            info['description'] = 'Stochastic segmentation for improved generalization'
            info['features'] = [
                'Multiple segmentation samples',
                'Improves model robustness',
                'Better generalization',
                'Handles rare words well'
            ]
        
        return info
