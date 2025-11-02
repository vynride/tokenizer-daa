import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class BPEService:
    def __init__(self):
        self.END_MARKER = '</w>'
        
    def _preprocess_text(self, text: str, lowercase: bool = False) -> str:
        if lowercase:
            text = text.lower()
        return text
    
    def _split_words(self, text: str) -> List[str]:
        parts = re.findall(r'\S+|\s+', text)
        return parts
    
    def _word_to_symbols(self, word: str, show_word_end: bool = True) -> List[str]:
        chars = list(word)
        if show_word_end:
            chars.append(self.END_MARKER)
        return chars
    
    def _get_pair_stats(self, tokenized_words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        stats = defaultdict(int)
        
        for seq in tokenized_words:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                stats[pair] += 1
        
        return stats
    
    def _merge_pair(self, sequence: List[str], pair: Tuple[str, str]) -> List[str]:
        merged = []
        i = 0
        
        while i < len(sequence):
            if i < len(sequence) - 1 and sequence[i] == pair[0] and sequence[i + 1] == pair[1]:
                merged.append(pair[0] + pair[1])
                i += 2
            else:
                merged.append(sequence[i])
                i += 1
        
        return merged
    
    def _apply_merge(self, tokenized_words: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
        return [self._merge_pair(seq, pair) for seq in tokenized_words]
    
    def learn_bpe_merges(
        self,
        words: List[str],
        max_merges: int,
        show_word_end: bool = True
    ) -> Tuple[List[List[str]], List[Tuple[str, str]], List[str]]:
        tokenized = [self._word_to_symbols(w, show_word_end) for w in words]
        merges = []
        
        for _ in range(max_merges):
            stats = self._get_pair_stats(tokenized)
            
            if not stats:
                break
            
            best_pair = max(stats.items(), key=lambda x: x[1])
            pair, count = best_pair
            
            if count < 2:
                break
            
            merges.append(pair)
            
            tokenized = self._apply_merge(tokenized, pair)
        
        vocab_set = set()
        for seq in tokenized:
            vocab_set.update(seq)
        
        vocab = sorted(list(vocab_set))
        
        return tokenized, merges, vocab
    
    def encode_text(
        self,
        text: str,
        max_merges: int = 50,
        vocab_size: int = 500,
        lowercase: bool = False,
        show_word_end: bool = True
    ) -> Dict[str, Any]:
        processed_text = self._preprocess_text(text, lowercase)
        
        parts = self._split_words(processed_text)
        
        words = [p for p in parts if not re.match(r'^\s+$', p)]
        
        tokenized, merges, vocab = self.learn_bpe_merges(
            words,
            max_merges,
            show_word_end
        )
        
        all_tokens = []
        token_ids = []
        word_idx = 0
        
        vocab_to_id = {token: idx for idx, token in enumerate(vocab)}
        
        for part in parts:
            if re.match(r'^\s+$', part):
                all_tokens.append(part)
                token_ids.append(-1)
            else:
                if word_idx < len(tokenized):
                    for token in tokenized[word_idx]:
                        all_tokens.append(token)
                        token_ids.append(vocab_to_id.get(token, -1))
                    word_idx += 1
        
        non_whitespace_tokens = [t for t in all_tokens if not re.match(r'^\s+$', t)]
        unique_tokens = set(non_whitespace_tokens)
        
        original_chars = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
        compression_ratio = len(non_whitespace_tokens) / original_chars if original_chars > 0 else 0
        
        return {
            'tokens': all_tokens,
            'token_ids': token_ids,
            'vocab': vocab,
            'merges': [[pair[0], pair[1]] for pair in merges],
            'stats': {
                'token_count': len(non_whitespace_tokens),
                'vocab_size': len(vocab),
                'unique_tokens': len(unique_tokens),
                'compression_ratio': round(compression_ratio, 3),
                'merge_count': len(merges)
            }
        }
    
    def decode_tokens(
        self,
        token_ids: List[int],
        vocab_mapping: Dict[str, str]
    ) -> str:
        tokens = []
        
        for token_id in token_ids:
            token_str = str(token_id)
            if token_str in vocab_mapping:
                token = vocab_mapping[token_str]
                token = token.replace(self.END_MARKER, ' ')
                tokens.append(token)
        
        return ''.join(tokens).strip()
    
    def train_tokenizer(self, text: str, vocab_size: int = 500) -> Dict[str, Any]:
        from bpe_tokenizer import BPETokenizer
        
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        tokenizer.train(text, verbose=False)
        
        return {
            'vocab_size': tokenizer.get_vocab_size(),
            'num_merges': len(tokenizer.merges),
            'message': 'Tokenizer trained successfully'
        }
