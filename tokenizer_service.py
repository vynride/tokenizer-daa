import re
from typing import Dict, List, Any
from bpe_service import BPEService
from wordpiece import WordPieceTokenizer
from subword_regularization_tokenizer import SubwordRegularizationTokenizer


class TokenizerService:
    
    TOKENIZER_TYPES = {
        'bpe': 'Byte Pair Encoding',
        'wordpiece': 'Wordpiece Language Model',
        'subword_regularization': 'Subword Regularization'
    }
    
    def __init__(self):
        self.bpe_service = BPEService()
        self.wordpiece_tokenizer = None
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
        
        # route to the selected tokenizer implementation
        if tokenizer_type == 'bpe':
            return self._encode_with_bpe(
                processed_text, max_merges, vocab_size, show_word_end
            )
        elif tokenizer_type == 'wordpiece':
            return self._encode_with_wordpiece(
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
    def _encode_with_wordpiece(
        self,
        text: str,
        vocab_size: int
    ) -> Dict[str, Any]:
        # train a fresh WordPiece tokenizer on the provided text and encode
        self.wordpiece_tokenizer = WordPieceTokenizer(vocab_size=vocab_size)
        self.wordpiece_tokenizer.train(text, num_iterations=10, verbose=False)

        tokens = self.wordpiece_tokenizer.tokenize(text)
        token_ids = self.wordpiece_tokenizer.encode(text)
        vocab = self.wordpiece_tokenizer.get_vocab()

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
            'tokenizer_type': 'wordpiece',
            'tokenizer_name': 'WordPiece'
        }
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
            
        elif tokenizer_type == 'wordpiece':
            tokenizer = WordPieceTokenizer(vocab_size=vocab_size)
            tokenizer.train(text, num_iterations=10, verbose=False)

            return {
                'vocab_size': len(tokenizer.vocab),
                'num_merges': 0,
                'message': 'WordPiece tokenizer trained successfully',
                'tokenizer_type': 'wordpiece'
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
        elif tokenizer_type == 'wordpiece':
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
