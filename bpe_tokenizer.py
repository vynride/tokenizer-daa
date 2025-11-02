import re
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import pickle


class BPETokenizer:   
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = {}
        self.inverse_special_tokens = {}

        self.pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)|"""
            r"""\w+|"""
            r"""[^\s\w]+|"""
            r"""\s+(?!\S)|"""
            r"""\s+""",
            re.IGNORECASE | re.UNICODE
        )
        
        self.simple_pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)|"""
            r"""\w+|"""
            r"""[^\s\w]+|"""
            r"""\s+""",
            re.IGNORECASE | re.UNICODE
        )
        
        self.compiled_pattern = self.simple_pattern
        
    def _get_stats(self, token_ids: List[int]) -> Dict[Tuple[int, int], int]:
        counts = defaultdict(int)
        for pair in zip(token_ids[:-1], token_ids[1:]):
            counts[pair] += 1
        return counts
    
    def _merge(self, token_ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        merged = []
        i = 0
        while i < len(token_ids):
            if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(token_ids[i])
                i += 1
        return merged
    
    def train(self, text: str, verbose: bool = False) -> None:
        chunks = self.compiled_pattern.findall(text)
        
        tokens = []
        for chunk in chunks:
            chunk_bytes = chunk.encode('utf-8')
            tokens.extend(list(chunk_bytes))
        
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.inverse_vocab = {bytes([i]): i for i in range(256)}
        
        token_ids = tokens.copy()
        
        num_merges = self.vocab_size - 256
        next_id = 256
        
        for i in range(num_merges):
            stats = self._get_stats(token_ids)
            
            if not stats:
                break
            
            max_pair = max(stats, key=stats.get)
            
            if verbose and i % 100 == 0:
                print(f"Merge {i}/{num_merges}: {max_pair} -> {next_id} (count: {stats[max_pair]})")
            
            self.merges[max_pair] = next_id
            
            token_a = self.vocab[max_pair[0]]
            token_b = self.vocab[max_pair[1]]
            new_token = token_a + token_b
            self.vocab[next_id] = new_token
            self.inverse_vocab[new_token] = next_id
            
            token_ids = self._merge(token_ids, max_pair, next_id)
            
            next_id += 1
        
        if verbose:
            print(f"Training complete. Final vocab size: {len(self.vocab)}")
    
    def _encode_chunk(self, chunk_bytes: bytes) -> List[int]:
        token_ids = list(chunk_bytes)
        
        while len(token_ids) >= 2:
            stats = self._get_stats(token_ids)
            
            pair_to_merge = None
            min_merge_idx = float('inf')
            
            for pair in stats:
                if pair in self.merges:
                    merge_idx = list(self.merges.keys()).index(pair)
                    if merge_idx < min_merge_idx:
                        min_merge_idx = merge_idx
                        pair_to_merge = pair
            
            if pair_to_merge is None:
                break
            
            new_id = self.merges[pair_to_merge]
            token_ids = self._merge(token_ids, pair_to_merge, new_id)
        
        return token_ids
    
    def encode(self, text: str, allowed_special: Union[str, set] = "none") -> List[int]:
        if allowed_special == "all":
            special_set = set(self.special_tokens.keys())
        elif allowed_special == "none":
            special_set = set()
        else:
            special_set = allowed_special
        
        if special_set:
            pattern = '|'.join(re.escape(token) for token in special_set)
            parts = re.split(f'({pattern})', text)
        else:
            parts = [text]
        
        token_ids = []
        
        for part in parts:
            if not part:
                continue
            
            if part in special_set and part in self.special_tokens:
                token_ids.append(self.special_tokens[part])
            else:
                chunks = self.compiled_pattern.findall(part)
                
                for chunk in chunks:
                    chunk_bytes = chunk.encode('utf-8')
                    chunk_ids = self._encode_chunk(chunk_bytes)
                    token_ids.extend(chunk_ids)
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        byte_chunks = []
        
        for token_id in token_ids:
            if token_id in self.inverse_special_tokens:
                if byte_chunks:
                    text = b''.join(byte_chunks).decode('utf-8', errors='replace')
                    byte_chunks = []
                else:
                    text = ''
                return text + self.inverse_special_tokens[token_id]
            
            if token_id in self.vocab:
                byte_chunks.append(self.vocab[token_id])
            else:
                byte_chunks.append(b'?')
        
        return b''.join(byte_chunks).decode('utf-8', errors='replace')
    
    def add_special_tokens(self, special_tokens: List[str]) -> None:
        next_id = len(self.vocab)
        
        for token in special_tokens:
            if token not in self.special_tokens:
                self.special_tokens[token] = next_id
                self.inverse_special_tokens[next_id] = token
                next_id += 1
    
    def save(self, filepath: str) -> None:
        data = {
            'vocab_size': self.vocab_size,
            'merges': self.merges,
            'vocab': {k: v for k, v in self.vocab.items()},
            'special_tokens': self.special_tokens,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.vocab_size = data['vocab_size']
        self.merges = data['merges']
        self.vocab = data['vocab']
        self.special_tokens = data['special_tokens']
        
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
    
    def get_vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)


class ChatTemplateTokenizer(BPETokenizer):
    def __init__(self, vocab_size: int = 10000):
        super().__init__(vocab_size)
        
        self.chat_templates = {
            'chatml': {
                'system_prefix': '<|im_start|>system\n',
                'system_suffix': '<|im_end|>\n',
                'user_prefix': '<|im_start|>user\n',
                'user_suffix': '<|im_end|>\n',
                'assistant_prefix': '<|im_start|>assistant\n',
                'assistant_suffix': '<|im_end|>\n',
                'special_tokens': ['<|im_start|>', '<|im_end|>', '<|endoftext|>']
            },
            'llama2': {
                'system_prefix': '[INST] <<SYS>>\n',
                'system_suffix': '\n<</SYS>>\n\n',
                'user_prefix': '',
                'user_suffix': ' [/INST] ',
                'assistant_prefix': '',
                'assistant_suffix': ' </s><s>[INST] ',
                'special_tokens': ['<s>', '</s>', '[INST]', '[/INST]', '<<SYS>>', '<</SYS>>']
            },
            'alpaca': {
                'system_prefix': '',
                'system_suffix': '\n\n',
                'user_prefix': '### Instruction:\n',
                'user_suffix': '\n\n',
                'assistant_prefix': '### Response:\n',
                'assistant_suffix': '\n\n',
                'special_tokens': ['<|endoftext|>']
            }
        }
        
        self.current_template = 'chatml'
    
    def set_chat_template(self, template_name: str) -> None:
        if template_name not in self.chat_templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        self.current_template = template_name
        
        template = self.chat_templates[template_name]
        if 'special_tokens' in template:
            self.add_special_tokens(template['special_tokens'])
    
    def add_custom_template(self, name: str, template: Dict[str, str]) -> None:
        required_keys = ['system_prefix', 'system_suffix', 'user_prefix', 
                        'user_suffix', 'assistant_prefix', 'assistant_suffix']
        
        for key in required_keys:
            if key not in template:
                raise ValueError(f"Template missing required key: {key}")
        
        self.chat_templates[name] = template
        
        if 'special_tokens' in template:
            self.add_special_tokens(template['special_tokens'])
    
    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False
    ) -> str:
        template = self.chat_templates[self.current_template]
        formatted = ""
        
        for message in messages:
            role = message['role']
            content = message['content']
            
            if role == 'system':
                formatted += template['system_prefix'] + content + template['system_suffix']
            elif role == 'user':
                formatted += template['user_prefix'] + content + template['user_suffix']
            elif role == 'assistant':
                formatted += template['assistant_prefix'] + content + template['assistant_suffix']
        
        if add_generation_prompt:
            formatted += template['assistant_prefix']
        
        return formatted
    
    def encode_chat(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
        allowed_special: str = "all"
    ) -> List[int]:
        formatted_text = self.apply_chat_template(messages, add_generation_prompt)
        return self.encode(formatted_text, allowed_special=allowed_special)


class RegexTokenizer:
    
    GPT2_PATTERN = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+""",
        re.UNICODE
    )
    
    GPT4_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\w]?\w+|[0-9]{1,3}| ?[^\s\w]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
        re.UNICODE
    )
    
    SIMPLE_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)|\w+|[^\s\w]+|\s+""",
        re.UNICODE
    )
    
    @staticmethod
    def tokenize(text: str, pattern: Optional[re.Pattern] = None) -> List[str]:
        if pattern is None:
            pattern = RegexTokenizer.SIMPLE_PATTERN
        
        return pattern.findall(text)


def demo():
    print("=" * 60)
    print("BPE Tokenizer Demo")
    print("=" * 60)
    
    training_text = """
    Hello, world! This is a test of the byte pair encoding algorithm.
    The algorithm learns to merge frequent byte pairs.
    It's commonly used in modern language models like GPT.
    Let's see how it works with various text examples.
    The quick brown fox jumps over the lazy dog.
    Machine learning and artificial intelligence are fascinating fields.
    """ * 10
    
    print("\n1. Training BPE Tokenizer...")
    tokenizer = BPETokenizer(vocab_size=500)
    tokenizer.train(training_text, verbose=True)
    
    print("\n2. Testing Encoding/Decoding...")
    test_text = "Hello, world! How are you?"
    print(f"Original text: {test_text}")
    
    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")
    print(f"Number of tokens: {len(encoded)}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    
    print("\n3. Testing Special Tokens...")
    tokenizer.add_special_tokens(['<|endoftext|>', '<|pad|>', '<|unk|>'])
    
    text_with_special = "Hello <|endoftext|> World"
    encoded_special = tokenizer.encode(text_with_special, allowed_special="all")
    print(f"Text with special tokens: {text_with_special}")
    print(f"Encoded: {encoded_special}")
    
    decoded_special = tokenizer.decode(encoded_special)
    print(f"Decoded: {decoded_special}")
    
    print("\n4. Testing Chat Templates...")
    chat_tokenizer = ChatTemplateTokenizer(vocab_size=500)
    chat_tokenizer.train(training_text, verbose=False)
    chat_tokenizer.set_chat_template('chatml')
    
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Hello! How are you?'},
        {'role': 'assistant', 'content': 'I am doing well, thank you!'},
        {'role': 'user', 'content': 'What can you help me with?'}
    ]
    
    formatted_chat = chat_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    print(f"Formatted chat:\n{formatted_chat}")
    
    chat_tokens = chat_tokenizer.encode_chat(messages, add_generation_prompt=True)
    print(f"\nChat tokens (first 20): {chat_tokens[:20]}")
    print(f"Total chat tokens: {len(chat_tokens)}")
    
    print("\n5. Testing Save/Load...")
    tokenizer.save('/tmp/bpe_tokenizer.pkl')
    print("Tokenizer saved to /tmp/bpe_tokenizer.pkl")
    
    new_tokenizer = BPETokenizer()
    new_tokenizer.load('/tmp/bpe_tokenizer.pkl')
    print("Tokenizer loaded successfully")
    
    test_encode = new_tokenizer.encode("Testing loaded tokenizer")
    print(f"Test encoding: {test_encode}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
