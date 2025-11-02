from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

from tokenizer_service import TokenizerService

app = Flask(__name__,
            static_folder='static',
            static_url_path='')

CORS(app)

tokenizer_service = TokenizerService()

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'Multi-Tokenizer API',
        'version': '1.0.0',
        'tokenizers': list(tokenizer_service.get_available_tokenizers().keys())
    })

@app.route('/api/tokenizers')
def get_tokenizers():
    tokenizers = []
    for key, name in tokenizer_service.get_available_tokenizers().items():
        info = tokenizer_service.get_tokenizer_info(key)
        tokenizers.append({
            'type': key,
            'name': name,
            'description': info.get('description', ''),
            'features': info.get('features', [])
        })
    
    return jsonify({
        'success': True,
        'tokenizers': tokenizers
    })

@app.route('/api/encode', methods=['POST'])
def encode():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: text'
            }), 400
        
        text = data['text']
        tokenizer_type = data.get('tokenizer_type', 'bpe')
        max_merges = data.get('max_merges', 50)
        vocab_size = data.get('vocab_size', 500)
        lowercase = data.get('lowercase', False)
        show_word_end = data.get('show_word_end', True)
        use_sampling = data.get('use_sampling', False)
        
        if not isinstance(text, str):
            return jsonify({
                'success': False,
                'error': 'Text must be a string'
            }), 400
        
        if tokenizer_type not in tokenizer_service.get_available_tokenizers():
            return jsonify({
                'success': False,
                'error': f'Invalid tokenizer type. Available: {list(tokenizer_service.get_available_tokenizers().keys())}'
            }), 400
        
        if len(text) == 0:
            return jsonify({
                'success': True,
                'tokens': [],
                'token_ids': [],
                'vocab': [],
                'merges': [],
                'tokenizer_type': tokenizer_type,
                'tokenizer_name': tokenizer_service.get_available_tokenizers()[tokenizer_type],
                'stats': {
                    'token_count': 0,
                    'vocab_size': 0,
                    'unique_tokens': 0,
                    'compression_ratio': 0
                }
            })
        
        result = tokenizer_service.encode_text(
            text=text,
            tokenizer_type=tokenizer_type,
            max_merges=max_merges,
            vocab_size=vocab_size,
            lowercase=lowercase,
            show_word_end=show_word_end,
            use_sampling=use_sampling
        )
        
        return jsonify({
            'success': True,
            **result
        })
    
    except Exception as e:
        app.logger.error(f"Error in /api/encode: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/decode', methods=['POST'])
def decode():
    try:
        data = request.get_json()
        
        if not data or 'token_ids' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: token_ids'
            }), 400
        
        token_ids = data['token_ids']
        vocab_mapping = data.get('vocab_mapping', {})
        tokenizer_type = data.get('tokenizer_type', 'bpe')
        
        result = tokenizer_service.decode_tokens(token_ids, vocab_mapping, tokenizer_type)
        
        return jsonify({
            'success': True,
            'text': result
        })
    
    except Exception as e:
        app.logger.error(f"Error in /api/decode: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/train', methods=['POST'])
def train():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: text'
            }), 400
        
        text = data['text']
        tokenizer_type = data.get('tokenizer_type', 'bpe')
        vocab_size = data.get('vocab_size', 500)
        
        if tokenizer_type not in tokenizer_service.get_available_tokenizers():
            return jsonify({
                'success': False,
                'error': f'Invalid tokenizer type. Available: {list(tokenizer_service.get_available_tokenizers().keys())}'
            }), 400
        
        result = tokenizer_service.train_tokenizer(text, tokenizer_type, vocab_size)
        
        return jsonify({
            'success': True,
            **result
        })
    
    except Exception as e:
        app.logger.error(f"Error in /api/train: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/examples')
def examples():
    return jsonify({
        'success': True,
        'examples': [
            {
                'name': 'Simple Greeting',
                'text': 'Hello, world! How are you today?'
            },
            {
                'name': 'Technical Text',
                'text': 'Byte Pair Encoding is a data compression technique used in natural language processing. It iteratively merges the most frequent pair of bytes.'
            },
            {
                'name': 'Code Snippet',
                'text': 'def hello_world():\n    print("Hello, world!")\n    return True'
            },
            {
                'name': 'Repeated Patterns',
                'text': 'The quick brown fox jumps over the lazy dog. The dog was very lazy. The fox was very quick.'
            },
            {
                'name': 'Mixed Content',
                'text': 'AI and ML are transforming software development. GPT-4, BERT, and other models use tokenization. BPE is key! 🚀'
            }
        ]
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print("=" * 60)
    print("🚀 Multi-Tokenizer Flask Backend")
    print("=" * 60)
    print(f"📡 Server running on: http://localhost:{port}")
    print(f"🌐 Frontend available at: http://localhost:{port}/")
    print(f"🔧 API endpoints available at: http://localhost:{port}/api/")
    print(f"📝 Debug mode: {debug}")
    print(f"🔤 Available tokenizers:")
    for key, name in tokenizer_service.get_available_tokenizers().items():
        print(f"   - {name} ({key})")
    print("=" * 60)
    print("\nPress CTRL+C to stop the server\n")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
