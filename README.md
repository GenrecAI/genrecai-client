# GenreCAI Python Client

A Python client library for accessing GenreCAI's powerful LLM and embedding models API.

## Installation

```bash
pip install genrecai
```

## Quick Start

```python
from genrecai import AI

# Chat generation
chat_ai = AI(
    base_url="http://localhost:80/chat",
    model_name="llama-3.3-70b-versatile"
)

# Stream generated text
for chunk in chat_ai.generate("Explain quantum computing"):
    print(chunk, end="", flush=True)
print()  # New line at end

# Generate embeddings
embed_ai = AI(
    base_url="http://localhost:80/embed",
    model_name="text-embedding-004"
)
embeddings = embed_ai.embed("This is a test sentence")
print(f"Generated embedding with {len(embeddings)} dimensions")
```

## Command Line Interface

The package includes a CLI for quick access to API features:

```bash
# Chat generation
genrecai --model llama-3.3-70b-versatile chat "Explain quantum computing"

# Generate embeddings
genrecai --model text-embedding-004 embed "This is a test sentence"

# List available models
genrecai --model any models
```

## API Reference

### Class: AI

#### Constructor

```python
AI(base_url: str, model_name: str)
```

- `base_url`: Full URL including endpoint (e.g., "http://localhost:80/chat" or "http://localhost:80/embed")
- `model_name`: Name of the model to use

#### Methods

##### generate()

```python
generate(prompt: str) -> Iterator[str]
```

Generates text using a chat model. Returns an iterator that yields text chunks.
Only available when using the `/chat` endpoint.

##### embed()

```python
embed(content: Union[str, List[str]], input_type: str = "document") -> List[float]
```

Generates embeddings for the given text(s).
Only available when using the `/embed` endpoint.

- `content`: Single string or list of strings to embed
- `input_type`: Type of input ("document" by default)
- Returns: List of floating point numbers representing the embedding

##### get_available_models()

```python
get_available_models() -> List[str]
```

Returns a list of available models for the current endpoint type.

## Error Handling

The library raises exceptions in these cases:
- Invalid endpoint usage (e.g., calling `generate()` on embed endpoint)
- API request failures (non-200 status codes)
- Invalid responses from the API

Example error handling:

```python
from genrecai import AI

try:
    ai = AI(base_url="http://localhost:80/chat", model_name="llama-3.3-70b-versatile")
    for chunk in ai.generate("Hello!"):
        print(chunk, end="")
except Exception as e:
    print(f"Error: {e}")
```
