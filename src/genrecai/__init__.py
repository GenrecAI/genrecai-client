import requests
import json
from typing import Union, List

class AI:
    def __init__(self, base_url="http://localhost:80/chat", model_name=None):
        """
        Initialize AI client.
        Args:
            base_url: Full URL including endpoint (e.g., "http://localhost:80/chat" or "http://localhost:80/embed")
            model_name: Name of the model to use
        """
        self.base_url = base_url.rstrip('/')  # Remove trailing slash if present
        self.endpoint = '/chat' if '/chat' in base_url else '/embed'
        self.base_server = self.base_url[:self.base_url.rindex(self.endpoint)]
        
        # Set default model based on endpoint
        assert model_name is not None, "Model name must be provided"
        self.model_name = model_name or ("llama-3.3-70b-versatile" if self.endpoint == '/chat' else "text-embedding-004")

    def generate(self, prompt: str):
        """Generate streaming response using chat model."""
        if self.endpoint != "/chat":
            raise ValueError("generate() can only be used with chat endpoint (/chat)")
            
        response = requests.get(
            f"{self.base_url}/generate",
            params={"model": self.model_name, "prompt": prompt},
            stream=True
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}")
        
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith('data: '):
                data = line[6:]  # Remove 'data: ' prefix
                if data == '[DONE]':
                    break
                try:
                    content = json.loads(data).get('content', '')
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue

    def embed(self, content: Union[str, List[str]], input_type: str = "document"):
        """Generate embeddings using embedding model."""
        if self.endpoint != "/embed":
            raise ValueError("embed() can only be used with embedding endpoint (/embed)")
            
        # For embed endpoint, we don't append anything
        response = requests.post(
            self.base_url,
            json={
                "model": self.model_name,
                "content": content,
                "input_type": input_type
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}")
            
        return response.json()["embeddings"]

    def get_available_models(self):
        """Get list of available models based on endpoint type."""
        # Construct proper URL for models endpoint
        if self.endpoint == "/chat":
            url = f"{self.base_server}/models"
        else:
            url = f"{self.base_server}/embed/models"
            
        response = requests.get(url)
        
        if response.status_code != 200:
            raise Exception(f"Failed to get models. Status: {response.status_code}")
            
        return response.json()

def main():
    """CLI interface for GenreCAI."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="GenreCAI API Client")
    parser.add_argument("--base-url", default="http://localhost:80", help="Base URL for the API")
    parser.add_argument("--model", required=True, help="Model name to use")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Generate text using chat model")
    chat_parser.add_argument("prompt", help="Input prompt for generation")

    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Generate embeddings")
    embed_parser.add_argument("text", help="Text to embed")
    embed_parser.add_argument("--input-type", default="document", help="Input type (default: document)")

    # List models command
    subparsers.add_parser("models", help="List available models")

    args = parser.parse_args()

    if args.command == "chat":
        ai = AI(base_url=f"{args.base_url}/chat", model_name=args.model)
        try:
            for chunk in ai.generate(args.prompt):
                print(chunk, end="", flush=True)
            print()  # New line at end
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "embed":
        ai = AI(base_url=f"{args.base_url}/embed", model_name=args.model)
        try:
            embeddings = ai.embed(args.text, input_type=args.input_type)
            print(f"Generated {len(embeddings)} dimensional embedding")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "models":
        # For models command, we'll create both chat and embed clients to get all models
        try:
            chat_ai = AI(base_url=f"{args.base_url}/chat", model_name=args.model)
            embed_ai = AI(base_url=f"{args.base_url}/embed", model_name=args.model)
            
            print("Available Chat Models:")
            for model in chat_ai.get_available_models():
                print(f"  - {model}")
                
            print("\nAvailable Embedding Models:")
            for model in embed_ai.get_available_models():
                print(f"  - {model}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()