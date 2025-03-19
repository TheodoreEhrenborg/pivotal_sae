#!/usr/bin/env python3

import json
import socketserver
import time
from pathlib import Path
from typing import Any

import torch
from beartype import beartype
from sae_lens import SAE, HookedSAETransformer
from transformers import AutoTokenizer, GemmaTokenizerFast

from sae_hacking.timeprint import timeprint


@beartype
def highlight_tokens_with_intensity(
    split_text: list[str], activations: torch.Tensor
) -> str:
    html_parts = []

    for token, activation in zip(split_text, activations, strict=True):
        activation = min(activation, 30)

        red = int(255 - (activation * 8))
        green = 255
        blue = int(255 - (activation * 8))

        color = f"#{red:02x}{green:02x}{blue:02x}"

        highlighted = f'<span style="background-color: {color};">{token}</span>'
        html_parts.append(highlighted)

    return "".join(html_parts)


@beartype
def create_html(
    split_text: list[str],
    activations: torch.Tensor,
    sae_id: str,
    sae_release: str,
    prompt: str,
    feature_idx: int,
) -> str:
    html_output = highlight_tokens_with_intensity(split_text, activations)

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Highlighted Text Example</title>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                font-size: 16px;
                line-height: 1.5;
                margin: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>Green Intensity Highlighting Example</h1>
        <p>{html_output}</p>
        <hr>
        <p>{sae_id=}</p>
        <hr>
        <p>{sae_release=}</p>
        <hr>
        <p>{prompt}</p>
        <hr>
        <p>{feature_idx}</p>
        <hr>
        <p>Activations: {list(zip(split_text, activations.tolist(), strict=True))}</p>
    </body>
    </html>
    """

    return full_html


@beartype
def get_feature_activation_per_token(
    model: HookedSAETransformer, sae: SAE, feature_idx: int, prompt: str
) -> torch.Tensor:
    """
    Returns an array showing how much a specific SAE feature activated on each token of the prompt.

    Args:
        model: The transformer model with SAE hooks
        sae: The SAE to analyze
        feature_idx: Index of the specific feature to track
        prompt: The input text prompt

    Returns:
        Tensor of shape [num_tokens] containing activation values for the specified feature
        across all tokens in the prompt
    """
    # Ensure the SAE uses its error term for accurate activation measurement
    sae.use_error_term = True

    # Reset the model and SAEs to ensure clean state
    model.reset_hooks()
    model.reset_saes()

    # Run the model with the SAE to get activations
    _, cache = model.run_with_cache_with_saes(prompt, saes=[sae])

    # Get the SAE activations from the cache
    # Shape: [batch_size, sequence_length, n_features]
    sae_acts = cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"]

    # Extract activations for the specified feature across all tokens
    # Assuming batch_size is 1, we take the first batch with sae_acts[0]
    feature_acts = sae_acts[0, :, feature_idx].cpu()

    model.reset_hooks()
    model.reset_saes()

    del _, cache, sae_acts, sae

    return feature_acts


@torch.inference_mode()
@beartype
def process_client_request(
    request_data: dict[str, Any],
    model: HookedSAETransformer,
    tokenizer: GemmaTokenizerFast,
) -> dict[str, Any]:
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary())
    try:
        sae_id = request_data["sae_id"]
        sae_release = request_data["sae_release"]
        prompt = request_data["prompt"]
        feature_idx = request_data["feature_idx"]
        output_dir = Path(request_data["output_dir"])
        device = request_data.get("device", "cuda")

        timeprint(f"Processing request for SAE {sae_id} with feature {feature_idx}")

        sae, _, _ = SAE.from_pretrained(
            release=sae_release, sae_id=sae_id, device=device
        )
        timeprint("Loaded SAE")

        activations = get_feature_activation_per_token(model, sae, feature_idx, prompt)
        timeprint("Got activations")

        split_text = tokenizer.tokenize(prompt, add_special_tokens=True)

        html_output = create_html(
            split_text, activations, sae_id, sae_release, prompt, feature_idx
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_prompt.html"
        with open(output_file, "w") as f:
            f.write(html_output)

        timeprint(f"Saved results to {output_file}")

        return {"status": "success", "message": f"Results saved to {output_file}"}

    except Exception as e:
        timeprint(f"Error processing request: {e}")
        return {"status": "error", "message": str(e)}


class TCPHandler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            # Get data size first (4 bytes indicating message size)
            size_bytes = self.request.recv(4)
            if not size_bytes:
                return

            size = int.from_bytes(size_bytes, byteorder="big")

            # Receive data
            data = b""
            while len(data) < size:
                chunk = self.request.recv(min(4096, size - len(data)))
                if not chunk:
                    break
                data += chunk

            if data:
                request_data = json.loads(data.decode("utf-8"))
                timeprint(f"Received request from {self.client_address[0]}")

                # Process the request
                model = self.server.model
                tokenizer = self.server.tokenizer
                result = process_client_request(request_data, model, tokenizer)

                # Send response back
                response_data = json.dumps(result).encode("utf-8")
                size_bytes = len(response_data).to_bytes(4, byteorder="big")
                self.request.sendall(size_bytes + response_data)

        except Exception as e:
            timeprint(f"Error handling client request: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9000)

    args = parser.parse_args()

    timeprint(f"Starting server on {args.host}:{args.port}")
    timeprint(f"Loading model {args.model} on {args.device}")

    # Load model and tokenizer once
    model = HookedSAETransformer.from_pretrained(args.model, device=args.device)
    timeprint("Loaded model")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    timeprint("Loaded tokenizer")

    # Create server
    server = socketserver.ThreadingTCPServer((args.host, args.port), TCPHandler)

    # Store model and tokenizer in the server instance
    server.model = model
    server.tokenizer = tokenizer

    timeprint("Server ready to accept connections")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        timeprint("Shutting down server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
