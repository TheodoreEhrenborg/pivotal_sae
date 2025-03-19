#!/usr/bin/env python3

import json
import socket
from argparse import ArgumentParser
from ast import literal_eval
from pathlib import Path


def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--sae-release", required=True)
    parser.add_argument("--sae-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-idx", required=True, type=int)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--server-host", default="localhost")
    parser.add_argument("--server-port", type=int, default=9000)
    return parser


def maybe_get(old_value, name):
    strng = input(f"Enter new value for {name} (currently {old_value}): ")
    if strng == "":
        return old_value
    return literal_eval(strng)


def send_request_to_server(request_data, host, port):
    try:
        # Connect to server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))

            # Serialize the request data
            serialized_data = json.dumps(request_data).encode("utf-8")

            # Send size first, then data
            size_bytes = len(serialized_data).to_bytes(4, byteorder="big")
            s.sendall(size_bytes + serialized_data)

            # Get response size
            size_bytes = s.recv(4)
            if not size_bytes:
                return {"status": "error", "message": "No response from server"}

            size = int.from_bytes(size_bytes, byteorder="big")

            # Receive response
            data = b""
            while len(data) < size:
                chunk = s.recv(min(4096, size - len(data)))
                if not chunk:
                    break
                data += chunk

            # Parse and return response
            return json.loads(data.decode("utf-8"))

    except ConnectionRefusedError:
        return {
            "status": "error",
            "message": f"Connection to server at {host}:{port} refused. Is the server running?",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error communicating with server: {str(e)}",
        }


def main():
    args = make_parser().parse_args()

    sae_id = args.sae_id
    sae_release = args.sae_release
    prompt = args.prompt
    feature_idx = args.feature_idx
    output_dir = args.output_dir
    device = args.device

    print(f"Connecting to server at {args.server_host}:{args.server_port}")

    while True:
        # Prepare request data
        request_data = {
            "sae_id": sae_id,
            "sae_release": sae_release,
            "prompt": prompt,
            "feature_idx": feature_idx,
            "output_dir": str(output_dir),
            "device": device,
        }

        # Send request to server
        print("Sending request to server...")
        response = send_request_to_server(
            request_data, args.server_host, args.server_port
        )

        # Display response
        if response["status"] == "success":
            print(f"Success: {response['message']}")
        else:
            print(f"Error: {response['message']}")

        # Get new parameters for next run
        print("\nEnter new values (or press Enter to keep current value):")
        sae_id = maybe_get(sae_id, "sae_id")
        sae_release = maybe_get(sae_release, "sae_release")
        prompt = maybe_get(prompt, "prompt")
        feature_idx = maybe_get(feature_idx, "feature_idx")

        # Check if user wants to continue
        continue_run = input("\nContinue with another run? (y/n): ")
        if continue_run.lower() != "y":
            break


if __name__ == "__main__":
    main()
