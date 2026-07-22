"""
Week 7-8 — Standalone sanity check for the Ollama + Llama 3.2 setup.

Run this BEFORE opening the Streamlit app, especially when demoing —
it's a much faster way to debug "why is my report generation broken"
than clicking through the UI.

Usage:
    python check_ollama.py
    python check_ollama.py --host http://localhost:11434 --model llama3.2
"""

import argparse
import sys

import requests

from llm_report import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    print(f"1. Checking Ollama server at {args.host} ...")
    try:
        resp = requests.get(f"{args.host}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"   FAILED — is `ollama serve` running? Error: {e}")
        sys.exit(1)
    print("   OK — server is reachable.")

    tags = resp.json().get("models", [])
    names = [m.get("name", "") for m in tags]
    print(f"2. Checking model '{args.model}' is pulled ...")
    if not any(args.model in n for n in names):
        print(f"   NOT FOUND. Installed models: {names or '(none)'}")
        print(f"   Run: ollama pull {args.model}")
        sys.exit(1)
    print("   OK — model is available.")

    print("3. Sending a test generation request ...")
    try:
        resp = requests.post(
            f"{args.host}/api/generate",
            json={"model": args.model, "prompt": "Reply with exactly: OK", "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"   FAILED — {e}")
        sys.exit(1)
    text = resp.json().get("response", "").strip()
    print(f"   OK — model responded: {text!r}")

    print("\nAll checks passed. The Streamlit app's LLM path should work.")


if __name__ == "__main__":
    main()
