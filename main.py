#!/usr/bin/env python3
"""Main entry point for the Finnegans Wake translator."""

import os
import sys
from trainer import QwenFinnegansTrainer

def main():
    """Main execution flow."""
    print("=== Finnegans Wake Style Translator ===")
    
    # Check if we should skip training (for development)
    skip_training = os.environ.get("SKIP_TRAINING", "false").lower() == "true"
    
    if not skip_training:
        # Check if Finnegans Wake text exists
        if not os.path.exists("Finnegans_Wake.txt"):
            print("Error: Finnegans_Wake.txt not found in repository!")
            sys.exit(1)
        
        # Check if model already exists
        if not os.path.exists("./qwen-finnegans-model"):
            print("Starting model training...")
            trainer = QwenFinnegansTrainer()
            trainer.run_training("Finnegans_Wake.txt")
            print("Training completed!")
        else:
            print("Model already exists, skipping training.")
    else:
        print("Skipping training (SKIP_TRAINING=true)")
    
    # Start the API server
    print("Starting API server...")
    os.system("python api.py")

if __name__ == "__main__":
    main()It 