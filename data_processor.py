#!/usr/bin/env python3
"""Process Finnegans Wake text for training."""

import re
from typing import List, Tuple

class FinnegansWakeProcessor:
    def __init__(self, text_file: str):
        self.text_file = text_file
        self.raw_text = self._load_text()
        self.processed_text = self._clean_text()
    
    def _load_text(self) -> str:
        """Load the raw text file."""
        with open(self.text_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _clean_text(self) -> str:
        """Clean and preprocess the text."""
        text = self.raw_text
        
        # Remove Project Gutenberg header/footer
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
        
        start_idx = text.find(start_marker)
        if start_idx != -1:
            text = text[start_idx:]
            text = text[text.find('\n') + 1:]  # Remove the marker line
        
        end_idx = text.find(end_marker)
        if end_idx != -1:
            text = text[:end_idx]
        
        # Basic cleaning
        text = re.sub(r'\r\n', '\n', text)  # Normalize line endings
        text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce excessive newlines
        
        return text.strip()
    
    def create_training_pairs(self, chunk_size: int = 512) -> List[Tuple[str, str]]:
        """Create training pairs for style transfer."""
        sentences = self._split_into_sentences()
        pairs = []
        
        # Create pairs where input is "normal" text and output is Joyce's style
        for sentence in sentences:
            if len(sentence.strip()) > 20:  # Filter very short sentences
                # Simple prompt format for style transfer
                input_text = f"Translate this to Finnegans Wake style: {sentence}"
                output_text = sentence
                pairs.append((input_text, output_text))
        
        return pairs
    
    def _split_into_sentences(self) -> List[str]:
        """Split text into sentences, handling Joyce's unique punctuation."""
        # Joyce uses unconventional punctuation, so we'll be more flexible
        sentences = re.split(r'[.!?]+\s+', self.processed_text)
        return [s.strip() for s in sentences if s.strip()]