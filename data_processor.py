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
        
        # Create training pairs that teach Joyce's style patterns
        for sentence in sentences:
            if len(sentence.strip()) > 20 and len(sentence.strip()) < 200:  # Filter very short/long sentences
                # Create a "normalized" version as input and Joyce's original as output
                normalized = self._normalize_sentence(sentence)
                if normalized != sentence:  # Only use pairs where there's a difference
                    input_text = f"Rewrite in Finnegans Wake style: {normalized}"
                    output_text = sentence.strip()
                    pairs.append((input_text, output_text))
        
        return pairs
    
    def _normalize_sentence(self, sentence: str) -> str:
        """Create a more normalized version of Joyce's text for training pairs."""
        normalized = sentence.strip()
        
        # Reverse some of Joyce's stylistic choices to create input text
        # Fix obvious portmanteaus and unusual spellings
        normalized = re.sub(r'\b(\w+)een\b', r'\1ing', normalized)  # "dayeen" -> "day"
        normalized = re.sub(r'\bth\'', 'the', normalized)  # "th'" -> "the"
        normalized = re.sub(r'\ban\'', 'and', normalized)  # "an'" -> "and"
        normalized = re.sub(r'\byer\b', 'your', normalized)  # "yer" -> "your"
        normalized = re.sub(r'\bmeself\b', 'myself', normalized)  # "meself" -> "myself"
        normalized = re.sub(r'\bsez\b', 'says', normalized)  # "sez" -> "says"
        
        # Clean up excessive punctuation
        normalized = re.sub(r'[!]{2,}', '!', normalized)
        normalized = re.sub(r'[?]{2,}', '?', normalized)
        
        # Remove some of Joyce's invented words (keep it simple)
        normalized = re.sub(r'\b\w*anna\w*\b', 'and', normalized)  # Various "anna" constructions
        
        return normalized
    
    def _split_into_sentences(self) -> List[str]:
        """Split text into sentences, handling Joyce's unique punctuation."""
        # Joyce uses unconventional punctuation, so we'll be more flexible
        sentences = re.split(r'[.!?]+\s+', self.processed_text)
        return [s.strip() for s in sentences if s.strip()]