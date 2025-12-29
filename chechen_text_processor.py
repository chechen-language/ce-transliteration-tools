#!/usr/bin/env python3
"""
Enhanced Chechen Text Processor

Advanced text processing module for Chechen language with sophisticated
character normalization, validation, and quality reporting.

Based on mechanisms from chechen-spellchecker/tokenize.py but adapted
for the transliteration-tools workflow.
"""

import re
import logging
from collections import Counter
from typing import Dict, List, Set, Tuple, Optional
# from pathlib import Path  # Reserved for future use


class ChechenTextProcessor:
    """Enhanced Chechen text processor with advanced normalization and validation."""
    
    def __init__(self, enable_logging: bool = False, log_file: Optional[str] = None):
        # Chechen alphabet (including palochka ӏ)
        self.chechen_letters = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюяӏ')
        
        # Valid single-letter words in Chechen
        self.valid_single_letters = {'а', 'и', 'я', 'ю'}
        
        # Legal punctuation and symbols
        self.legal_symbols = {'-', '.', ',', '!', '?', ':', ';', '"', "'", 
                             '(', ')', '[', ']', ' ', '\n', '\t', '\r'}
        
        # Character normalization mapping
        self.char_replacements = {
            # Palochka variants
            'i': 'ӏ', 'I': 'ӏ', '1': 'ӏ', 'ι': 'ӏ', 'і': 'ӏ',
            # Accented characters
            'à': 'а', 'á': 'а', 'ò': 'о', 
            'è': 'е', 'é': 'е',
            'y': 'у'
        }
        
        # Initialize tracking containers
        self.single_letter_words: Dict[str, int] = {}
        self.suspicious_transformations: List[Tuple[str, str, int]] = []
        self.length_changes: List[Tuple[str, str, int]] = []
        self.non_chechen_chars: Dict[str, Dict[str, int]] = {}
        self.blacklisted_words: Set[str] = set()
        
        # Set up logging if enabled
        self.enable_logging = enable_logging
        if enable_logging:
            self._setup_logging(log_file)
    
    def _setup_logging(self, log_file: Optional[str] = None):
        """Set up logging configuration."""
        if log_file is None:
            log_file = 'chechen_processing.log'
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def load_blacklist(self, filepath: str) -> bool:
        """Load blacklisted words from file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.blacklisted_words = {
                    line.strip().lower() for line in f 
                    if line.strip()
                }
            if self.enable_logging:
                logging.info(f"Loaded {len(self.blacklisted_words)} blacklisted words")
            return True
        except FileNotFoundError:
            if self.enable_logging:
                logging.warning(f"Blacklist file not found: {filepath}")
            return False
        except Exception as e:
            if self.enable_logging:
                logging.error(f"Error loading blacklist: {e}")
            return False
    
    def _track_non_chechen_char(self, char: str, word: str):
        """Track non-Chechen characters and their context."""
        if char not in self.non_chechen_chars:
            self.non_chechen_chars[char] = {}
        if word not in self.non_chechen_chars[char]:
            self.non_chechen_chars[char][word] = 0
        self.non_chechen_chars[char][word] += 1
    
    def _has_numbers(self, word: str) -> bool:
        """Check if word contains numbers."""
        return any(char.isdigit() for char in word)
    
    def _is_roman_numeral(self, word: str) -> bool:
        """
        Check if word is a Roman numeral that should be excluded from processing.
        
        Detects Roman numerals from I to MMMCMXCIX (3999) in both cases.
        These commonly appear in corpus as chapter/section markers.
        
        Examples: I, II, III, IV, V, VI, VII, VIII, IX, X, XI, XII, etc.
        """
        import re
        
        # Roman numeral pattern - handles 1 to 3999
        # Matches: I, II, III, IV, V, VI, VII, VIII, IX, X, XI, XII, XIII, XIV, XV, etc.
        roman_pattern = r'^(?=[MDCLXVI])M{0,4}(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$'
        
        # Check both uppercase and lowercase
        return bool(re.match(roman_pattern, word.upper()))
    
    def _apply_smart_character_replacements(self, word: str, track_changes: bool = False) -> str:
        """
        Apply character replacements with smart handling of digits.
        
        Only converts '1' to 'ӏ' (palochka) when it's truly isolated in Chechen words.
        Preserves years (1977ш), multi-digit numbers (11-чу), and all numeric contexts.
        
        Examples:
          - "1977ш" → "1977ш" (year preserved)
          - "11-чу" → "11-чу" (multi-digit number preserved)  
          - "к1ант" → "кӏант" (isolated '1' converted)
          - "х1ун" → "хӏун" (isolated '1' converted)
        """
        import re
        
        original_word = word
        
        # First, apply non-digit character replacements
        for old_char, new_char in self.char_replacements.items():
            if old_char not in '1':  # Handle '1' specially
                word = word.replace(old_char, new_char)
        
        # Smart '1' to 'ӏ' replacement - only convert truly isolated '1' characters
        # Preserves years (1977ш), multi-digit numbers (11-чу), and numeric contexts
        if '1' in word:
            # Don't convert if word contains sequences of 2+ consecutive digits
            # This catches years like "1977ш", "1964ш" and multi-digit numbers
            if re.search(r'\d{2,}', word):
                return word
                
            # Don't convert if '1' is adjacent to other digits
            # This catches cases like "21", "12", etc.
            if re.search(r'[0-9]1|1[0-9]', word):
                return word
                
            # Don't convert if word contains multiple separate digits  
            # This catches cases like "1-2", "3-1-5", etc.
            if len(re.findall(r'\d', word)) > 1:
                return word
                
            # Don't convert numeric contexts with hyphens (preserve existing logic)
            # This catches "1-чу", "11-чу", etc.
            if re.search(r'\d+-', word):
                return word
                
            # Safe to convert isolated '1' to 'ӏ' (cases like "к1ант", "х1ун")
            new_word = word.replace('1', 'ӏ')
            # Track the transformation if requested
            if track_changes and new_word != word:
                self.suspicious_transformations.append(
                    (original_word, new_word, 1)
                )
            word = new_word
        
        return word
    
    def normalize_word(self, word: str, track_changes: bool = False) -> str:
        """
        Normalize word with advanced character replacement and validation.
        
        Returns empty string if word should be skipped.
        """
        original_word = word
        
        # Convert to lowercase
        word = word.lower()
        
        # Skip Roman numerals entirely (I, II, III, IV, V, VI, etc.)
        # These appear in corpus as chapter/section markers and should not be processed
        if self._is_roman_numeral(word):
            return ""
        
        # Track non-Chechen characters before normalization
        for char in word:
            if (char not in self.chechen_letters and 
                char not in self.legal_symbols and 
                not char.isdigit()):
                self._track_non_chechen_char(char, original_word)
        
        # Apply character replacements with smart digit handling
        word = self._apply_smart_character_replacements(word, track_changes)
        
        # Skip words with remaining numbers (after character replacement)
        if self._has_numbers(word):
            return ""
        
        # Handle compound words with hyphens
        if '-' in word:
            parts = word.split('-')
            # Keep compound words if all parts are valid Chechen
            if all(all(char in self.chechen_letters for char in part) 
                   for part in parts if part):
                return word
        
        # Remove non-Chechen characters
        cleaned = ''.join(char for char in word if char in self.chechen_letters)
        
        return cleaned
    
    def is_valid_word(self, word: str) -> bool:
        """Check if word contains only valid Chechen characters."""
        if '-' in word:
            parts = word.split('-')
            return all(all(char in self.chechen_letters for char in part) 
                      for part in parts if part)
        return all(char in self.chechen_letters for char in word)
    
    def extract_and_clean_words(self, text: str, 
                               track_changes: bool = True) -> List[str]:
        """
        Extract and clean words from text with advanced processing.
        
        Args:
            text: Input text to process
            track_changes: Whether to track suspicious changes for reporting
            
        Returns:
            List of cleaned, valid words
        """
        valid_words = []
        
        # Extract words using regex
        raw_words = re.findall(r'[\w\-]+', text)
        
        for word in raw_words:
            original_word = word.lower()
            cleaned_word = self.normalize_word(word, track_changes)
            
            # Skip empty results
            if not cleaned_word:
                continue
            
            # Handle single-letter words
            if len(cleaned_word) == 1:
                if track_changes:
                    self.single_letter_words[cleaned_word] = \
                        self.single_letter_words.get(cleaned_word, 0) + 1
                
                if cleaned_word in self.valid_single_letters:
                    valid_words.append(cleaned_word)
                continue
            
            # Skip if length changed during cleaning (potentially problematic)
            if track_changes and len(cleaned_word) != len(original_word):
                self.length_changes.append((original_word, cleaned_word, 1))
                continue
            
            # Skip blacklisted words
            if cleaned_word in self.blacklisted_words:
                continue
            
            # Track suspicious transformations
            if track_changes and cleaned_word != original_word and cleaned_word:
                self.suspicious_transformations.append(
                    (original_word, cleaned_word, 1)
                )
            
            valid_words.append(cleaned_word)
        
        return valid_words
    
    def process_text(self, text: str) -> Dict[str, int]:
        """
        Process single text string and return word frequencies.
        Maintains quality tracking for generate_quality_report().
        
        Args:
            text: Input text string to process
            
        Returns:
            Dictionary of word frequencies
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        words = self.extract_and_clean_words(text, track_changes=True)
        return dict(Counter(words).most_common())
    
    def process_corpus(self, corpus_data: List[Dict]) -> Dict[str, int]:
        """
        Process entire corpus and return word frequencies.
        
        Args:
            corpus_data: List of corpus entries with 'text' field
            
        Returns:
            Dictionary of word frequencies
        """
        all_words = []
        
        for i, entry in enumerate(corpus_data):
            if i % 1000 == 0 and self.enable_logging:
                logging.info(f"Processing entry {i}/{len(corpus_data)}")
            
            if not isinstance(entry, dict) or "text" not in entry:
                continue
            
            text = entry["text"]
            if not isinstance(text, str):
                continue
            
            words = self.extract_and_clean_words(text, track_changes=True)
            all_words.extend(words)
        
        # Count frequencies
        return dict(Counter(all_words).most_common())
    
    def generate_quality_report(self) -> str:
        """Generate comprehensive quality report."""
        report_lines = []
        
        # Non-Chechen characters report
        if self.non_chechen_chars:
            report_lines.append("=== NON-CHECHEN CHARACTERS ===")
            for char, word_counts in sorted(self.non_chechen_chars.items()):
                total_count = sum(word_counts.values())
                report_lines.append(f"\nCharacter '{char}' (U+{ord(char):04X}): {total_count} occurrences")
                
                # Show top words containing this character
                top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                for word, count in top_words:
                    report_lines.append(f"  '{word}': {count}")
                
                if len(word_counts) > 5:
                    remaining = len(word_counts) - 5
                    remaining_count = sum(count for _, count in top_words[5:])
                    report_lines.append(f"  ... {remaining} more words ({remaining_count} occurrences)")
        
        # Single-letter words report
        if self.single_letter_words:
            report_lines.append("\n=== SINGLE-LETTER WORDS ===")
            for word, count in sorted(self.single_letter_words.items(), key=lambda x: x[1], reverse=True):
                status = "✓ valid" if word in self.valid_single_letters else "✗ invalid"
                report_lines.append(f"'{word}': {count} occurrences ({status})")
        
        # Suspicious transformations report
        if self.suspicious_transformations:
            report_lines.append("\n=== SUSPICIOUS TRANSFORMATIONS ===")
            transform_counter = Counter(
                (orig, fixed) for orig, fixed, _ in self.suspicious_transformations
            )
            for (orig, fixed), count in transform_counter.most_common(20):
                report_lines.append(f"'{orig}' → '{fixed}': {count} times")
        
        # Length changes report
        if self.length_changes:
            report_lines.append("\n=== LENGTH CHANGES (SKIPPED) ===")
            length_counter = Counter(
                (orig, fixed) for orig, fixed, _ in self.length_changes
            )
            for (orig, fixed), count in length_counter.most_common(20):
                report_lines.append(f"'{orig}' ({len(orig)}) → '{fixed}' ({len(fixed)}): {count} times")
        
        return "\n".join(report_lines)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get processing statistics."""
        return {
            'non_chechen_chars': len(self.non_chechen_chars),
            'suspicious_transformations': len(self.suspicious_transformations),
            'length_changes': len(self.length_changes),
            'single_letter_words': len(self.single_letter_words),
            'blacklisted_words': len(self.blacklisted_words)
        }