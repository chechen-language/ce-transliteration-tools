#!/usr/bin/env python3
"""
Unified Chechen Corpus Toolkit

A comprehensive tool for processing Chechen language corpus data with advanced
text processing, quality analysis, export generation, and corpus normalization.

Replaces the fragmented workflow of multiple scripts with a single, intuitive CLI.

Usage:
    python chechen_corpus_toolkit.py input_file --mode <mode> [options]

Modes:
    analyze     - Quality analysis only (no file generation)
    process     - Generate filtered TSV exports directly from corpus
    fix-corpus  - Apply character normalizations to corpus JSON
    all         - Complete pipeline (fix corpus + generate all exports)
"""

import json
import csv
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

from chechen_text_processor import ChechenTextProcessor


class CorpusAnalyzer:
    """Quality analysis functionality extracted from analyze_corpus_quality.py"""
    
    def __init__(self, blacklist_file: Optional[str] = None):
        self.processor = ChechenTextProcessor(enable_logging=False)
        if blacklist_file:
            self.processor.load_blacklist(blacklist_file)
    
    def analyze_corpus(self, corpus_data: List[Dict]) -> Dict[str, Any]:
        """Analyze corpus quality and return comprehensive results."""
        # Process corpus to get quality metrics
        word_frequencies = self.processor.process_corpus(corpus_data)
        
        total_words = sum(word_frequencies.values())
        unique_words = len(word_frequencies)
        
        # Prepare analysis results
        analysis_result = {
            'statistics': {
                'total_texts': len(corpus_data),
                'total_words': total_words,
                'unique_words': unique_words
            },
            'character_issues': self._analyze_character_issues(),
            'transformation_analysis': self._analyze_transformations(),
            'processing_stats': self.processor.get_statistics(),
            'recommendations': self._generate_recommendations()
        }
        
        return analysis_result
    
    def _analyze_character_issues(self) -> Dict[str, Dict]:
        """Analyze non-Chechen characters by category."""
        if not self.processor.non_chechen_chars:
            return {}
        
        categories = {
            'latin_chars': {},
            'diacritics': {},
            'punctuation': {},
            'other_chars': {}
        }
        
        for char, word_counts in self.processor.non_chechen_chars.items():
            total = sum(word_counts.values())
            
            if ord(char) < 128:  # ASCII/Latin
                categories['latin_chars'][char] = total
            elif 0x0300 <= ord(char) <= 0x036F:  # Combining diacritics
                categories['diacritics'][char] = total
            elif char in '""''—–…':  # Smart quotes and dashes
                categories['punctuation'][char] = total
            else:
                categories['other_chars'][char] = total
        
        return categories
    
    def _analyze_transformations(self) -> Dict[str, List]:
        """Analyze character transformation patterns."""
        if not self.processor.suspicious_transformations:
            return {}
        
        categories = {
            'palochka_fixes': [],
            'accent_fixes': [],
            'other_fixes': []
        }
        
        for orig, fixed, count in self.processor.suspicious_transformations:
            if 'ӏ' in fixed and any(c in orig for c in 'iI1ιі'):
                categories['palochka_fixes'].append((orig, fixed, count))
            elif any(c in orig for c in 'àáòèéy'):
                categories['accent_fixes'].append((orig, fixed, count))
            else:
                categories['other_fixes'].append((orig, fixed, count))
        
        return categories
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        stats = self.processor.get_statistics()
        
        if stats['non_chechen_chars'] == 0:
            recommendations.append("✅ Excellent! No character normalization issues detected.")
        else:
            recommendations.append(f"⚠️  Found {stats['non_chechen_chars']} types of non-Chechen characters.")
            recommendations.append("   → Consider using fix-corpus mode to normalize characters.")
        
        if stats['length_changes'] > 0:
            recommendations.append(f"⚠️  {stats['length_changes']} words skipped due to length changes.")
            recommendations.append("   → These words may contain mixed scripts or encoding issues.")
        
        if not self.processor.blacklisted_words:
            recommendations.append("• Consider adding a blacklist file to exclude non-words")
        
        return recommendations


class ExportGenerator:
    """Export filtering functionality extracted from export_wordlist.py"""
    
    @staticmethod
    def filter_palochka_words(wordlist: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Filter words containing the Chechen letter ӏ (palochka)."""
        return [(word, count) for word, count in wordlist if 'ӏ' in word or 'Ӏ' in word]
    
    @staticmethod
    def filter_keyman_words(wordlist: List[Tuple[str, int]], 
                           min_length: int = 1, max_length: int = 27) -> List[Tuple[str, int]]:
        """Filter words optimized for Keyman keyboard predictions."""
        # Valid single-letter Chechen words
        valid_single_letters = {'а', 'и', 'я', 'ю'}
        filtered_words = []
        
        for word, count in wordlist:
            # Handle single character words
            if len(word) == 1:
                # Keep valid single-letter Chechen words
                if word in valid_single_letters:
                    filtered_words.append((word, count))
                continue
            
            # Remove words that are just numbers
            if word.isdigit():
                continue
            
            
            # For Keyman: apply length filter
            if not (min_length <= len(word) <= max_length):
                continue
            
            # Remove words with excessive repetition (like "ааа")
            if len(set(word)) == 1 and len(word) > 2:
                continue
            
            filtered_words.append((word, count))
        
        return filtered_words


class CorpusNormalizer:
    """JSON corpus normalization functionality"""
    
    def __init__(self):
        self.processor = ChechenTextProcessor(enable_logging=False)
        self.transformations = []
    
    def normalize_corpus(self, corpus_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Apply character normalizations to corpus JSON."""
        normalized_corpus = []
        self.transformations = []
        
        for entry in corpus_data:
            if not isinstance(entry, dict) or "text" not in entry:
                normalized_corpus.append(entry)
                continue
            
            original_text = entry["text"]
            if not isinstance(original_text, str):
                normalized_corpus.append(entry)
                continue
            
            # Apply character normalizations
            normalized_text = self._normalize_text(original_text)
            
            # Track transformation if text changed
            if normalized_text != original_text:
                self.transformations.append({
                    'original': original_text[:100],  # First 100 chars for reference
                    'normalized': normalized_text[:100],
                    'changes': self._count_character_changes(original_text, normalized_text)
                })
            
            # Create normalized entry
            normalized_entry = entry.copy()
            normalized_entry["text"] = normalized_text
            normalized_corpus.append(normalized_entry)
        
        return normalized_corpus, self.transformations
    
    def _normalize_text(self, text: str) -> str:
        """Apply the same character normalizations as ChechenTextProcessor."""
        normalized = text
        
        # Apply character replacements
        for old_char, new_char in self.processor.char_replacements.items():
            normalized = normalized.replace(old_char, new_char)
        
        return normalized
    
    def _count_character_changes(self, original: str, normalized: str) -> Dict[str, int]:
        """Count specific character transformations."""
        changes = {}
        for old_char, new_char in self.processor.char_replacements.items():
            old_count = original.count(old_char)
            new_count = normalized.count(old_char)
            if old_count > new_count:
                changes[f"{old_char}→{new_char}"] = old_count - new_count
        return changes


class ReportGenerator:
    """Unified report generation for all modes - matches tokenize.py format"""
    
    @staticmethod
    def generate_analysis_report(analysis_result: Dict[str, Any], processor: ChechenTextProcessor) -> str:
        """Generate comprehensive analysis report matching tokenize.py format."""
        lines = []
        
        # Include complete detailed analysis first
        analysis_lines = ReportGenerator._generate_unified_analysis(
            processor, 
            max_words_per_char=10, 
            max_transformations=None,  # Show all transformations
            include_length_changes=True
        )
        if analysis_lines:
            lines.extend(analysis_lines)
        
        # Summary statistics at the end
        stats = analysis_result['statistics']
        lines.append("Summary Statistics:")
        lines.append(f"Total texts processed: {stats['total_texts']:,}")
        lines.append(f"Total words processed: {stats['total_words']:,}")
        lines.append(f"Unique words found: {stats['unique_words']:,}")
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_processing_report(word_frequencies: Dict[str, int], 
                                 exports: Dict[str, List[Tuple[str, int]]], 
                                 processor: ChechenTextProcessor,
                                 total_texts: int = 0) -> str:
        """Generate processing report with complete analysis like tokenize.py."""
        lines = []
        
        # Top 10 most frequent words
        top_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
        lines.append("Top 10 most frequent words:")
        for i, (word, count) in enumerate(top_words, 1):
            lines.append(f"  {i:2d}. '{word}': {count:,} occurrences")
        lines.append("")
        
        # Include SAME complete detailed analysis as analysis report
        analysis_lines = ReportGenerator._generate_unified_analysis(
            processor, 
            max_words_per_char=10,    # SAME as analysis report
            max_transformations=None, # SAME as analysis report (unlimited)
            include_length_changes=True  # SAME as analysis report
        )
        if analysis_lines:
            lines.extend(analysis_lines)
        
        # Processing Summary (for report file)
        total_words = sum(word_frequencies.values())
        unique_words = len(word_frequencies)
        lines.append("Processing Summary:")
        lines.append(f"Total words processed: {total_words:,}")
        lines.append(f"Unique words found: {unique_words:,}")
        lines.append("")
        lines.append("Export Summary:")
        for export_type, export_words in exports.items():
            export_name = export_type.capitalize()
            lines.append(f"  {export_name}: {len(export_words):,} words")
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_normalization_report(transformations: List[Dict]) -> str:
        """Generate normalization report for fix-corpus mode."""
        lines = []
        
        if not transformations:
            lines.append("✅ No character normalizations needed - corpus is clean!")
            return "\n".join(lines)
        
        lines.append(f"Texts modified: {len(transformations):,}")
        lines.append("")
        
        # Count total character changes
        total_changes = {}
        for transform in transformations:
            for change, count in transform['changes'].items():
                total_changes[change] = total_changes.get(change, 0) + count
        
        if total_changes:
            lines.append("Character transformations applied:")
            for change, count in sorted(total_changes.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {change}: {count:,} transformations")
            lines.append("")
        
        # Show sample transformations
        lines.append("Sample transformations:")
        for i, transform in enumerate(transformations[:5]):
            lines.append(f"\n  Sample {i+1}:")
            original = transform['original'][:80] + ('...' if len(transform['original']) > 80 else '')
            normalized = transform['normalized'][:80] + ('...' if len(transform['normalized']) > 80 else '')
            lines.append(f"    Before: {original}")
            lines.append(f"    After:  {normalized}")
        
        if len(transformations) > 5:
            lines.append(f"\n... and {len(transformations) - 5:,} more transformations")
        
        return "\n".join(lines)
    
    @staticmethod
    def _generate_unified_analysis(processor: ChechenTextProcessor, 
                                 max_words_per_char: int = 10,
                                 max_transformations: int = None,
                                 include_length_changes: bool = True) -> List[str]:
        """Generate unified analysis with configurable detail levels."""
        lines = []
        
        # Non-Chechen characters
        if processor.non_chechen_chars:
            lines.append("Non-Chechen characters found:")
            # Show each character with count and Unicode (like single-letter words format)
            for char in sorted(processor.non_chechen_chars.keys()):
                word_counts = processor.non_chechen_chars[char]
                total_count = sum(word_counts.values())
                lines.append(f"  '{char}' (U+{ord(char):04X}): {total_count} occurrences")
            lines.append("")
            
            # Then show detailed breakdown for each character
            for char, word_counts in sorted(processor.non_chechen_chars.items()):
                total_count = sum(word_counts.values())
                lines.append(f"\nCharacter '{char}' (Unicode: U+{ord(char):04X}) found {total_count} times:")
                # Show top N words containing this character
                sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
                for word, count in sorted_words[:max_words_per_char]:
                    lines.append(f"  In word '{word}': {count} occurrences")
                if len(word_counts) > max_words_per_char:
                    remaining = sum(count for _, count in sorted_words[max_words_per_char:])
                    lines.append(f"  ... and {remaining} more occurrences in other words")
            lines.append("")
        
        # Single-letter words
        if processor.single_letter_words:
            lines.append("Single-letter words found:")
            valid_single_letters = {'а', 'и', 'я', 'ю'}
            for word, count in processor.single_letter_words.items():
                status = "valid" if word in valid_single_letters else "invalid"
                lines.append(f"  '{word}': {count} occurrences ({status})")
            lines.append("")
        
        # Suspicious transformations
        if processor.suspicious_transformations:
            if max_transformations is None:
                lines.append("All suspicious word transformations:")
            else:
                lines.append(f"Suspicious word transformations (top {max_transformations}):")
            
            transform_counter = Counter(
                (orig, fixed) for orig, fixed, _ in processor.suspicious_transformations
            )
            
            transformations = transform_counter.most_common(max_transformations)
            for (orig, fixed), count in transformations:
                lines.append(f"  '{orig}' → '{fixed}': {count} occurrences")
            lines.append("")
        
        # Length changes (separate section for analyze mode)
        if include_length_changes and processor.length_changes:
            lines.append("Suspicious transformations with length changes (skipped):")
            length_change_counter = Counter(
                (orig, fixed) for orig, fixed, _ in processor.length_changes
            )
            for (orig, fixed), count in length_change_counter.most_common():
                lines.append(f"  '{orig}' ({len(orig)}) → '{fixed}' ({len(fixed)}): {count} occurrences")
            lines.append("")
        
        return lines


class ChechenCorpusToolkit:
    """Main toolkit class orchestrating all functionality"""
    
    def __init__(self):
        self.analyzer = None
        self.normalizer = None
    
    def _print(self, message: str) -> None:
        """Print message to console."""
        print(message)
        
    def load_corpus(self, file_path: str) -> List[Dict]:
        """Load JSON corpus data from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if not isinstance(data, list):
                    raise ValueError("Corpus must be a JSON array")
                self._print(f"Loaded {len(data)} texts from corpus")
                return data
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in '{file_path}': {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error loading corpus: {e}", file=sys.stderr)
            sys.exit(1)
    
    def save_json(self, data: Any, file_path: str) -> None:
        """Save data to JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
            self._print(f"Saved to: {file_path}")
        except Exception as e:
            print(f"Error saving JSON: {e}", file=sys.stderr)
            sys.exit(1)
    
    def save_tsv(self, wordlist: List[Tuple[str, int]], file_path: str, silent: bool = False) -> Dict[str, any]:
        """Save wordlist to TSV file.
        
        Returns:
            Dictionary with export information for deferred output
        """
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter='\t')
                writer.writerow(['word', 'count'])
                writer.writerows(wordlist)
            
            export_info = {
                'word_count': len(wordlist),
                'file_path': file_path,
                'message': f"Exported {len(wordlist)} words to {file_path}"
            }
            
            if not silent:
                self._print(export_info['message'])
            
            return export_info
            
        except Exception as e:
            print(f"Error saving TSV: {e}", file=sys.stderr)
            sys.exit(1)
    
    def mode_analyze(self, input_file: str, blacklist_file: Optional[str] = None, 
                    save_report: Optional[str] = None, output_dir: str = "exports") -> None:
        """Execute analyze mode - quality analysis only."""
        self._print("=== ANALYZE MODE ===")
        
        # Create output directory for consistency
        if save_report:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load corpus
        corpus_data = self.load_corpus(input_file)
        
        # Analyze corpus
        self._print("Analyzing corpus quality...")
        self.analyzer = CorpusAnalyzer(blacklist_file)
        analysis_result = self.analyzer.analyze_corpus(corpus_data)
        
        # Generate and display report
        report = ReportGenerator.generate_analysis_report(analysis_result, self.analyzer.processor)
        self._print(f"\n{report}")  # Always show analysis report
        
        # Save report if requested
        if save_report:
            # Use consistent path structure like other modes
            if save_report == 'analysis_report.txt':
                report_file = f"{output_dir}/analysis_report.txt"
            else:
                report_file = save_report
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"Chechen Corpus Quality Analysis\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input file: {input_file}\n")
                f.write(f"Output directory: {output_dir}\n")
                f.write("=" * 50 + "\n\n")
                f.write(report)
            self._print(f"Analysis report saved to: {report_file}")
    
    def mode_process(self, input_file: str, export_types: List[str], 
                    output_dir: str = "exports", min_frequency: int = 1,
                    blacklist_file: Optional[str] = None, save_report: bool = False) -> None:
        """Execute process mode - generate filtered exports directly from corpus."""
        self._print("=== PROCESS MODE ===")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load corpus
        corpus_data = self.load_corpus(input_file)
        
        # Process corpus
        self._print("Processing corpus with advanced text processor...")
        processor = ChechenTextProcessor(enable_logging=False)
        if blacklist_file:
            if processor.load_blacklist(blacklist_file):
                self._print(f"Loaded blacklist: {blacklist_file}")
        
        word_frequencies = processor.process_corpus(corpus_data)
        
        # Convert to sorted wordlist
        wordlist = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
        
        # Apply minimum frequency filter
        if min_frequency > 1:
            wordlist = [(word, count) for word, count in wordlist if count >= min_frequency]
        
        # Generate exports
        exports = {}
        export_info = {}
        export_generator = ExportGenerator()
        
        for export_type in export_types:
            if export_type == 'palochka':
                filtered_words = export_generator.filter_palochka_words(wordlist)
                output_file = f"{output_dir}/palochka_words.tsv"
            elif export_type == 'keyman':
                filtered_words = export_generator.filter_keyman_words(wordlist, min_length=1, max_length=27)
                output_file = f"{output_dir}/keyman_wordlist.tsv"
            else:
                print(f"Warning: Unknown export type '{export_type}', skipping")
                continue
            
            # Save export (silent mode - collect info for later display)
            info = self.save_tsv(filtered_words, output_file, silent=True)
            exports[export_type] = filtered_words
            export_info[export_type] = info
        
        # Generate and save processing report (collect info for later display)
        report_file_info = None
        if save_report:
            report = ReportGenerator.generate_processing_report(word_frequencies, exports, processor, len(corpus_data))
            report_file = f"{output_dir}/processing_report.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"Chechen Corpus Processing Report\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input file: {input_file}\n")
                f.write(f"Output directory: {output_dir}\n")
                f.write(f"Export types: {', '.join(export_types)}\n")
                f.write(f"Minimum frequency: {min_frequency}\n")
                f.write("=" * 50 + "\n\n")
                f.write(report)
            report_file_info = report_file
            
            # Show report in console
            self._print(f"\n{report}")
        
        # END SUMMARY - Display all export and completion info
        self._print("")
        for export_type, info in export_info.items():
            self._print(info['message'])
        
        if report_file_info:
            self._print(f"Processing report saved to: {report_file_info}")
        
        # Always show Processing Summary in CLI
        total_words = sum(word_frequencies.values())
        unique_words = len(word_frequencies)
        
        self._print("Processing Summary:")
        self._print(f"Total words processed: {total_words:,}")
        self._print(f"Unique words found: {unique_words:,}")
        self._print("")
        self._print("Export Summary:")
        for export_type, info in export_info.items():
            export_name = export_type.capitalize()
            self._print(f"  {export_name}: {info['word_count']:,} words")
        
        self._print(f"\n✅ Processing complete! Generated {len(exports)} export(s)")
    
    def mode_fix_corpus(self, input_file: str, output_file: str, 
                       save_report: bool = False) -> None:
        """Execute fix-corpus mode - normalize corpus JSON."""
        self._print("=== FIX-CORPUS MODE ===")
        
        # Load corpus
        corpus_data = self.load_corpus(input_file)
        
        # Normalize corpus
        self._print("Applying character normalizations to corpus...")
        self.normalizer = CorpusNormalizer()
        normalized_corpus, transformations = self.normalizer.normalize_corpus(corpus_data)
        
        # Save normalized corpus
        self.save_json(normalized_corpus, output_file)
        
        # Generate report
        if save_report or transformations:
            report = ReportGenerator.generate_normalization_report(transformations)
            
            # Show report if there are transformations
            if transformations:
                self._print(f"\n{report}")
            
            if save_report:
                # Use consistent naming: place report in same directory as output file
                output_path = Path(output_file)
                report_file = output_path.parent / f"{output_path.stem}_normalization_report.txt"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(f"Chechen Corpus Normalization Report\n")
                    f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Input file: {input_file}\n")
                    f.write(f"Output file: {output_file}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(report)
                self._print(f"Normalization report saved to: {report_file}")
        
        self._print(f"\n✅ Corpus normalization complete!")
    
    def mode_all(self, input_file: str, export_types: List[str], 
                output_dir: str = "exports", min_frequency: int = 1,
                blacklist_file: Optional[str] = None) -> None:
        """Execute all mode - complete pipeline."""
        self._print("=== ALL MODE - COMPLETE PIPELINE ===")
        
        # Step 1: Fix corpus
        fixed_corpus_file = f"{output_dir}/normalized_corpus.json"
        self._print("\n🔧 Step 1: Normalizing corpus...")
        self.mode_fix_corpus(input_file, fixed_corpus_file, save_report=False)
        
        # Step 2: Process exports from normalized corpus
        self._print("\n📊 Step 2: Processing exports...")
        self.mode_process(fixed_corpus_file, export_types, output_dir, 
                         min_frequency, blacklist_file, save_report=True)
        
        self._print(f"\n🎉 Complete pipeline finished! Check {output_dir}/ for all outputs")


def main():
    parser = argparse.ArgumentParser(
        description='Unified Chechen Corpus Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quality analysis only
  python chechen_corpus_toolkit.py corpus.json --mode analyze
  
  # Generate palochka export for iOS
  python chechen_corpus_toolkit.py corpus.json --mode process --export palochka
  
  # Generate all exports with quality report
  python chechen_corpus_toolkit.py corpus.json --mode process --export all --save-report
  
  # Fix corpus JSON with normalizations
  python chechen_corpus_toolkit.py corpus.json --mode fix-corpus --output fixed_corpus.json
  
  # Complete pipeline
  python chechen_corpus_toolkit.py corpus.json --mode all --export all
        """
    )
    
    # Required arguments
    parser.add_argument('input_file', help='Input JSON corpus file')
    parser.add_argument('--mode', required=True, 
                       choices=['analyze', 'process', 'fix-corpus', 'all'],
                       help='Processing mode')
    
    # Common options
    parser.add_argument('--output-dir', default='exports',
                       help='Output directory (default: exports)')
    parser.add_argument('--blacklist', help='Blacklist file path')
    parser.add_argument('--min-frequency', type=int, default=1,
                       help='Minimum word frequency (default: 1)')
    parser.add_argument('--save-report', action='store_true',
                       help='Save quality/processing report')
    
    # Mode-specific options
    parser.add_argument('--export', action='append', 
                       choices=['palochka', 'keyman', 'all'],
                       help='Export type(s) to generate (can be used multiple times)')
    parser.add_argument('--output', help='Output file (for fix-corpus mode)')
    
    args = parser.parse_args()
    
    # Validate mode-specific arguments
    if args.mode in ['process', 'all']:
        if not args.export:
            print("Error: --export is required for process/all modes", file=sys.stderr)
            sys.exit(1)
        
        # Handle 'all' export type
        if 'all' in args.export:
            args.export = ['palochka', 'keyman']
    
    if args.mode == 'fix-corpus' and not args.output:
        print("Error: --output is required for fix-corpus mode", file=sys.stderr)
        sys.exit(1)
    
    # Initialize toolkit
    toolkit = ChechenCorpusToolkit()
    
    # Execute based on mode
    start_time = time.time()
    
    try:
        if args.mode == 'analyze':
            toolkit.mode_analyze(
                args.input_file, 
                args.blacklist, 
                'analysis_report.txt' if args.save_report else None,
                args.output_dir
            )
            
        elif args.mode == 'process':
            toolkit.mode_process(
                args.input_file, 
                args.export, 
                args.output_dir,
                args.min_frequency,
                args.blacklist,
                args.save_report
            )
            
        elif args.mode == 'fix-corpus':
            toolkit.mode_fix_corpus(
                args.input_file,
                args.output,
                args.save_report
            )
            
        elif args.mode == 'all':
            toolkit.mode_all(
                args.input_file,
                args.export,
                args.output_dir,
                args.min_frequency,
                args.blacklist
            )
        
        end_time = time.time()
        print(f"\n⏱️  Total execution time: {end_time - start_time:.1f} seconds")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()