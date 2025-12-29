# Chechen Language Corpus Processing Toolkit

Advanced toolkit for processing Chechen language corpus data with sophisticated text analysis, quality control, and specialized wordlist generation for iOS text replacement shortcuts, Keyman keyboard software, linguistic analysis, and spell checkers.

[![Language](https://img.shields.io/badge/Language-Chechen-blue.svg)](https://en.wikipedia.org/wiki/Chechen_language)
[![Python](https://img.shields.io/badge/Python-3.7%2B-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Primary Data Source**: [corpora.dosham.info](https://corpora.dosham.info) - supports any JSON corpus with proper structure.

## Key Features

- **Advanced Text Processing**: Intelligent character normalization (і→ӏ, ι→ӏ, 1→ӏ, accent correction)
- **Quality Analysis**: Comprehensive corpus health checking with Unicode-level reporting
- **Four Processing Modes**: analyze, process, fix-corpus, and complete pipeline
- **Specialized Exports**: Custom wordlists optimized for iOS and Keyman applications
- **Smart Filtering**: Roman numeral detection, encoding issue resolution, compound word handling
- **Performance**: Efficiently processes large corpora (450k+ words in seconds)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quality analysis
python chechen_corpus_toolkit.py data/corpus.json --mode analyze --save-report

# Generate iOS text replacement wordlist
python chechen_corpus_toolkit.py data/corpus.json --mode process --export palochka

# Generate Keyman wordlist
python chechen_corpus_toolkit.py data/corpus.json --mode process --export keyman

# Generate all exports
python chechen_corpus_toolkit.py data/corpus.json --mode process --export all
```

## Processing Modes

### Quality Analysis

Analyze corpus quality before processing:

```bash
# Basic analysis
python chechen_corpus_toolkit.py input.json --mode analyze

# Save detailed report
python chechen_corpus_toolkit.py input.json --mode analyze --save-report

# Use custom blacklist
python chechen_corpus_toolkit.py input.json --mode analyze --blacklist blacklist.txt
```

**Analysis Features:**

- Non-Chechen character detection with Unicode codes
- Single-letter word validation (а, и, я, ю are valid)
- Character transformation preview (і→ӏ, ι→ӏ, etc.)
- Processing recommendations and quality metrics

### Direct Processing

Generate specialized wordlist exports:

```bash
# iOS text replacement shortcuts (words with palochka ӏ)
python chechen_corpus_toolkit.py input.json --mode process --export palochka

# Keyman keyboard predictions (optimized 2-30 char words)
python chechen_corpus_toolkit.py input.json --mode process --export keyman

# Both exports with detailed reporting
python chechen_corpus_toolkit.py input.json --mode process --export all --save-report --min-frequency 2
```

**Export Types:**

- **`palochka`**: Words containing ӏ (Chechen palochka) for iOS text replacement shortcuts
- **`keyman`**: Optimized for Keyman keyboard software predictions (1-27 chars, filtered for quality)
- **`all`**: Generates both palochka and keyman exports

### Corpus Normalization
Apply character normalizations to source corpus:

```bash
# Normalize corpus with character fixes
python chechen_corpus_toolkit.py input.json --mode fix-corpus --output cleaned.json

# Generate normalization report
python chechen_corpus_toolkit.py input.json --mode fix-corpus --output fixed.json --save-report
```

**Normalizations Applied:**

- i → ӏ (Latin lowercase i to Chechen palochka)
- I → ӏ (Latin uppercase I to Chechen palochka)
- і → ӏ (Ukrainian і to Chechen palochka)
- ι → ӏ (Greek iota to Chechen palochka)
- 1 → ӏ (digit 1 to Chechen palochka, context-aware - preserves years like 1977ш)
- à → а (Latin à with grave accent)
- á → а (Latin á with acute accent)
- è → е (Latin è with grave accent)
- é → е (Latin é with acute accent)
- ò → о (Latin ò with grave accent)
- y → у (Latin y to Cyrillic у)
- Roman numeral exclusion (I, II, III, IV, etc.)

### Complete Pipeline

Normalize corpus + generate all exports in one command:

```bash
# Complete processing pipeline
python chechen_corpus_toolkit.py input.json --mode all --export all

# Pipeline with frequency filtering and blacklist
python chechen_corpus_toolkit.py input.json --mode all --export all --min-frequency 3 --blacklist exclude.txt
```

## Output Files

```
exports/
├── palochka_words.tsv      # iOS text replacement shortcuts
├── keyman_wordlist.tsv     # Keyman keyboard predictions
├── analysis_report.txt     # Quality analysis results
└── processing_report.txt   # Processing details
```

**TSV Format:**
```
word    count
халкъан 1234
дош     856
```

## Advanced Processing Engine

The `ChechenTextProcessor` class provides sophisticated text analysis:

### Smart Character Normalization

- **Context-aware '1' conversion**: Preserves years (1977ш) while converting isolated cases (к1ант → кӏант)
- **Roman numeral detection**: Excludes I, II, III, IV, etc. from processing
- **Compound word handling**: Preserves hyphenated words with valid Chechen parts
- **Quality tracking**: Records all transformations for analysis

### Text Validation
- **Chechen alphabet validation**: Ensures words contain only valid Chechen characters
- **Single-letter word filtering**: Validates against allowed set (а, и, я, ю)
- **Blacklist support**: Excludes known non-words and artifacts

## Python Integration

### Simple Text Processing API

```python
from chechen_text_processor import ChechenTextProcessor

# Initialize processor
processor = ChechenTextProcessor(enable_logging=True)

# Process single text string
text = "Х1ара хьаннаш хедош, шайна ирзош дохуш, цкъа мацах кхузахь баха хевшина нах."
word_frequencies = processor.process_text(text)
quality_report = processor.generate_quality_report()

print(f"Found {len(word_frequencies)} unique words")
print(quality_report)
```

### Full Corpus Processing API

```python
# Process structured corpus data
corpus_data = [
    {"text": "Корех арахьаьжира со, хенан хӏотам муха бу-те аьлла."},
    {"text": "Нохчийн меттан хьехархочо цӏахь бан болх беллера тхуна."},
    {"text": "Х1ара хьаннаш хедош, шайна ирзош дохуш, цкъа мацах кхузахь баха хевшина нах."}
]

processor = ChechenTextProcessor(enable_logging=True)
processor.load_blacklist('exclusions.txt')  # Optional blacklist
word_frequencies = processor.process_corpus(corpus_data)

# Get quality analysis
report = processor.generate_quality_report()
print(report)
```

## Usage Examples

### iOS Text Replacement Shortcuts

! NEED REFACTOR HERE: the generated palochka wordlist arent ready to to import to ios replacements, there is just a tsv file, there are another repository who prepare the replacmement file.
Generate words with palochka (ӏ) for iOS keyboard shortcuts (Settings > General > Keyboard > Text Replacement):
```bash
python chechen_corpus_toolkit.py corpus.json --mode process --export palochka --save-report
# Output: exports/palochka_words.tsv
```

### Keyman Keyboard Predictions

Create optimized wordlists for [Keyman keyboard predictions](https://help.keyman.com/developer/current-version/guides/lexical-models):
```bash
python chechen_corpus_toolkit.py corpus.json --mode process --export keyman --min-frequency 2
# Output: exports/keyman_wordlist.tsv
```

### Corpus Structure
Your corpus JSON should follow this structure:
```json
[
  {
    "text": "Нохчийн меттан хьехархочо цӏахь бан болх беллера тхуна.",
  }
]
```

## Troubleshooting

### Common Issues

**Character encoding issues**: Use analysis mode first to identify problems

```bash
python chechen_corpus_toolkit.py problem_corpus.json --mode analyze --save-report
```

**Poor quality output**: Apply corpus normalization before processing

```bash
python chechen_corpus_toolkit.py messy_corpus.json --mode fix-corpus --output clean_corpus.json
python chechen_corpus_toolkit.py clean_corpus.json --mode process --export all
```

**Too many low-frequency words**: Use minimum frequency filtering

```bash
python chechen_corpus_toolkit.py corpus.json --mode process --export keyman --min-frequency 3
```

## Requirements

<!-- ```bash
pip install -r requirements.txt
``` -->

- Python 3.12

## License

MIT License - see [LICENSE](LICENSE) file.
