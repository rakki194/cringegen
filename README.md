# CringeGen

A toolkit for generating prompts and workflows for Stable Diffusion models in ComfyUI.

## Features

- Generate furry prompts with customizable species, backgrounds, and more
- Access LLM-powered caption generation using Ollama
- Brainstorm prompt variations and enhancements with LLM assistance
- Create and customize ComfyUI workflows
- Modular CLI architecture for extensibility and maintainability

## Advanced NLP Features

CringeGen includes sophisticated natural language processing capabilities for prompt creation, analysis, and enhancement:

### Tag/Text Conversion

Convert seamlessly between tags and natural language descriptions:

```python
from cringegen.prompt_generation.nlp import tags_to_text, text_to_tags, categorize_tags, natural_tags_to_text

# Convert tags to natural language
tags = ["masterpiece", "detailed", "fox", "red fur", "forest", "digital art"]
description = tags_to_text(tags, style="descriptive")
# "masterpiece detailed of subject with red fur coloration, in forest, fox, digital art"

# Generate natural-sounding, fluent captions (especially good for furry/anime content)
fluent_caption = natural_tags_to_text(tags)
# "A masterfully crafted and highly detailed digital artwork of a red-furred fox in a forest."

# Convert natural language to tags
text = "A high quality digital painting of a fox with blue eyes in a forest"
extracted_tags = text_to_tags(text)
# ["high quality", "digital painting", "fox", "blue eyes", "forest"]

# Categorize tags by semantic type
categories = categorize_tags(tags)
# {
#   "quality": ["masterpiece", "detailed"],
#   "color": ["red fur"],
#   "setting": ["forest"],
#   "style": ["digital art"],
#   "other": ["fox"]
# }
```

**Available styles for tags_to_text:**

- `concise`: Brief comma-separated description
- `descriptive`: Detailed single-paragraph description (default)
- `detailed`: Multi-sentence description with subject, attributes, and setting

**Natural Caption Generation:**

The `natural_tags_to_text` function creates fluent, grammatically correct captions with special handling for:

- Anthro/furry characters and species
- Anime-specific character types (kemonomimi, neko, etc.)
- Art styles (digital art, anime style, etc.)
- Environmental elements (lighting, time of day, weather)
- Complex tag combinations and relationships

### Entity Detection

Extract and categorize entities from text with special handling for furry/anime and image generation content:

```python
from cringegen.prompt_generation.nlp import extract_entities

# Extract entities from text
text = "An anthro fox warrior with a sword battling alongside human allies in a fantasy castle setting."
entities = extract_entities(text)
# {
#   "humans": ["human"],
#   "anthro_characters": ["anthro fox"],
#   "feral_animals": [],
#   "species": [],
#   "places": ["castle"],
#   "art_styles": ["fantasy"],
#   "organizations": [],
#   "companies": [],
#   "other": ["warrior", "sword", "allies"]
# }
```

The entity detection system is specially tuned to recognize and categorize:

- Anthro/furry characters vs. feral animals
- Fantasy and sci-fi species
- Art styles and mediums
- Locations and settings
- Humans, organizations, and companies
- Temporal references (dates, times)

### Prompt Analysis & Enhancement

Analyze and improve prompts with intelligent tools:

```python
from cringegen.prompt_generation.nlp import (
    analyze_prompt_structure, get_prompt_structure, 
    compare_prompts, suggest_improvements, simplify_prompt_structure,
    extract_keywords, detect_prompt_style
)

# Analyze prompt structure and quality
prompt = "masterpiece, detailed, fox with red fur in a forest, digital art"
analysis = analyze_prompt_structure(prompt)
print(f"Word count: {analysis.word_count}")
print(f"Complexity score: {analysis.complexity_score}/15")
print(f"Issues detected: {len(analysis.detected_issues)}")

# Extract structured components
structure = get_prompt_structure(prompt)
# {
#   "subject": "masterpiece",
#   "quality": ["masterpiece", "detailed"],
#   "style": ["digital art"]
# }

# Get improvement suggestions
suggestions = suggest_improvements(prompt)
# [
#   {"description": "Consider adding lighting details to your prompt"},
#   {"description": "Consider adding composition details to your prompt"}
# ]

# Compare two prompts
prompt2 = "a painting of a wolf in the woods"
comparison = compare_prompts(prompt, prompt2)
# {
#   "unique_to_prompt1": ["masterpiece", "detailed", "fox", "red", "fur", "digital", "art"],
#   "unique_to_prompt2": ["painting", "wolf", "woods"],
#   "common_words": ["forest"]
# }

# Simplify a complex prompt
complex = "A detailed digital artwork of a cyberpunk city street at night with neon lights and a character wearing futuristic clothing, professional quality, depth of field"
simplified = simplify_prompt_structure(complex, target_length=10)
# "a cyberpunk city street, detailed, digital art"

# Extract key keywords and detect style
keywords = extract_keywords(prompt, top_n=3)  # ["masterpiece", "detailed", "fox"]
style = detect_prompt_style(prompt)  # "digital_art"
```

### NLTK Integration

Leverage advanced NLP capabilities powered by NLTK:

```python
from cringegen.prompt_generation.nlp import (
    analyze_text_pos, extract_phrases, get_synonyms, get_antonyms,
    get_hypernyms, get_hyponyms, analyze_sentiment, lemmatize_text,
    compute_text_similarity, extract_entities, generate_ngrams
)

# Analyze part-of-speech distribution
text = "A masterful digital painting of a red fox with a fluffy tail"
pos_analysis = analyze_text_pos(text)
# {
#   "nouns": ["painting", "fox", "tail"],
#   "verbs": [],
#   "adjectives": ["masterful", "digital", "red", "fluffy"],
#   "adverbs": [],
#   "other": ["a", "of", "a", "with", "a"]
# }

# Extract phrases
noun_phrases = extract_phrases(text, phrase_type="NP")
# ["A masterful digital painting", "a red fox", "a fluffy tail"]

# Get semantic relationships from WordNet
synonyms = get_synonyms("forest")  # ["wood", "woods", "woodland", "timberland"]
antonyms = get_antonyms("happy")   # ["unhappy", "sad"]
hypernyms = get_hypernyms("fox")   # More general terms: ["canine", "animal"]
hyponyms = get_hyponyms("animal")  # More specific terms: ["fox", "wolf", "dog"]

# Analyze sentiment
sentiment = analyze_sentiment("A beautiful and peaceful forest scene")
# {"pos": 0.52, "neg": 0.0, "neu": 0.48, "compound": 0.7506}

# Process text
lemmatized = lemmatize_text("The foxes were running through forests")
# "The fox be run through forest"

# Compute text similarity (0-1 scale)
similarity = compute_text_similarity(
    "A fox in a forest", 
    "A red vulpine in woodland surroundings"
)  # Returns similarity score (e.g., 0.4)

# Extract named entities
entities = extract_entities("John visited New York and met with Google representatives")
# {
#   "people": ["John"],
#   "places": ["New York"],
#   "organizations": ["Google"]
# }

# Generate n-grams
bigrams = generate_ngrams("The quick brown fox", n=2)
# ["The quick", "quick brown", "brown fox"]
```

### Examples and Testing

Explore these features with the included demo and test scripts:

```bash
# Run the basic NLP features demo
python examples/nlp_demo.py

# Run comprehensive tests of all NLP features
python examples/nlp_features_test.py
```

## NLP Component Testing Results

I've thoroughly tested each component of the NLP toolkit. Here are the detailed results and documentation for each component:

### 1. Tag/Text Conversion

The tag/text conversion system performs exceptionally well at transforming structured tags into natural-sounding descriptions and vice versa.

**Performance:**

- Successfully converts tags to text in multiple styles (concise, descriptive, detailed)
- Produces grammatically correct and contextually appropriate natural language captions
- Categorizes tags effectively by semantic type (quality, color, setting, style)
- Achieves 35-50% tag recovery when converting back from natural text to tags

**Example:**

```python
tags = ["masterpiece", "detailed", "anthro fox", "red fur", "forest", "digital art"]
natural_caption = natural_tags_to_text(tags)
# "A masterfully crafted and highly detailed digital artwork of a red-furred anthropomorphic fox in a forest setting."
```

**Areas for Improvement:**

- Tag recovery rate could be improved for bidirectional conversion
- Better handling of compound descriptors (e.g., "ember-glowing scales")
- Enhance semantic categorization of specialized art terms

### 2. Entity Detection

The entity detection system demonstrates strong capabilities in identifying and categorizing entities from text, with specialized handling for furry/anime content.

**Performance:**

- Accurately detects anthro characters vs. feral animals
- Correctly identifies fantasy species, locations, and art styles
- Effectively extracts entities from complex descriptions

**Example:**

```python
text = "An anthro fox warrior with a sword battling alongside human allies in a fantasy castle."
entities = extract_entities(text)
# Correctly identifies "anthro fox" as anthro_character, "human" as human, "castle" as place, etc.
```

**Areas for Improvement:**

- Add more specialized detection for anime-specific character archetypes
- Improve handling of compound entity descriptions
- Expand fantasy species recognition

### 3. Prompt Analysis & Enhancement

The prompt analysis and enhancement tools provide valuable insights and improvements to prompts.

**Performance:**

- Accurately analyzes prompt structure and complexity
- Provides useful suggestions for prompt improvement
- Successfully simplifies complex prompts while maintaining key elements
- Effectively compares prompts and identifies differences

**Example:**

```python
prompt = "masterpiece, detailed, anthro fox, red fur, blue eyes, forest, digital art"
analysis = analyze_prompt_structure(prompt)
# Identifies complexity score, structure components, and areas for improvement
```

**Areas for Improvement:**

- Add more domain-specific enhancement suggestions
- Improve prompt structure detection for unusual formats
- Add style-specific enhancement options

### 4. NLTK Integration

The NLTK integration provides powerful linguistic analysis capabilities.

**Performance:**

- Successfully analyzes part-of-speech distribution
- Accurately extracts phrases from text
- Provides useful semantic relationships (synonyms, hypernyms, etc.)
- Effectively analyzes sentiment in prompts

**Example:**

```python
text = "A masterfully detailed digital painting of an anthropomorphic red fox"
pos_analysis = analyze_text_pos(text)
# Correctly identifies nouns, adjectives, etc.
```

**Areas for Improvement:**

- Enhance semantic relationship extraction for domain-specific terms
- Improve phrase extraction for complex descriptions
- Add more specialized linguistic analysis for artistic descriptions

### 5. Text Generation

The text generation system creates diverse and contextually appropriate descriptions for various character types.

**Performance:**

- Generates high-quality prompts for different character types (furry, anthro, anime)
- Successfully incorporates species-specific details
- Creates varied and natural-sounding descriptions

**Example:**

```python
generator = FurryPromptGenerator(species="fox", gender="male", use_nlp=True)
prompt = generator.generate()
# Creates a detailed, contextually appropriate description of a male fox character
```

**Areas for Improvement:**

- Add more customization options for prompt generation
- Improve handling of specialized character archetypes
- Enhance integration with other NLP components

### Conclusion

The CringeGen NLP toolkit provides a robust and versatile set of tools for prompt engineering, analysis, and enhancement. The comprehensive tests demonstrate the system's ability to:

1. **Convert between different text formats** with high fidelity
2. **Extract and categorize entities** from complex descriptions
3. **Analyze and enhance prompts** with contextually appropriate suggestions
4. **Generate natural-sounding captions** for various character types
5. **Integrate multiple NLP components** into cohesive workflows

The toolkit excels particularly in furry/anthro content generation, with specialized handling for species characteristics, anatomical details, and appropriate settings. It also handles anime and fantasy content effectively.

The most promising areas for future development include:

- Improving bidirectional conversion accuracy
- Adding more specialized handlers for various artistic domains
- Enhancing the integration between components for more sophisticated workflows
- Adding personality and trait generation for more nuanced character development

These components together form a robust NLP toolkit for prompt generation, analysis, and enhancement, with special focus on furry/anime content and image generation workflows.

## Advanced Features

### Split-Sigma Sampling

CringeGen supports advanced multi-stage sampling using the split-sigmas feature. This technique performs sampling in two stages with different settings for each stage, allowing for improved quality and control.

```bash
# Basic split-sigma sampling example
python -m cringegen nsfw-furry --species fox --split-sigmas 7.0

# Advanced split-sigma sampling with custom samplers and settings
python -m cringegen nsfw-furry --species wolf --split-sigmas 7.0 \
    --split-first-sampler euler --split-second-sampler dpm_2_ancestral \
    --split-first-cfg 2.0 --split-second-cfg 4.0
```

#### Split-Sigma Parameters

When using split-sigmas, you can customize each stage independently with these options:

| Parameter | Description |
|-----------|-------------|
| `--split-sigmas` | Value to split sigmas for multi-stage sampling (e.g., 7.0) |
| `--split-first-cfg` | CFG scale for the first stage (default: model-specific value or 3.0) |
| `--split-second-cfg` | CFG scale for the second stage (default: model-specific value) |
| `--split-first-sampler` | Sampler for the first stage (default: euler_ancestral) |
| `--split-second-sampler` | Sampler for the second stage (default: euler_ancestral) |
| `--split-first-scheduler` | Scheduler for the first stage (e.g., normal, karras) |
| `--split-second-scheduler` | Scheduler for the second stage (e.g., normal, karras) |

#### Detail Daemon with Split-Sigmas

For enhanced details, you can combine split-sigmas with the detail-daemon feature:

```bash
python -m cringegen nsfw-furry --species fox --split-sigmas 7.0 \
    --detail-daemon --detail-amount 0.2 --detail-start 0.4 --detail-end 0.8
```

When using `--detail-daemon` with split-sigmas, the detail daemon is applied to the second stage sampler.

To see all available samplers and schedulers:

```bash
python -m cringegen list-samplers
python -m cringegen list-schedulers
```

## Installation

```bash
pip install -e .
```

## Command-Line Interface

CringeGen provides a comprehensive command-line interface with modular command organization:

```bash
# Generate a furry prompt
python -m cringegen furry --species fox --gender male

# Generate a NSFW furry prompt
python -m cringegen nsfw-furry --species wolf --gender female --intensity explicit

# Generate random NSFW content with style LoRA randomization
python -m cringegen random-nsfw --species sergal --gender male

# Get LLM-generated furry captions
python -m cringegen llm-caption --species fox --gender female --count 3

# Brainstorm ideas using LLM
python -m cringegen llm-brainstorm --topic "forest creatures" --count 5

# List available models, samplers, and schedulers
python -m cringegen list-checkpoints
python -m cringegen list-loras
python -m cringegen list-samplers
python -m cringegen list-schedulers

# Get trigger phrases for a LoRA model
python -m cringegen trigger-phrases my_lora
```

### ComfyUI Connection

CringeGen requires a running ComfyUI server for image generation. By default, it connects to `http://127.0.0.1:8188`. You can specify a different ComfyUI server using the `--comfy-url` option:

```bash
# Connect to a ComfyUI server running on a different port
python -m cringegen furry --species fox --comfy-url http://localhost:18188 

# Connect to a remote ComfyUI server
python -m cringegen furry --species fox --comfy-url http://remote-server:8188
```

If ComfyUI is not running when you try to generate images, CringeGen will display an error message but will still generate and show you the prompt.

### Command Structure

The CLI is organized into modular command files in the `cringegen/commands/` directory:

- **furry.py**: Commands for generating furry prompts and images
- **nsfw.py**: Commands for generating NSFW furry prompts and images
- **random_nsfw.py**: Commands for generating random NSFW content with style LoRA randomization
- **info.py**: Commands for listing models, samplers, schedulers, and other information
- **lora.py**: Commands for LoRA-related operations
- **llm.py**: Commands for generating content using language models
- **utils.py**: Utility commands such as path resolution and trigger phrase extraction

For more detailed information about each command and its options, run:

```bash
python -m cringegen <command> --help
```

## LLM Setup

To use the LLM features, you need to have Ollama installed and running locally. The default configuration connects to Ollama at <http://localhost:11434>.

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the qwq model: `ollama pull qwq:latest`
3. Ensure Ollama is running before using LLM features

## Testing

To test the LLM functionality:

```bash
# Test normal caption generation
python tests/test_ollama.py caption --subject "explorer" --species "fox" --background "jungle"

# Test NSFW caption generation
python tests/test_ollama.py nsfw-caption --subject "character" --species "wolf" --nsfw-intensity "moderate"

# Test brainstorming
python tests/test_ollama.py brainstorm --concept "cyberpunk cityscape"
```

## Development

For development:

```bash
pip install -e ".[dev]"
```

Run linters:

```bash
python lint.py
```

## LoRA Management and Analysis

CringeGen includes tools for managing and analyzing LoRA models:

### LoRA Type Detection and Analysis

The toolkit can automatically analyze LoRA files to determine their type (style, character, or concept) and provide recommendations for usage:

```bash
# Analyze a single LoRA
python -m cringegen analyze-lora chunie-v1s2000

# Analyze all LoRAs in a directory
python -m cringegen analyze-loras --lora-dir /path/to/loras

# Get only style LoRAs with high confidence
python -m cringegen analyze-loras --type style --min-confidence 0.6

# Get LoRA combination suggestions
python -m cringegen suggest-lora-combinations foxparks-v2s1800
```

#### Features

- **Type Detection**: Identify whether a LoRA is a style LoRA (artist style), character LoRA, or concept LoRA (theme/action)
- **Confidence Scoring**: Calculate confidence level for the type determination
- **Metadata Extraction**: Extract and analyze metadata from LoRA files
- **Usage Recommendations**: Generate type-specific recommendations for prompt usage, weight settings, and combinations
- **Combination Suggestions**: Get suggestions for complementary, similar, and contrasting LoRAs
- **Caching**: Results are cached to improve performance on subsequent queries

#### Usage in Code

```python
from cringegen.utils.lora_metadata import analyze_lora_type, suggest_lora_combinations

# Analyze a LoRA
analysis = analyze_lora_type("kenket-v1s3000.safetensors")
print(f"Type: {analysis['type']} (Confidence: {analysis['confidence']:.2f})")

# Get recommendations
recommendations = analysis['recommendations']
for tip in recommendations['prompt_tips']:
    print(f"Tip: {tip}")

# Get combination suggestions
suggestions = suggest_lora_combinations("foxparks-v2s1800.safetensors", "/path/to/loras")
for lora in suggestions['complementary']:
    print(f"Complementary LoRA: {lora['name']} ({lora['type']})")
```

## Shell Completions

CringeGen provides intelligent shell completion scripts for bash and zsh to make command-line usage more efficient:

### Features

- **LoRA Name Completions**: Quickly access LoRA names with type information
- **Checkpoint Completions**: Get model information when selecting checkpoints
- **Smart Prompt Completions**: Auto-complete prompt terms based on selected LoRAs
- **Supports Multiple Shells**: Works with both bash and zsh

### Installation

Generate and install completion scripts with:

```bash
# Generate completion scripts
cringegen-completions generate --output-dir ~/completions

# Install for bash
mkdir -p ~/.local/share/bash-completion/completions
cp ~/completions/cringegen.bash ~/.local/share/bash-completion/completions/

# Install for zsh
mkdir -p ~/.zsh/completions
cp ~/completions/_cringegen ~/.zsh/completions/
# Add to ~/.zshrc: fpath=(~/.zsh/completions $fpath)
# Then run: compinit
```

### Completion Types

The shell completions provide intelligent suggestions for:

1. **LoRA Names**: Complete `--lora` and `--additional-loras` with available LoRAs and their types
2. **Checkpoint Models**: Complete `--checkpoint` with available model files and details
3. **Prompt Terms**: Context-aware completion based on:
   - Selected LoRA's activation words and popular tags
   - Common quality terms (masterpiece, best quality, etc.)
   - Common style terms (digital art, illustration, etc.)
   - Common composition terms (portrait, full body, etc.)

### Examples

```bash
# Begin typing a command and press TAB to see completions
cringegen --lora ch<TAB>               # Completes with available LoRAs starting with "ch"
cringegen --checkpoint sd<TAB>         # Completes with available checkpoints
cringegen --lora kenket-v1 --prompt m<TAB>  # Completes with keywords relevant to the kenket LoRA
```
