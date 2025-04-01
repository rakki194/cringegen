# cringegen Data Structures

This directory contains all the data structures used by cringegen for generating anthro/furry art prompts. Each file is organized around a specific domain of data needed for the generation process.

## Overview

The data module is organized into themed files, each containing specific data structures:

- **Taxonomy Data**: Species classifications, body coverings, and anatomical references
- **Color System**: Color palettes, patterns, and species-specific coloration
- **Character Data**: Famous characters, character types, and defining features
- **Art Styles**: Visual art styles, media-specific styles, and rendering techniques
- **Settings & Environments**: Locations, time of day, weather, and atmosphere descriptors
- **LoRA Management**: Classification and management of LoRA models for stable diffusion
- **Anatomy Data**: Species-specific anatomical terms and references
- **Accessories**: Character accessories and clothing items by species
- **Habitats**: Species-specific environmental contexts

## File Structure and Data Types

### `__init__.py`

Acts as the central import hub, making all data structures available through a unified interface. Uses explicit imports and defines an `__all__` list for better organization and access control.

### `taxonomy.py`

Contains hierarchical classification systems for species:

- `SPECIES_TAXONOMY`: Maps species names to taxonomic groups
- `BODY_COVERING_BY_TAXONOMY`: Associates taxonomic groups with appropriate body covering terms (fur, scales, feathers)
- `MALE_ANATOMY` / `FEMALE_ANATOMY`: Anatomical terms organized by taxonomy
- `ANTHRO_SPECIES`: Comprehensive list of anthropomorphic species
- `POPULAR_ANTHRO_SPECIES`: Most commonly used species
- `FANTASY_SPECIES`: Non-Earth species (dragons, sergals, protogens)
- `TAXONOMY_GROUPS`: Higher-level classification of taxonomic groups
- `ANTHRO_DESCRIPTOR_SET`: Text descriptors for anthropomorphic characters

**Usage**: Provides the foundation for species selection and appropriate attribute assignment.

### `colors.py`

Manages color systems and patterns:

- `BASIC_COLORS`: Fundamental color names
- `EXPANDED_COLORS`: Detailed color variations organized by base color
- `COLOR_PATTERNS`: Pattern types by body covering (fur patterns, scale patterns, etc.)
- `COLOR_DISTRIBUTIONS`: Templates for describing color distributions
- `SPECIES_SPECIFIC_PATTERNS`: Special color patterns for specific species
- `COMMON_MARKINGS`: Common anatomical markings by species
- `COLOR_DESCRIPTION_TEMPLATES`: Templates for generating natural-sounding color descriptions
- `SPECIES_COLORS`: Natural or common colors associated with specific species
- `COLOR_PALETTES`: Predefined color combinations

**Usage**: Enables creation of realistic and species-appropriate color descriptions.

### `characters.py`

Contains character-related data:

- `FAMOUS_FURRY_CHARACTERS`: Well-known characters organized by media source
- `CHARACTER_TO_FULL_NAME`: Mapping of character short names to full names
- `ALL_ANTHRO_CHARACTERS`: Comprehensive list of anthropomorphic characters
- `ANIME_CHARACTER_TYPES`: Character archetypes from anime
- `ANIME_CHARACTER_FEATURES`: Distinctive visual features of anime characters
- `GAME_CHARACTER_TYPES`: Character archetypes from gaming

**Usage**: Provides reference data for character-based prompts and recognizable entities.

### `settings.py`

Contains environmental and contextual data:

- `BACKGROUND_SETTINGS`: Detailed descriptors for different environments
- `TIME_OF_DAY`: Lighting and atmosphere descriptors for different times
- `WEATHER_CONDITIONS`: Weather features and associated atmospheres
- `SEASONS`: Seasonal descriptors and features
- `MOOD_DESCRIPTORS`: Emotional and atmospheric qualities
- `SPECIES_HABITATS`: Natural habitats associated with specific species

**Usage**: Creates contextually appropriate settings and atmospheres for characters.

### `lora.py`

Manages LoRA (Low-Rank Adaptation) models for Stable Diffusion:

- `STYLE_KEYWORDS`: Keywords identifying style LoRAs
- `CHARACTER_KEYWORDS`: Keywords identifying character LoRAs
- `KINK_KEYWORDS`: Keywords identifying kink-related LoRAs
- `CONCEPT_KEYWORDS`: Keywords identifying concept LoRAs
- `SPECIFIC_KINK_LORAS`/`SPECIFIC_CHARACTER_LORAS`: Named LoRA models
- Various pattern dictionaries for identifying LoRA types
- `EXCLUDED_LORAS`: LoRAs to avoid in generation
- `ARTIST_SELECTION_CONFIG`: Configuration for artist selection
- `LORA_STRENGTH_RANGES`: Appropriate strength ranges for different LoRA types
- `KINK_LORA_RECOMMENDATIONS`: Recommended LoRAs for specific scenarios

**Usage**: Allows intelligent selection and configuration of LoRA models.

### `styles.py`

Comprehensive art style data:

- `ART_STYLES`: Categorized art styles (traditional, digital, fine art movements)
- `FILM_TV_STYLES`: Visual styles from film and television
- `GAME_ART_STYLES`: Visual styles from video games
- `ANIME_ART_STYLES`: Styles from anime and manga
- `STYLE_DESCRIPTORS`: Terms describing stylistic qualities
- `RENDERING_DESCRIPTORS`: Technical aspects of rendering
- `STYLE_MODIFIERS`: Terms that modify artistic styles

**Usage**: Provides vocabulary for detailed artistic style specifications.

### `anatomy.py`

Contains anatomical reference terms:

- `MALE_ANATOMY`: Male anatomical terms by taxonomy group
- `FEMALE_ANATOMY`: Female anatomical terms by taxonomy group

**Usage**: Provides appropriate anatomical references for different species.

### `accessories.py`

Character accessories and clothing:

- `SPECIES_ACCESSORIES`: Accessories and clothing items organized by species, form type (anthro/feral), and gender

**Usage**: Helps generate appropriate accessory descriptions for different character types.

### `habitats.py`

Contains environmental context data:

- `SPECIES_HABITATS`: Maps species to their natural or common habitats

**Usage**: Helps place characters in contextually appropriate environments.

### `backgrounds.py`

Detailed background and setting information:

- `TIME_OF_DAY`: Time periods with lighting and atmospheric qualities
- `WEATHER_CONDITIONS`: Weather types with features and atmosphere
- `SEASONS`: Seasonal descriptors and qualities
- Additional detailed background descriptors

**Usage**: Provides rich environmental context for character scenes.

## Data Structure Relationships

The data structures are designed to work together:

1. **Species → Taxonomy → Body Covering**: Species selection determines taxonomy, which determines appropriate body covering terms
2. **Species → Habitats → Backgrounds**: Species influence appropriate habitats, which influence background selection
3. **Color Patterns + Species → Color Descriptions**: Color pattern selection is influenced by species-specific patterns
4. **Art Style + Character Type → Rendering Approach**: Art style selection influences appropriate character rendering
5. **LoRA Selection + Style/Character**: LoRA models are selected based on style and character requirements

## Usage Examples

- **Character Generation**: Combine species, color patterns, and accessories
- **Setting Generation**: Use species habitats, time of day, and weather conditions
- **Style Application**: Combine art styles with rendering descriptors
- **LoRA Selection**: Filter and select appropriate LoRA models for given prompts

## Extending the Data

When adding new entries:

1. Follow the existing data structure patterns
2. Ensure consistency in naming conventions
3. Update the `__init__.py` imports if adding new dictionaries
4. Consider cross-references between different data structures

## Data Structure Design Principles

The data is organized with these principles:

- **Hierarchical Organization**: Data is structured in hierarchies (taxonomy → species)
- **Separation of Concerns**: Different types of data in separate modules
- **Comprehensive Coverage**: Wide range of options within each category
- **Natural Language Orientation**: Structured to facilitate natural language generation
- **Extensibility**: Easy to add new entries to existing structures
