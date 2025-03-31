# CringeGen Data Structure Refactoring Plan

This document outlines a comprehensive plan to refactor and improve the data structures in the CringeGen project.

## Current Issues

- **Duplication**: Several data structures are duplicated across files (e.g., anatomy data, time of day definitions)
- **Inconsistent Structure**: Similar data uses different formats in different files
- **Poor Organization**: Related data is scattered across multiple files
- **Limited Extensibility**: Current global dictionary approach makes it hard to extend or specialize data
- **Lack of Validation**: No systematic way to check for redundancies or inconsistencies

## Refactoring Plan

### 1. Anatomy Data Consolidation

**Issue**: `MALE_ANATOMY` and `FEMALE_ANATOMY` are duplicated in both `taxonomy.py` and `anatomy.py`.

**Tasks**:

- [x] Move all anatomical data to `anatomy.py`
- [x] Normalize data structures (dict of dicts vs lists vs tuples)
- [x] Create helper functions for easier data access
- [x] Update imports in `__init__.py`

### 2. Environment and Background Consolidation

**Issue**: `TIME_OF_DAY`, `WEATHER_CONDITIONS`, and other environmental data are duplicated with different structures.

**Tasks**:

- [x] Create a new `environments.py` file
- [x] Move all time of day data to the new file with a unified structure
- [x] Move all weather conditions to the new file
- [x] Move all background settings to the new file
- [x] Create consistent nested dictionary structures
- [x] Add habitat information by species
- [x] Update imports in `__init__.py`

### 3. Style System Reorganization

**Issue**: Art style data is overly complex and has redundancies.

**Tasks**:

- [x] Create new style_taxonomy.py with hierarchical structure
- [x] Organize styles by medium, period, technique, etc.
- [x] Add relationship data between styles
- [x] Add style attributes (mood, technical qualities, etc.)
- [x] Implement helper functions for style selection and combination
- [x] Update imports in `__init__.py`
- [x] Create compatibility with existing style-using code

### 4. LoRA System Improvements

**Issue**: LoRA data lacks organization, proper metadata, and intelligent selection mechanisms.

**Tasks**:

- [x] Create new lora_taxonomy.py with hierarchical structure
- [x] Organize LoRAs into category-based structures (style, character, concept, etc.)
- [x] Add comprehensive metadata for each LoRA (creator, version, strength range, etc.)
- [x] Implement version tracking for LoRAs
- [x] Create compatibility frameworks for different model versions
- [x] Add intelligent selection and combination algorithms
- [x] Implement utilities in lora_utils.py for working with the taxonomy
- [x] Add backward compatibility with legacy LoRA data
- [x] Update imports in `__init__.py`

### 5. Character System Reorganization

**Issue**: Character data is scattered and lacks cohesion.

**Tasks**:

- [x] Create character_taxonomy.py with a comprehensive structure
- [x] Organize character elements (species, clothing, accessories, etc.)
- [x] Add relationship data between character elements
- [x] Implement character generation algorithms
- [x] Create utility functions for character manipulation
- [x] Update imports in `__init__.py`
- [x] Add model-specific tags for popular characters (e621, danbooru)
- [x] Create CLI command for generating character-specific prompts
- [ ] Separate canonical characters into individual files

### 6. Character Data Separation

**Issue**: Notable character data is all in one file, making it hard to maintain as more characters are added.

**Tasks**:

- [ ] Create a characters directory structure for character-specific data
- [ ] Move Blaidd and other notable character data to individual files
- [ ] Create a standardized character data format
- [ ] Implement automatic loading of character files
- [ ] Update CLI to reflect new character organization
- [ ] Add documentation for adding new character templates

### 7. Data Validation System

**Issue**: No systematic way to validate data consistency or identify redundancies.

**Tasks**:

- [ ] Implement validation functions for each data category
- [ ] Create tests to check for duplicate entries
- [ ] Add consistency checks for naming conventions
- [ ] Implement reference validation (ensure all references point to valid entries)
- [ ] Create a data integrity test suite

### 8. Centralized Configuration

**Issue**: Configuration values are scattered throughout data files.

**Tasks**:

- [ ] Create a dedicated `config.py` file
- [ ] Move configuration parameters from data files
- [ ] Implement hierarchical configuration structure
- [ ] Update data structures to use centralized configuration

### 9. Enhanced Data Relationships

**Issue**: Relationships between data structures are implicit and hard to trace.

**Tasks**:

- [ ] Define explicit relationships between data structures
- [ ] Create mapper functions to navigate between related data
- [ ] Implement weighted relationships for generating coherent combinations
- [ ] Add context-aware data selection helpers

### 10. Modular Class-Based System

**Issue**: Current system uses large global dictionaries, making it hard to extend or specialize.

**Tasks**:

- [ ] Design a class-based system for core data types
- [ ] Create base classes with shared attributes
- [ ] Implement class hierarchy for taxonomy, species, etc.
- [ ] Add validation methods to classes
- [ ] Create a transition plan from dictionaries to class instances

## Implementation Order

1. ✅ Anatomy Consolidation (simplest change)
2. ✅ Environment and Background Consolidation
3. ✅ Style System Reorganization
4. ✅ LoRA System Improvements
5. ✅ Character System Reorganization
6. Character Data Separation
7. Data Validation System
8. Centralized Configuration
9. Enhanced Data Relationships
10. Modular Class-Based System (most complex change)

## Testing Strategy

For each step:

1. Create unit tests for the existing behavior
2. Implement the changes
3. Verify the tests still pass
4. Add new tests for enhanced functionality

## Documentation

- Update README.md after each major refactoring step
- Create a data structure overview document
- Document the class hierarchy and relationships
- Add examples of how to use the new systems
