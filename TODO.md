# TODO List for CringeGen

A list of features and improvements planned for future versions.

- [x] Refactor data structures in ./cringegen/data for better organization
  - [x] Create a unified taxonomy system (consolidate species_data.py with overlapping data in other files)
  - [x] Centralize character data (merge overlapping entries from furry_characters.py, anime_data.py, etc.)
  - [x] Create a unified color system (consolidate color_data.py with color information in other files)
  - [x] Standardize LoRA-related data structures (consolidate lora_data.py and lora_filtering.py)
  - [ ] Create clear documentation for each data structure and its purpose
  - [ ] Add type hints to all data structures
  - [x] Update all imports and references in the codebase to use the new structure
  - [x] Create comprehensive tests to validate the data structure integrity

- [ ] Improve bidirectional conversion for higher tag recovery rate (currently 35-50%)
  - [ ] Add advanced text parsing for complex descriptors
  - [ ] Implement context-aware tag extraction
  - [ ] Create specialized extractors for artistic terminology
- [ ] Enhance handling of compound descriptors
  - [ ] Add pattern recognition for compound terms like "ember-glowing scales"
  - [ ] Create specialized parsers for color descriptors with modifiers
- [ ] Improve semantic categorization
  - [ ] Add more specialized categories for artistic domains
  - [ ] Create hierarchical tag classification
- [ ] Expand anime character archetype detection
  - [ ] Add recognition for specific anime character types (tsundere, yandere, etc.)
  - [ ] Implement detection for anime-specific visual elements
- [ ] Improve compound entity parsing
  - [ ] Create multi-word entity detection with attribute binding
  - [ ] Add contextual disambiguation for similar entities
- [ ] Expand fantasy species recognition
  - [ ] Add more comprehensive fantasy taxonomy
  - [ ] Implement mythology-specific entity detection
- [ ] Add even more characters and species to every database
- [ ] and yeah in general extend all parts of the database where it makes sense
- [ ] Add domain-specific enhancement suggestions
  - [ ] Create style-specific suggestion engines
  - [ ] Implement genre-aware enhancement options
- [ ] Improve structure detection for unusual formats
  - [ ] Add parsers for non-standard prompt formats
  - [ ] Create format normalization preprocessing
- [ ] Add style-specific enhancement options
  - [ ] Create anime/manga-specific enhancers
  - [ ] Add furry/anthro-specific enhancement options
- [ ] Enhance semantic relationship extraction for domain-specific terms
  - [ ] Create custom WordNet extensions for artistic domains
  - [ ] Implement context-aware synonym generation
- [ ] Improve phrase extraction for complex descriptions
  - [ ] Add specialized chunkers for artistic descriptions
  - [ ] Create domain-specific part-of-speech taggers
- [ ] Add specialized linguistic analysis for art
  - [ ] Implement style-specific analyzers
  - [ ] Create sentiment analysis calibrated for artistic descriptions
- [ ] Add more customization options
  - [ ] Create fine-grained control for stylistic elements
  - [ ] Add personality trait injection
  - [ ] Implement mood-based generation options
- [ ] Improve specialized character archetype handling
  - [ ] Add support for common anime/furry archetypes
  - [ ] Create consistent character attribute binding
- [ ] Enhance integration with other NLP components
  - [ ] Create bidirectional workflows between generators and analyzers
  - [ ] Implement feedback loops for iterative enhancement
- [ ] Create comprehensive test suite for all NLP components
  - [ ] Add benchmarking for performance metrics
  - [ ] Create reproducible test cases for each feature
- [ ] Improve documentation with detailed examples
  - [ ] Create a detailed NLP components guide
  - [ ] Add visual documentation for complex workflows
  - [ ] Create interactive demos for key features
- [x] Implement unique fursona generation mode
  - [x] Add support for both SFW and NSFW fursona generation
  - [x] Create robust artistic style specification options
  - [x] Add support for color and pattern customization
  - [ ] Implement personality trait generation for fursonas
  - [ ] Add backstory generation for fursona characters
  - [ ] Create consistent character generation across multiple images
  - [ ] Implement fursona profile card generation
  - [ ] Add save/load functionality for favorite fursonas
- [ ] Add performance optimization and caching for LLM requests.
  - [x] Implement local caching of samplers and schedulers
  - [ ] Add batching support for multiple LLM requests
  - [ ] Create fallback mechanisms for offline operation
- [ ] Implement advanced image generation and manipulation
  - [ ] Implement image editing and post-processing features
  - [ ] Add facial and paw detail enhancement via detailed inpainting
- [ ] Implement XY plot functionality for parameter experimentation
  - [x] Create a new command for XY plot generation
  - [x] Implement grid-based image generation with variable parameters
  - [x] Support varying different parameters on X and Y axes:
    - [x] Checkpoint models
    - [x] LoRA models with variable weights
    - [x] Sampler types
    - [x] Scheduler types
    - [x] CFG scale values
    - [x] Step counts
    - [x] LoRA weights (model and clip strength)
    - [x] Seeds
    - [x] Prompt variations
  - [x] Add support for plotting multiple prompts in a single grid
  - [x] Create a grid output format with proper labeling of axes
  - [x] Implement image grid viewing capabilities with optional remote downloading via SSH
    - [x] Add automatic opening of generated grid images with imv
  - [x] Implement support for sharing and exporting XY plot configurations
  - [x] Add support for model activation keywords when varying models
  - [x] Leverage imx Rust library for powerful image grid plotting
- [ ] Add command for LoRA compatibility analysis
- [ ] Add command for batch prompt generation
- [ ] Add command for model-specific optimizations
- [ ] Improve error handling and user feedback
- [ ] Create specialized workflows for different generation scenarios
- [ ] Add more configuration options for workflows
- [ ] Implement workflow export/import
- [ ] Add multi-step generation workflows
- [ ] Add NLP features documentation to README.md
- [ ] Add comprehensive documentation for all modules
- [ ] Create examples for each feature
- [ ] Add API documentation
- [ ] Create a user guide
- [ ] Add tutorials for common use cases
  - [x] Create NLP features demo and test scripts
  - [ ] Add tutorial for tag conversion and categorization
  - [ ] Add tutorial for prompt analysis and enhancement
  - [ ] Add tutorial for NLTK integration usage
- [x] Create and test NLP utilities demo script
- [x] Implement comprehensive NLP features test script
- [ ] Add unit tests for all components
- [ ] Create integration tests
- [ ] Add automated testing
- [ ] Implement a test suite for workflow generation
- [ ] Add support for more model types
- [ ] Implement a simple web UI for prompt generation
- [ ] Add support for custom LoRA handling
- [ ] Create model-specific templates
- [ ] Add multi-language support

## XY Plot Implementation Strategy

The XY plot functionality will enable users to experiment with different parameters and visualize their impact on generated images. The implementation will leverage the `imx` Rust library for powerful image grid generation and formatting capabilities.

1. **Core Infrastructure**
   - Create a new module `cringegen/commands/xyplot.py` for the XY plot command
   - Implement a grid-based generation system that varies two parameters independently
   - Design a flexible parameter specification system that can handle different data types (strings, numbers, arrays)
   - Create Rust bindings to utilize the `imx` library's image grid functionality
   - Implement efficient image handling with support for various formats (JPEG, PNG, WebP, JXL)

2. **Parameter Variation Support**
   - Implement handlers for each parameter type:
     - Model handler: loads different checkpoint models
     - LoRA handler: loads different LoRA models with configurable weights
     - Sampler/Scheduler handler: switches between different sampling methods
     - Numeric parameter handler: varies numeric values (CFG, steps, weights)
     - Seed handler: manages controlled seed variation
     - Prompt variation handler: supports text substitution or modification

3. **Grid Generation Pipeline**
   - Create a job scheduling system to manage multiple generation tasks
   - Implement progress tracking for long-running grid generations
   - Build output filename and directory structure conventions
   - Add support for resuming interrupted grid generations
   - Use `imx` for efficient image processing and layout generation

4. **Output Visualization with imx**
   - Utilize `imx`'s `create_plot` function for generating professional grid layouts
   - Implement rich label formatting with proper alignment options (start, center, end)
   - Support multiline text in grid labels for detailed parameter descriptions
   - Configure label padding and alignment for optimal readability
   - Leverage `imx`'s automatic image scaling and alignment capabilities
   - Support debug visualization mode for layout troubleshooting

5. **User Interface**
   - Design a command-line interface with intuitive parameter specification
   - Support for configuration files to define complex grid experiments
   - Add interactive mode for exploring results and refining parameters
   - Create visualization of parameter space before generation
   - Implement customizable label formatting options

6. **Integration**
   - Connect to existing ComfyUI workflows for actual image generation
   - Reuse remote/SSH functionality for distributed processing
   - Integrate with existing image viewing capabilities (--show flag) and the `--remote` flag for copying the grid over SSH
   - Create a Rust-Python bridge to efficiently leverage `imx` capabilities
   - Ensure compatibility with all existing generation modes (furry, nsfw, etc.)
   - Implement JXL format support for advanced users

7. **Advanced Features**
   - Create export options for sharing experiments
   - Support for animation between grid points
   - Add model keyword trigger detection and automatic application
   - Implement safe numeric conversions for image data processing
   - Utilize `imx`'s layout debugging capabilities for development and troubleshooting
   - Support Unicode and emoji in grid labels for expressive parameter descriptions
   - Add letterboxing removal and transparency handling options

This functionality will help users:

- Compare model performance across different conditions
- Find optimal parameter values for specific styles or prompts
- Understand the impact of different samplers, schedulers, and other parameters
- Create consistent image series by controlling parameter variations
- Share and document their experimental processes and findings

- [ ] Standardize interfaces for all prompt generation components
  - [ ] Define clear APIs for all generator types
  - [ ] Implement consistent parameter handling across generators
  - [ ] Add proper documentation for all interfaces
