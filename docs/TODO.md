# TODO: Future Cringegen Enhancements

## Optimal Resolution Check Feature

### Overview

Implement an optional feature to check and warn users when image dimensions don't match the optimal values for different model architectures.

### Requirements

- Add a configuration option to enable/disable resolution checks
- Check optimal resolutions based on model architecture:
  - SD 1.5 models: optimal resolution is 512×512 (262,144 pixels)
  - SDXL models: optimal resolution is 1024×1024 (1,048,576 pixels)
- Support various aspect ratios that maintain the same pixel count
- Show clear warnings when dimensions aren't optimal

### Implementation Plan

1. **Configuration Options**
   - Add `--check-resolution` flag to CLI commands
   - Add `check_resolution: bool` to configuration file
   - Default to `False` to avoid spamming warnings for power users

2. **Model Architecture Detection**
   - Enhance the existing `is_sdxl_model()` function
   - Add a function to detect SD 1.5 models
   - Create a generic `get_model_architecture(checkpoint_name)` function that returns "sdxl", "sd15", or "other"

3. **Resolution Validation**
   - For SD 1.5: Validate if width × height ≈ 262,144 pixels (allowing a small margin)
   - For SDXL: Validate if width × height ≈ 1,048,576 pixels (allowing a small margin)
   - Support common aspect ratios that maintain similar pixel counts:
     - SD 1.5: 512×512, 448×576, 576×448, 384×640, 640×384, etc.
     - SDXL: 1024×1024, 896×1152, 1152×896, 768×1344, 1344×768, etc.

4. **Warning System**
   - Enhance the existing colored warning system
   - Show clear suggestions for better resolutions
   - Provide explanation why optimal resolutions matter for each model

5. **Documentation**
   - Update user documentation to explain the feature
   - Add examples of optimal resolutions for different use cases

### Suggested Message Format

```
WARNING: Non-optimal resolution for SDXL model (768×768 = 589,824 pixels).
         Optimal pixel count is 1,048,576. Consider using 1024×1024 or equivalent.
```

### Priority: Medium

This feature will help users, especially beginners, get better results by using recommended image dimensions for their specific model architecture.

## Model Classification System

### Overview

Implement a comprehensive model classification system that can detect both the base architecture and the specific model family from checkpoint names.

### Requirements

- Detect model architecture (SD1.5, SDXL, SD3.5, Flux, LTX, Lumina, etc.)
- Detect model family/series (noob, yiffymix, dreamshaper, epicrealism, etc.)
- Create a database of model-specific prompt prefixes and settings
- Integrate with existing workflows to apply model-specific optimizations

### Implementation Plan

1. **Model Detection**
   - Enhance the newly implemented `get_model_info()` function
   - Add pattern recognition for more model types
   - Implement test coverage with test cases for all supported models
   - Create fallback mechanisms for unknown models

2. **Model-Specific Optimizations**
   - Create a database/dictionary of model-specific prompt prefixes
   - Add model-specific default generation parameters (steps, CFG, sampler)
   - Document the optimal settings for each model family

3. **Prompt Tag Injection**
   - Implement automatic prompt prefix/tag injection based on model family
   - Allow users to disable automatic tag injection
   - Create a system to document which tags work best with which models

4. **Workflow Integration**
   - Integrate model detection with generation workflows
   - Automatically select optimal workflow based on model architecture
   - Add model-specific UI elements or warnings when appropriate

5. **Documentation**
   - Document all supported models and their detection patterns
   - Create a user guide explaining how the model classification system works
   - Add examples of model-specific prompt prefixes and their effects

### Example Model Classification Dictionary

```python
MODEL_PREFIXES = {
    "noob": "masterpiece, best quality, realistic, photorealistic,",
    "yiffymix": "masterpiece, best quality, highly detailed,",
    "epicrealism": "RAW, analog, nikon, film grain,",
    # etc.
}
```

### Priority: High

This feature will significantly improve generation quality by tailoring prompts and parameters to specific model architectures and families. It will also simplify user experience by reducing the need for manual prompt engineering.
