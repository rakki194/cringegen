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
