# Character Template System

This directory contains individual character templates for the CringeGen prompt generation system. Each character is defined in its own file, making it easier to add, modify, and maintain character definitions.

## Adding a New Character

To add a new character:

1. Copy `template.py` to a new file named after your character (all lowercase, use underscores for spaces)
   - Example: `nick_wilde.py` for Nick Wilde from Zootopia

2. Edit the file and fill in the character's details:
   - `CHARACTER_INFO`: Metadata about the character
   - `CHARACTER_TEMPLATE`: The main template defining the character's traits
   - Optional elements like poses, outfits, and recommendations

3. Test your character with:

   ```bash
   python -m cringegen.cli character your_character_name --suggest-loras
   ```

## Required Components

Each character file must include:

- `CHARACTER_INFO`: Dictionary with character metadata
- `CHARACTER_TEMPLATE`: CharacterTemplate instance with character traits
- `__all__` list with the exported variables

## Optional Components

You can also define:

- `CHARACTER_POSES`: Dictionary of pose variations for the character
- `CHARACTER_OUTFITS`: Dictionary of clothing variations
- `RECOMMENDED_ADDITIONS`: List of recommended prompt elements
- `RECOMMENDED_LORAS`: Dictionary of LoRAs with recommended weights

## Example

See `blaidd.py` or `nick_wilde.py` for complete examples of character templates.

## Best Practices

1. Be detailed and specific with character traits
2. Include canonical elements that make the character recognizable
3. Add appropriate appearance and personality traits
4. Set accurate model tags for platforms (e621, danbooru, etc.)
5. Include references to canonical designs when possible

## Structure

Each character template should have a consistent structure:

```python
CHARACTER_INFO = {...}  # Metadata
CHARACTER_TEMPLATE = CharacterTemplate(...)  # Main template
CHARACTER_POSES = {...}  # Optional poses
CHARACTER_OUTFITS = {...}  # Optional outfits
RECOMMENDED_ADDITIONS = [...]  # Optional additions
RECOMMENDED_LORAS = {...}  # Optional LoRAs
```

## Loading Mechanism

Characters are automatically loaded when the module is imported. The system will find all `.py` files in this directory (except for `__init__.py` and `template.py`) and load the character templates from them.
