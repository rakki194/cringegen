## NoobAI Prompting Guide for Furry Art Generation

NoobAI is an exceptional model for generating furry artwork, having been trained on extensive datasets including e621 (approximately 13 million images). This guide will help you create the most effective prompts for NoobAI models.

### Optimal Technical Parameters

For best results with NoobAI models, use these parameters:

- **Sampler**: Euler (not Euler A) for v-prediction models
- **CFG Value**: 4.0-5.0 (sweet spot for detail vs. creativity)
- **Steps**: 28-35 for optimal detail and convergence
- **Resolution**: Choose from: 768x1344, 832x1216, 896x1152, 1024x1024, 1152x896, 1216x832, 1344x768

### Prompt Structure Best Practices

Organize your prompts in this logical order:

```
<quality tags>, <character count>, <species>, <character details>, <artist>, <general tags>, <scene/environment>, <action>, <expression>
```

For example:

```
masterpiece, best quality, newest, absurdres, highres, anthro, 1girl, fox, red fur, blue eyes, [artist name], detailed fur, forest, standing, happy expression
```

### Key Quality Tags

Always begin your prompt with these quality tags for NoobAI models:

```
masterpiece, best quality, newest, absurdres, highres
```

For enhanced aesthetics, consider adding `very awa` which instructs the model to use its top 5% aesthetic understanding.

### Character Definition Essentials

Start character descriptions with:

- **Character count**: Use `1girl`, `1boy`, `1male`, `1female`, `solo` for single characters
- **Species**: For furry content, add `anthro` followed by the species (e.g., `fox`, `wolf`, `dragon`)
- **Description tags**: Add details like `red fur`, `blue eyes`, `muscular build`

### Artist Style Influence

Include artist names directly without prefixes:

- Correct: `miles-df, fluff-kevlar, kenket`
- Incorrect: ~~`by miles-df`, `artist:miles-df`~~

For stronger style influence, use bracket emphasis:

- Moderate influence: `[artist]`
- Strong influence: `[[artist]]`
- Very strong: `[[[artist]]]`

### E621 Tag Compatibility

NoobAI has excellent understanding of e621 tags. When using these tags:

- No underscores needed (unlike some other models)
- Use commas to separate tags
- For character names with series, use: `character_name (series)` (e.g., `krystal (star_fox)`)

### Species-Specific Anatomical Details

NoobAI understands detailed anatomical terms for different species. Use these to enhance your prompts, especially for NSFW content:

- **Canine-specific**: `canine penis`, `knot`, `sheath` for males; `canine pussy` for females
- **Feline-specific**: `feline penis`, `barbed penis` for males; `feline pussy` for females
- **Equine-specific**: `equine penis`, `flared penis` for males; `equine pussy` for females
- **Reptilian-specific**: `hemipenes`, `genital slit` for males; `cloaca` for females
- **Avian-specific**: `cloaca` for both genders

For best results with anatomical details:

1. Use species-appropriate terms (e.g., "knot" for wolves, foxes, dogs)
2. Place anatomical terms after character description but before pose/action
3. For SFW content, avoid specific genital terms and use general anatomy instead

### Taxonomical Tag Organization

NoobAI recognizes taxonomical groupings, which can enhance prompt coherence:

- **Canine**: Wolves, foxes, dogs, coyotes
- **Feline**: Cats, lions, tigers, leopards
- **Equine**: Horses, ponies, zebras, donkeys
- **Bovine**: Cows, bulls, bison, buffalo
- **Cervid**: Deer, elk, moose, reindeer
- **Rodent**: Mice, rats, squirrels, beavers
- **Lagomorph**: Rabbits, hares, bunnies
- **Reptile**: Dragons, lizards, snakes, dinosaurs
- **Avian**: Birds, gryphons, avians

Include the taxonomical group along with the species for better results:

```
anthro, canine, wolf, male
```

### Useful Tag Categories for Furry Art

- **Species modifiers**: `anthro`, `feral`, `semi-anthro`
- **Body types**: `muscular`, `chubby`, `skinny`, `athletic`
- **Expressions**: `smile`, `angry`, `sad`, `excited`, `seductive`
- **Poses**: `standing`, `sitting`, `lying`, `crouching`, `action_pose`
- **Clothing**: `jacket`, `shirt`, `pants`, `underwear`, `nude`, `swimsuit`
- **Environments**: `forest`, `bedroom`, `city`, `beach`, `mountains`
- **Lighting/mood**: `sunset`, `dark`, `bright`, `moody`, `cheerful`

### Example Prompts for NoobAI Furry Generation

#### Basic Anthro Character

```
masterpiece, best quality, newest, absurdres, highres, anthro, 1male, wolf, gray fur, blue eyes, casual clothes, forest background, standing, slight smile
```

#### Stylized Character with Artist Influence

```
masterpiece, best quality, newest, absurdres, highres, anthro, 1female, fox, red fur, white muzzle, green eyes, [[miles-df]], [[kenket]], detailed fur texture, outdoor setting, dynamic pose
```

#### Complex Scene with Multiple Characters

```
masterpiece, best quality, newest, absurdres, highres, anthro, 2girls, fox, wolf, different species, casual interaction, cafe setting, sitting, table, coffee cups, urban background, daylight, cheerful expressions
```

#### NSFW Character with Anatomical Detail

```
masterpiece, best quality, newest, absurdres, highres, anthro, 1male, canine, wolf, gray fur, muscular, presenting, canine penis, knot, bedroom setting, seductive pose, subtle lighting
```

### Effective Negative Prompts

Always include these in your negative prompts to avoid common issues:

```
nsfw, worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, mutated hands, mutated fingers, extra fingers, fused fingers, too many fingers, missing fingers, missing arms, misshapen body, signature, watermark, username
```

For specifically non-furry results, also add:

```
mammal, anthro, furry, ambiguous form, feral, semi-anthro
```

### Advanced Tips

1. **Weight adjustments**: Use colons for emphasis (e.g., `red fur:1.2` for stronger influence)
2. **Date tags**: Add `year 2024` or `newest` for modern aesthetic styles
3. **Background enhancement**: Add `scenery porn, amazing background` for detailed environments
4. **Dynamic composition**: Add `dynamic angle, depth of field` for more interesting compositions
5. **Use meaningful suffixes**: For V-Pred models, add `colorful, detailed light` for better lighting
6. **Species-appropriate accessories**: Include anatomically accurate accessories for your species (e.g., `collar` for canines, `ear rings` for felines)

### Using the Automated Tags System

The cringegen toolkit includes a sophisticated system that can automatically generate species-specific tag sets:

```bash
# Generate species-specific tags for a dragon character
python -m cringegen llm-nsfw-caption --species dragon --gender female --tags-only

# Generate a full NSFW prompt with anatomically correct tags
python -m cringegen llm-nsfw-caption --species wolf --gender male --nsfw-intensity explicit
```

The automated system will:

1. Select taxonomically appropriate terms for the species
2. Add proper anatomical details based on gender and species
3. Include appropriate accessories and physical characteristics
4. Maintain e621 tag compatibility for optimal results

By following these guidelines, you'll maximize NoobAI's furry art generation capabilities, creating detailed and visually appealing anthro characters with accurate anatomy and high aesthetic quality.
