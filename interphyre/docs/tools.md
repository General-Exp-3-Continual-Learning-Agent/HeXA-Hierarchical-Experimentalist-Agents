# Tools

Interphyre provides several command-line tools for development, testing, and data collection.

## Viewer

Interactive level visualization and demonstration tool.

### Basic Usage

```bash
# View a level with specific action
python -m interphyre.viewer catapult --seed 42 --action 0.5 3.0 0.6

# Run random agent demo
python -m interphyre.viewer --demo catapult --trials 10 --seed 42

# View solutions from file
python -m interphyre.viewer catapult --seed 42 --solutions successes.json

# Record video
python -m interphyre.viewer catapult --seed 42 --action 0.5 3.0 0.6 --record --format mp4
```

### Parameters

- `--level` - Level name to visualize
- `--seed` - Random seed for level generation
- `--action` - Action to apply (x, y, radius)
- `--demo` - Run random agent demo mode
- `--solutions` - Path to solutions JSON file
- `--trials` - Maximum number of trials (demo mode)
- `--record` / `--record-video` - Record video output
- `--format` / `--video-format` - Video format (mp4 or gif)
- `--pause` - Pause duration in seconds

## Data Collection

Collect training data using CEM or random agents.

### Usage

```bash
python tools/collect_data.py \
    --level catapult \
    --seeds "42,69,123" \
    --output-dir data/catapult \
    --max-attempts 1000 \
    --agent cem
```

### Parameters

- `--level` - Level name
- `--seeds` - Comma-separated list of seeds
- `--output-dir` - Directory to save results
- `--max-attempts` - Maximum solution attempts per seed
- `--agent` - Agent type (cem or random)
- `--workers` - Number of parallel workers

### Output Files

- `successes.json` - Successful solutions with actions
- `failures.json` - Failed attempts for analysis

## Benchmarking

Benchmark random agent performance on levels.

### Usage

```bash
python tools/benchmark_random_agent.py \
    --level catapult \
    --trials 1000 \
    --seeds "42,69,123"
```

### Output

- Success rate statistics
- Performance metrics
- Action distribution analysis

## Best Practices

1. **Use consistent seeds** - For reproducible results across experiments
2. **Set appropriate max-attempts** - Balance between data quality and collection time
3. **Use CEM agent for difficult levels** - Random agent may not find solutions
4. **Filter problematic seeds** - Some levels have inherent difficulty variability

## Troubleshooting

**"Level not found" error**
- Ensure level name is correct (use `list_levels()` to see available levels)
- Check that level module is imported in `interphyre/levels/__init__.py`

**Solutions not loading**
- Verify JSON file path is correct
- Check JSON format matches expected structure (dict with "seed" and "action" keys)

**Video recording fails**
- Ensure ffmpeg is installed for MP4 output
- Try GIF format if MP4 encoding fails
- Check output directory has write permissions
