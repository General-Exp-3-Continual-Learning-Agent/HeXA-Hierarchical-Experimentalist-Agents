# Contributing

This project is centered on authoring and refining physics levels. The fastest way to contribute is to add or update a level definition.

## Adding a new level

1. Create a new module in `interphyre/levels/` (for example, `my_new_level.py`).
2. Define a `success_condition(engine)` function to describe the win criteria.
3. Add a `build_level(seed=None)` function that constructs objects and returns a `Level`.
4. Register the level with the `@register_level` decorator.
5. Include a short description in `metadata` so it can appear in docs.

Example structure:

```python
from interphyre.level import Level
from interphyre.levels import register_level


def success_condition(engine):
    return True


@register_level
def build_level(seed=None) -> Level:
    return Level(
        name="my_new_level",
        objects={},
        action_objects=[],
        success_condition=success_condition,
        metadata={
            "description": "Short goal statement for the level."
        },
    )
```

## Adding level previews

Level preview videos live in `docs/assets/levels/` and are linked from the Levels gallery. Add a WebM with the same base name as the module (for example, `my_new_level.webm`) and create a matching markdown page if one is missing.

## Updating docs

- `docs/levels.md` holds the gallery of one-ball levels.
- Per-level pages live in `docs/levels/` and include the goal plus source references.
- `mkdocs.yml` controls site navigation.
