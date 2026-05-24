# Level Registry

The registry lives in `interphyre/levels/__init__.py` and provides the decorator used by level modules.

## register_level

`register_level` registers a level builder in the global registry by calling the function once to read its `Level.name`.

```python
from interphyre.levels import register_level

@register_level
def build_level(seed=None):
    ...
```

## load_level

`load_level(name, seed=None)` dynamically imports the module if needed and returns a built `Level`.

```python
from interphyre.levels import load_level
level = load_level("two_body_problem", seed=42)
```

Notes:

- Level modules should define `build_level(seed=None)` and return a `Level`.
- The registry uses the `Level.name` field as the lookup key.
