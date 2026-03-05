# weighted-levenshtein

## Use Cases

Most Levenshtein libraries assign a cost of 1 to every edit operation. This library lets you assign different costs to insertions, deletions, substitutions, and transpositions for any combination of characters. The core algorithms are implemented in Rust via [PyO3](https://pyo3.rs/) for performance.

For example, in OCR correction, substituting `0` for `O` should cost less than substituting `X` for `O`. In typo correction, substituting `X` for `Z` should cost less since they are adjacent on a QWERTY keyboard.

**Levenshtein** supports per-character insertion, deletion, and substitution costs.

**Damerau-Levenshtein** additionally supports transposition costs.

## Installation

```
pip install weighted-levenshtein
```

## Usage

```python
import numpy as np
from weighted_levenshtein import lev, osa, dam_lev

insert_costs = np.ones(128, dtype=np.float64)
insert_costs[ord('D')] = 1.5  # inserting 'D' costs 1.5

print(lev('BANANAS', 'BANDANAS', insert_costs=insert_costs))  # 1.5

delete_costs = np.ones(128, dtype=np.float64)
delete_costs[ord('S')] = 0.5  # deleting 'S' costs 0.5

print(lev('BANANAS', 'BANANA', delete_costs=delete_costs))  # 0.5

substitute_costs = np.ones((128, 128), dtype=np.float64)
substitute_costs[ord('H'), ord('B')] = 1.25  # substituting 'H' for 'B' costs 1.25

print(lev('HANANA', 'BANANA', substitute_costs=substitute_costs))  # 1.25
print(lev('BANANA', 'HANANA', substitute_costs=substitute_costs))  # 1.0 (not symmetrical!)

transpose_costs = np.ones((128, 128), dtype=np.float64)
transpose_costs[ord('A'), ord('B')] = 0.75

# lev does not support transpositions; use osa or dam_lev
print(dam_lev('ABNANA', 'BANANA', transpose_costs=transpose_costs))  # 0.75
```

`lev`, `osa`, and `dam_lev` are aliases for `levenshtein`, `optimal_string_alignment`, and `damerau_levenshtein`, respectively.

## Limitations

- All string lookups are case sensitive.
- Only the first 128 ASCII characters are supported. Cost arrays are indexed by `ord()` value.
- Cost arrays must be `numpy.float64` arrays of shape `(128,)` for insert/delete costs and `(128, 128)` for substitute/transpose costs. Defaults to all-ones if not provided.

## Attribution

Originally created by [David Su](https://github.com/dsu1995) at InfoScout. Rewritten in Rust by [Vikram Saraph](https://github.com/vhxs).

## References

- [Levenshtein distance / Wagner-Fischer algorithm](https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm)
- [Optimal String Alignment distance](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance#Optimal_string_alignment_distance)
- [Damerau-Levenshtein distance](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance#Distance_with_adjacent_transpositions)
