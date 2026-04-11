"""Knowledge localization and reconstruction analysis.

This sub-package contains the fact-level experiments that quantify
selective unlearning at the trigram level:

- ``trigrams``       Extract corpus-specific token trigrams.
- ``localization``   Compute per-neuron gradient attributions.
- ``suppression``    Zero out top-k neurons and measure selectivity.
- ``fact_level_eu``  Apply EU at fact level for granularity disambiguation.
- ``reconstruction`` Masked-token / next-token reconstruction analysis.
"""

__all__: list[str] = []
