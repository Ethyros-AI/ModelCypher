# Owner-Local Research Artifacts

**Status:** repository policy for clean checkouts and CI

ModelCypher does not commit the bulk `results/` tree or owner training datasets.
Those paths can contain model-derived traces, per-seed benchmark outputs, and
large research inventories. They remain on the owner's research volume and are
not evidence that a clean public checkout can reproduce by itself.

The public repository tracks the corresponding reports, protocols, manifests,
code, and synthetic tests. Tests that require a retained owner artifact must:

1. run when that artifact is present,
2. skip with this policy named when it is absent,
3. never synthesize a replacement result,
4. never convert absence into a passing validation claim.

Maintained documentation may reference an owner-local result family only when
the family is explicitly grandfathered in the doctrine audit or linked to a
tracked report. Source-code, script, paper, and ordinary data references must
still resolve in a clean checkout.

CI reports these unavailable artifact checks as skips. Code, CLI contracts,
synthetic geometry, packaging, architecture boundaries, and tracked evidence
policy continue to run normally.
