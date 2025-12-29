# Publication Guide: GitHub Releases + Zenodo DOI

**Created**: 2025-12-29
**Status**: Ready for execution

---

## Why GitHub + Zenodo?

| Platform | Benefit |
|----------|---------|
| **GitHub Releases** | Version control, direct links, community engagement |
| **Zenodo** | Automatic DOI, academic citations, EU Open Science infrastructure |

No endorsement required. No peer review gatekeeping. Publish your research directly.

---

## Step 1: Create GitHub Release

### Prerequisites
- [ ] Ensure all papers are finalized
- [ ] Update version in pyproject.toml if needed
- [ ] Commit all changes

### Release Structure

Create release `v0.1.0-papers` with:

```
ModelCypher Paper Series v0.1.0

This release includes the ModelCypher research paper series:

## Papers

| Paper | Title | Status |
|-------|-------|--------|
| [Paper 0](papers/paper-0-the-shape-of-knowledge.md) | The Shape of Knowledge | Framework/Theory |
| [Paper 1](papers/paper-1-invariant-semantic-structure.md) | Invariant Semantic Structure | Validated |
| [Paper 5](papers/paper-5-semantic-highway.md) | The Semantic Highway | Validated |
| [Paper 4](papers/paper-4-modelcypher-toolkit.md) | ModelCypher Toolkit | Systems Paper |
| [Paper 2](papers/paper-2-entropy-safety-signal.md) | Entropy Safety Signal | Methodology (Preliminary) |
| [Paper 3](papers/paper-3-cross-architecture-transfer.md) | Cross-Architecture Transfer | Methodology (Preliminary) |

## Key Finding

Semantic primes are NOT geometrically special compared to random words. Both achieve CKA > 0.9 across model families. This is a *stronger* result than originally hypothesized: invariance is universal, not limited to theoretically-motivated concepts.

## Installation

\`\`\`bash
pip install modelcypher
# or
poetry install
\`\`\`

## Citation

If you use ModelCypher in your research, please cite:

\`\`\`bibtex
@software{kempf2025modelcypher,
  author = {Kempf, Jason},
  title = {ModelCypher: A Geometric Toolkit for LLM Analysis},
  year = {2025},
  url = {https://github.com/Ethyros-AI/ModelCypher},
  version = {0.1.0}
}
\`\`\`
```

### CLI Commands

```bash
# Tag the release
git tag -a v0.1.0-papers -m "ModelCypher Paper Series v0.1.0"
git push origin v0.1.0-papers

# Or use GitHub CLI
gh release create v0.1.0-papers \
  --title "ModelCypher Paper Series v0.1.0" \
  --notes-file papers/RELEASE_NOTES.md \
  --prerelease
```

---

## Step 2: Connect Zenodo

### One-Time Setup

1. **Go to**: https://zenodo.org/account/settings/github/
2. **Sign in** with GitHub
3. **Flip the switch** for `Ethyros-AI/ModelCypher` (or your fork)
4. **Done** - Zenodo will now automatically create DOIs for releases

### After Creating GitHub Release

1. Zenodo automatically detects the release (within minutes)
2. Go to https://zenodo.org/account/settings/github/
3. Click "Get DOI" next to your repository
4. **Copy the DOI badge** for your README

### DOI Badge Format

Add to README after release:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

---

## Step 3: Academic Visibility

### Add to ORCID (Optional)

If you have an ORCID:
1. Go to https://orcid.org
2. Add Work → Search & Link → Zenodo
3. Link the DOI to your profile

### Google Scholar

Google Scholar will automatically index GitHub releases with DOIs after ~2-4 weeks.

### Semantic Scholar

Submit via: https://www.semanticscholar.org/product/api/tutorial/

---

## Paper Status Summary

| Paper | Ready | DOI Will Cover |
|-------|-------|----------------|
| Paper 0 (Framework) | ✅ | Theory & formalism |
| Paper 1 (Invariant Structure) | ✅ | CKA cross-family results |
| Paper 2 (Entropy Safety) | ⚠️ Preliminary | Methodology only |
| Paper 3 (Cross-Architecture) | ⚠️ Preliminary | Methodology only |
| Paper 4 (Toolkit) | ✅ | Systems & implementation |
| Paper 5 (Semantic Highway) | ✅ | ID cliff empirical results |

---

## Alternative: Zenodo Direct Upload

If you want separate DOIs per paper (not recommended for first release):

1. Go to https://zenodo.org/deposit/new
2. Upload PDF or markdown
3. Fill metadata
4. Publish

**Recommendation**: Single release first, split papers later if needed.

---

## Timeline

| Action | Time Required |
|--------|---------------|
| Finalize papers | Done |
| Create GitHub release | 5 minutes |
| Connect Zenodo | 5 minutes |
| DOI appears | ~5-15 minutes |
| Update README with DOI | 2 minutes |

**Total**: ~15-30 minutes to have citable research with a DOI.
