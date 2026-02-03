# Draft Email to MIT PSFC

**To:** Cristina Rea (rea@psfc.mit.edu), Lucas Spangher
**Subject:** Manifold distance for disruption prediction - potentially complementary approach

---

Dr. Rea, Dr. Spangher,

I'm reaching out with some preliminary results that may be relevant to your work on disruption prediction. I want to be upfront: I'm not a plasma physicist—I'm a former attorney who transitioned to ML research a year ago, working on geometric interpretability of neural networks. But I stumbled onto something in your domain that I think warrants feedback from people who actually know the physics.

**Background:**

I've been developing tools to analyze the geometry of high-dimensional state spaces in language models—measuring things like local dimension, expansion ratio, and spectral entropy of activation trajectories. The core insight is that these systems operate on low-dimensional manifolds within high-dimensional measurement spaces, and abnormal behavior shows up as geometric divergence before it appears in individual features.

I wondered if the same approach might apply to tokamak diagnostics.

**What I did:**

Using publicly available FAIR-MAST data (44 diagnostic channels, ~100 shots), I:

1. Confirmed plasma dynamics live on a ~3.5D manifold within the 44D measurement space
2. Built an unsupervised anomaly detector using geometric features
3. Found that 5 of 7 top geometric anomalies were confirmed disruptions (validated via plasma current termination)
4. Compared detection lead times:
   - Raw diagnostic spikes (3σ): ~20 ms
   - **Manifold distance from stable region: ~1000 ms**

The manifold distance approach—measuring how far the current state is from a learned representation of stable plasma operation—detected precursors 400-750 ms earlier than raw geometric features on individual shots.

**The conceptual frame:**

The Navier-Stokes/MHD equations are reactive—they describe local state evolution but don't see global constraints until they're violated. Manifold geometry is predictive—it measures position in state space and can detect drift toward the boundary before local dynamics spike.

This suggests disruption prediction might benefit from being reframed as topology recognition rather than trajectory forecasting.

**Why I'm writing:**

I'm aware of your DisruptionBench work and the recent Nature paper on RL-based tearing instability avoidance. The RL approach learns control policies end-to-end (black box). What I'm proposing is complementary: an explicit, interpretable manifold that could provide a navigation signal—not just "you're about to disrupt" but "you're drifting in this direction, here's the gradient back to stability."

I have no idea if this is novel, obvious, or already tried and rejected. I'd genuinely appreciate any feedback:

- Is this approach already being pursued?
- Are there obvious flaws a plasma physicist would immediately see?
- If there's any merit here, what would be the right next step?

I'm happy to share the code and full results. Everything was done on public data (FAIR-MAST S3) and is reproducible.

Thank you for your time. I know unsolicited emails from outsiders are often noise, but the potential application to ITER-scale disruption avoidance seemed worth the awkwardness of reaching out.

Best regards,

Jason Kempf
[Your contact info]

GitHub: [link to plasma/ directory if you make it public]

---

**Attachments to include:**
- REACTIVE-VS-PREDICTIVE.md (the theoretical framing)
- GEOMETRY_FINDINGS.md (the empirical results)
- Or just link to a public repo
