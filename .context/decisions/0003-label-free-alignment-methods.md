# 0003 — Stronger label-free alignment methods

## Status

Accepted as a bounded diagnostic comparison; not accepted as portability evidence.

## Decision

Compare the existing cosine-plus-MSE target-adapter training against three label-free alternatives:

- source-covariance-whitened cosine plus MSE;
- ridge affine regression from initial SmolLM shared latents to trained Qwen shared latents;
- centered orthogonal Procrustes alignment.

All methods use the same unlabeled candidate inputs and matched initial target adapter. Whitening
statistics and closed-form maps are estimated without answers. Ridge and Procrustes maps are folded
into the existing adapter's final linear layer, preserving the standard inference path and artifact
format. The Qwen head is not part of any alignment objective.

## Rationale

Whitening tests whether anisotropy obscures alignment, ridge tests an unconstrained regularized
linear correspondence, and Procrustes tests whether a rotation/reflection plus translation is
sufficient in the equal-width shared latent space. These are the smallest stronger baselines that
do not change the decision architecture or introduce supervised target optimization.

## Consequences

- Procrustes is compatible because both adapters emit 256-dimensional shared latents.
- Ridge and Procrustes alter only the target adapter's final affine parameters.
- In the seed-0 bounded diagnostic, none exceeded the existing 0.4219 ID cosine-plus-MSE result;
  ridge and Procrustes reached 0.4062 and whitening reached 0.2969.
- Lower global latent reconstruction loss did not imply better frozen-head decisions: ridge had the
  lowest raw alignment loss but lower ID accuracy than cosine plus MSE.
