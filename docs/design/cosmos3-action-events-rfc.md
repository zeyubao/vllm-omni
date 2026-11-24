# Draft: Cosmos3 action-ready output

Status: local implementation proposal; not an accepted public API or maintainer decision.

## Problem

An action policy can finish early positions before the remaining joint denoising steps.
Synchronous HTTP from the model couples execution to a serving endpoint and makes output
transport depend on benchmark code.

## Proposed ownership

The model creates detached action-ready events after the Euler update and condition-mask
restoration. Only the output rank publishes. Worker request streaming owns transport,
ordering and cancellation. Serving owns optional HTTP callback delivery. LIBERO input,
normalization and benchmark presets live outside the core execution package.

An event identifies its action indices, completed step, total sampling steps and protocol
version. Request identity is bound by worker transport. Each position is delivered once.
Action events opt into retaining the final payload; existing media streams keep their
final-media suppression behavior. Final output indices follow partial event indices.

## Compatibility

The existing benchmark `/predict*` protocol is retained by a FastAPI frontend. Callback
URLs are validated and removed before model execution. Existing OpenPI robot serving is
not modified: its one-request/one-action-reply behavior does not automatically implement
incremental actions. A future OpenPI integration should define the incremental wire contract
before changing that endpoint.

## Questions for upstream review

- Should incremental actions use the existing OpenPI connection, a generic event stream,
  or an explicitly documented callback adapter?
- Is retaining the final payload a general streaming-output capability or action-specific metadata?
- Which compatibility frontend and benchmark components belong in the main repository?

## Required evidence

Numerical action-prefix parity, first-event ordering, final payload preservation, no duplicate
positions, cancellation and disconnect behavior, and a real 910B smoke with matching dependencies.
Review must distinguish host sampler timings from synchronized device timings and policy wall time.
