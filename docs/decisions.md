# Implementation decisions

The original files remain in the sibling assignment files folder and are not published.
The user approved recalculating metrics while retaining original fixtures and documenting differences.

## Source discrepancies

The supplied DTI labels are 30.4%, 42%, 50%. Itemized debts plus proposed housing yield 56%, 71.4737%, 83.3333%.
The starter sums the aggregate debt total as another obligation; this implementation excludes that aggregate and validates it against the itemized sum.
The expected approval and conditional outcomes are therefore inconsistent with the supplied financial inputs. We do not alter facts to force those labels.
Reserves use liquid funds remaining after down payment and closing costs, rather than gross liquid assets.
The large-deposit threshold follows policy 3.3: min($1000,25% monthly income), plus aggregate review at $1000. The policy has inconsistent strict/inclusive wording; exactly-threshold single deposits are not flagged unless aggregate review applies.

## Human oversight

Use main brief: risk >30 OR DTI >43% OR appraisal LTV >90%. Also review bias flags and denials.
A real LangGraph interrupt preserves the checkpoint until a supplied human-review record resumes it.
Notebook review records are explicitly simulated demonstrations, not actual senior-underwriter approvals.

## Model and orchestration

Great Learning proxy, gpt-4o-mini, temperature zero. Chroma with live embeddings from text-embedding-3-small.
Specialists run sequentially under supervisor routing, consistent with the learner graph and SVG routing.
Tools run deterministically before model calls; the model interprets the resulting metrics and retrieved evidence.
A transparent educational risk score allocates 0-25 per domain; hard policy failures override score bands.
No claim is made that these scores are calibrated default probabilities.

## Limits

This is a course prototype, not a production lending or compliance system. Policy content is assignment material, not independently validated current law.
Keyword bias screening can produce false positives and miss discrimination. Zero flags do not prove fairness.
Provided descriptions of documents are not verification of actual paystubs, tax returns or appraisals.
All supplied applications lack an explicit loan program; conventional rules are an explicit educational assumption.
The graph uses process-local in-memory checkpoints; production needs secure persistence, identity verification and authorization.
No live credit bureau, LOS sync, notification delivery, legal certification or deployment is implemented.
