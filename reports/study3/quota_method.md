# Coverage-quota-preserving reranking

This extension tests a specific explanation suggested by Study 2: a rule model
may improve ranking among rule-covered targets but lose recall when it gives
more of the fixed top-K slots to rule-covered candidates. The observed loss
does not establish causation; the present constraint makes that trade-off
testable on new, previously unobserved confirmation users.

For user u, let C(u) be the historical candidate pool and c(u,i) the binary
rule-coverage flag derived without the next-basket target. Let q(u) be the
number of covered items in the base model's top-K. Select q(u) covered items
and min(K, |C(u)|)-q(u) uncovered items, using the mixed score
(1-w) * base_score + w * rule_model_score within each group. Finally order
selected items by that mixed score. Product ID resolves all score ties.

The function reads no labels and preserves every candidate row. Its returned
values encode a complete ranking, not calibrated probabilities. The function
must be called again for each K. The base and rule models use the same training
protocol and candidate universe; w is selected on validation only and fixed
before confirmation. Include w=0 as a no-change control. If validation chooses
w=0, the hypothesis has not earned a model change; report that result honestly.

This is a simple task-specific constrained reranking experiment. Neither the
quota idea nor reranking should be claimed as globally novel. Its course-level
contribution is linking a measured failure pattern to a transparent intervention
and evaluating that intervention prospectively, including negative results.

The constraint guarantees only the *number* of covered recommendations. It
does not guarantee recall, NDCG, fairness, or non-inferiority. It can still replace
a correct uncovered item with an incorrect uncovered item. It can also block a
useful cross-group change when the base quota is wrong. If the rule model never
changes within-group ranking, the selected set is unchanged and Recall cannot
improve. Users with no covered candidates or all covered candidates receive an
unconstrained mixed ranking. When fewer than K candidates exist, all candidates
are returned, so only rank-sensitive metrics can change. The quotas are model
derived, not ground-truth coverage allocations, and have no causal meaning.

Recommended comparisons are: base, unconstrained rule model, and constrained
reranking (including a validation-selected mixing weight). Evaluate identical
users/candidates/K, total recall, covered/uncovered hit contributions with the
same total-target denominator, exposure counts, and paired uncertainty. A null
overall result should narrow the proposed explanation rather than trigger
additional tuning on confirmation users.
