# TIFU-KNN implementation note

This is an independently written sparse adaptation of the released TIFU-KNN implementation, for this project's **historical-candidate, repeat-target** task. It is not a reproduction of the paper's published benchmark scores.

## Primary references

- Hu, He, Gao and Zhang (SIGIR 2020), [Modeling Personalized Item Frequency Information for Next-basket Recommendation](https://arxiv.org/abs/2006.00556), especially Section 3.2 and Table 3.
- [Author repository](https://github.com/HaojiHu/TIFUKNN) and [released implementation](https://github.com/HaojiHu/TIFUKNN/blob/master/TIFUKNN.py), inspected on 2026-11-14: `group_history_list`, `temporal_decay_sum_history`, `KNN`, and `merge_history`.

The paper combines a target user's temporal frequency vector with the unweighted mean vector of Euclidean nearest neighbors. Table 3's Instacart configuration is 3 groups, within decay 0.9, across-group decay 0.7, 900 neighbors and personal weight 0.9. That configuration is included in the bounded validation grid. A weight of 1 is a personal-only ablation.

## Paper/code discrepancy

The paper describes resetting basket decay inside each group, and a particular uneven-group allocation. The released code instead applies basket decay relative to the whole history, balances group sizes with extra baskets in recent groups, and retains the requested group count in the group exponent even for shorter histories. Our `implementation="author_code"` follows those executable semantics explicitly. `group_count` is the number of groups, despite the original variable name `group_size`.

For zero-based chronological basket index j and group g, the implemented weight is `rb^(n-1-j) * rg^(m-1-g) / (actual_group_size * actual_number_of_groups)`. Sparse sums of binary baskets give the same vectors as the executable aggregation. Tests compare against an independent dense-loop reference and hand-calculated unequal/short histories.

## Study-specific decisions

Only declared training users enter the neighbor bank; held-out users contribute only their own histories. Self neighbors are excluded. Euclidean ties use ascending user ID. All catalog dimensions are retained without popularity filtering. Candidate output order is preserved; no candidates are dropped. Temporal settings, neighbor counts and personal weights are selected using validation labels only. These differences in data, candidate scope and evaluation mean the resulting scores must not be compared directly with the paper's full-basket benchmark table.
