# Automated test coverage

50 separately discovered automated cases passed on Python 3.12. Subtests and invalid-input variants are not counted as additional cases.

Executed command: `python tools/run_tests.py`. The standard `python -m unittest discover -s tests -v` command also executed all 50 successfully.

Run UTC: `2026-10-12T09:06:47.321499+00:00`. See [`test_results.json`](../reports/verification/test_results.json) for each recorded outcome.

Coverage: 20 data/protocol/preprocessing cases, 13 fitting/inference cases, and 17 scorer/CLI integration cases. These checks validate software behavior; they do not establish statistical generalization.

## ProtocolTests (11 cases)

Source: [`test_protocol.py`](../tests/test_protocol.py).

1. `test_loader_rejects_missing_and_extra_columns` — passed.
2. `test_features_require_schema_and_finite_numerics` — passed.
3. `test_duplicate_observations_are_distinct_from_label_conflicts` — passed.
4. `test_split_determinism_and_feature_disjointness` — passed.
5. `test_mlp_inference_is_deterministic_and_bounded` — passed.
6. `test_scorer_rejects_invalid_shapes_and_probabilities` — passed.
7. `test_official_label_normalization_and_missing_value` — passed.
8. `test_bad_label_is_rejected` — passed.
9. `test_keys_ignore_row_index_and_find_duplicate_features` — passed.
10. `test_validation_threshold_maximizes_positive_f1` — passed.
11. `test_transform_does_not_learn_test_categories_or_statistics` — passed.

## EvaluationTests (7 cases)

Source: [`test_evaluation.py`](../tests/test_evaluation.py).

12. `test_perfect_score_and_order_independent_join` — passed.
13. `test_wrong_ranking_known_average_precision` — passed.
14. `test_constant_probability_baseline` — passed.
15. `test_missing_extra_duplicate_and_blank_ids` — passed.
16. `test_nan_infinite_out_of_range_and_nonnumeric_predictions` — passed.
17. `test_schema_and_target_errors` — passed.
18. `test_invalid_threshold` — passed.

## DataBoundaryTests (9 cases)

Source: [`test_fitting_and_integration.py`](../tests/test_fitting_and_integration.py).

19. `test_loader_rejects_missing_label` — passed.
20. `test_loader_rejects_missing_numeric_feature` — passed.
21. `test_feature_column_order_is_enforced` — passed.
22. `test_empty_feature_matrix_is_rejected` — passed.
23. `test_categorical_missing_values_have_explicit_encoding` — passed.
24. `test_numeric_imputation_uses_training_median` — passed.
25. `test_stratified_split_preserves_minority_proportion` — passed.
26. `test_conflicting_duplicate_keeps_first_label` — passed.
27. `test_missing_category_feature_matches_are_detected` — passed.

## FittingTests (13 cases)

Source: [`test_fitting_and_integration.py`](../tests/test_fitting_and_integration.py).

28. `test_flat_validation_stops_after_patience` — passed.
29. `test_best_checkpoint_weights_are_restored` — passed.
30. `test_validation_improvement_resets_patience` — passed.
31. `test_tiny_validation_changes_do_not_reset_patience` — passed.
32. `test_min_delta_requires_strict_improvement` — passed.
33. `test_epoch_budget_still_restores_best_checkpoint` — passed.
34. `test_training_same_seed_reproduces_weights_and_history` — passed.
35. `test_modern_optimizer_applies_weight_decay` — passed.
36. `test_legacy_control_retains_final_epoch_without_early_stopping` — passed.
37. `test_invalid_training_configuration_fails_before_fit` — passed.
38. `test_invalid_training_arrays_fail_before_fit` — passed.
39. `test_mlp_inference_rejects_empty_vector_and_nonfinite_inputs` — passed.
40. `test_legacy_inference_returns_positive_softmax_probability` — passed.

## LocalIntegrationTests (10 cases)

Source: [`test_fitting_and_integration.py`](../tests/test_fitting_and_integration.py).

41. `test_prepare_separates_public_labels_and_never_reads_test` — passed.
42. `test_prepare_is_byte_reproducible` — passed.
43. `test_reference_reads_only_train_and_feature_files` — passed.
44. `test_reference_rejects_label_leakage_in_features` — passed.
45. `test_reference_rejects_invalid_training_labels` — passed.
46. `test_reference_rejects_id_overlap` — passed.
47. `test_reference_rejects_feature_overlap_under_different_ids` — passed.
48. `test_cli_scores_csv_and_writes_finite_json` — passed.
49. `test_cli_invalid_submission_exits_two_without_new_score` — passed.
50. `test_id_string_normalization_rejects_collisions` — passed.

## What the fitting tests establish

Controlled validation-AP sequences exercise plateau stopping, patience reset, strict minimum improvement, ties and budget exhaustion. Actual model state_dict tensors are compared with captured epoch checkpoints to verify restoration. Deterministic training and AdamW decay are checked; the legacy control is verified to retain its final epoch.

The real-data train/validation metrics and loss/AP curves are separate empirical checks, documented in [fitting_diagnostics.md](fitting_diagnostics.md). No assertion that overfitting or underfitting is completely eliminated is made.
