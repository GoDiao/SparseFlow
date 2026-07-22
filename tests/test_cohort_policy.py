import unittest

from sparseflow.cohort_policy import SessionWorkingSet, partition_working_sets


class CohortPolicyTest(unittest.TestCase):
    def test_prefers_the_cohort_with_the_smallest_union_growth(self):
        sessions = (
            SessionWorkingSet("a", frozenset({1, 2})),
            SessionWorkingSet("b", frozenset({1, 3})),
            SessionWorkingSet("c", frozenset({8, 9})),
        )
        cohorts = partition_working_sets(sessions, max_items=3)
        self.assertEqual(cohorts[0].session_ids, ("a", "b"))
        self.assertEqual(cohorts[0].working_set, frozenset({1, 2, 3}))
        self.assertEqual(cohorts[1].session_ids, ("c",))
        self.assertFalse(cohorts[0].overflow)

    def test_keeps_a_single_oversized_session_and_marks_overflow(self):
        cohorts = partition_working_sets(
            (SessionWorkingSet("large", frozenset(range(5))),), max_items=2
        )
        self.assertEqual(cohorts[0].session_ids, ("large",))
        self.assertTrue(cohorts[0].overflow)

    def test_rejects_invalid_budget(self):
        with self.assertRaises(ValueError):
            partition_working_sets((), 0)


if __name__ == "__main__":
    unittest.main()


# [Main Dev]
