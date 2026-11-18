import unittest
from audit_step5_cosine import summarize


def example():
    return {
        "scores": [
            {"child": "urn:rail", "parent": "urn:role", "scope": "RAIL_NEW_DOMAIN", "cosine": .215386, "status": "HELD_OUT_COSINE"},
            {"child": "urn:a", "parent": "urn:b", "scope": "SOURCE_IMPORT_REFERENCE", "cosine": -.009896, "status": "HELD_OUT_COSINE"}
        ],
        "heldout_source_import_pairs": 1,
        "mean_heldout_source_import_cosine": -.009896,
        "heldout_rail_cosine": .215386
    }


class AuditTests(unittest.TestCase):
    def test_rail_below_target_fails(self):
        report = summarize(example())
        self.assertEqual(report["rail_heldout_mean_cosine"], .215386)
        self.assertEqual(report["status"], "FAIL_COSINE_REQUIREMENT")
        self.assertFalse(report["threshold_met"])

    def test_reporting_does_not_swap_reference_mean(self):
        report = summarize(example())
        self.assertEqual(report["reference_heldout_mean_cosine"], -.009896)
        self.assertNotEqual(report["rail_heldout_mean_cosine"], report["reference_heldout_mean_cosine"])

    def test_valid_above_target(self):
        e = example()
        e["scores"][0]["cosine"] = .81
        e["heldout_rail_cosine"] = .81
        self.assertTrue(summarize(e)["threshold_met"])

    def test_min_pairs_cannot_be_spoofed(self):
        e = example()
        e["scores"][0]["cosine"] = .81
        e["heldout_rail_cosine"] = .81
        self.assertFalse(summarize(e, min_independent_rail_pairs=3)["threshold_met"])

    def test_duplicate_edges_rejected(self):
        e = example()
        e["scores"].append(dict(e["scores"][0]))
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            summarize(e)

    def test_nonfinite_rejected(self):
        e = example()
        e["scores"][0]["cosine"] = float("nan")
        with self.assertRaisesRegex(ValueError,"Invalid"):
            summarize(e)

    def test_reference_count_checked(self):
        e = example()
        e["heldout_source_import_pairs"] = 165
        with self.assertRaisesRegex(ValueError, "sample size"):
            summarize(e)


if __name__ == "__main__":
    unittest.main()
