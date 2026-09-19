"""Seed-lineage rejection tests using small, self-contained export manifests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from verify_feeding_ecology import sha, validate_cohort


class FeedingCohortTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.references = []
        self.arms = {}
        for initialization in [1435400301, 1435400302, 1435400303]:
            for arm in ['control', 'treatment']:
                path = self.root / f'{initialization}-{arm}.json'
                path.write_text(json.dumps({
                    'initialization_seed': initialization, 'arm': arm,
                    'weights_sha256': f'{initialization}-{arm}-weights',
                    'source_seed_ledger': {'training': [101], 'validation': [202],
                                           'confirmation': [1434999901, 1434999902]}}))
                self.references.append({'path': str(path), 'sha256': sha(path)})
                if initialization == 1435400301:
                    self.arms[arm] = {'manifest_sha256': sha(path),
                                      'weights_sha256': f'{initialization}-{arm}-weights'}
        parent = self.root / 'parent.json'
        parent.write_text(json.dumps({'source_seed_ledger': None,
                                     'development_seeds': [303], 'weights_sha256': 'parent'}))
        self.references.append({'path': str(parent), 'sha256': sha(parent)})
        self.arms['parent'] = {'manifest_sha256': sha(parent), 'weights_sha256': 'parent'}
        self.audit = {'seeds': [404, 405], 'reserved_confirmation': [1434999901, 1434999902],
                      'additional_exposed_seeds': [304], 'exposed_seeds': [101, 202, 303, 304],
                      'all_reserved_seeds': [1434999901, 1434999902],
                      'manifests': self.references,
                      'prior_experiment_search': {'exit_code': 1, 'stdout': '', 'stderr': ''}}
        self.plan = {'cohort': 'fresh_development', 'seeds': [404, 405],
                     'initialization_seed': 1435400301,
                     'reserved_confirmation': [1434999901, 1434999902], 'arms': self.arms,
                     'assessment_mode': 'sustained_feeding',
                     'boundary_verification': 'recorded_boundary_only'}
        self.save_audit()

    def save_audit(self):
        path = self.root / 'audit.json'
        path.write_text(json.dumps(self.audit))
        self.plan['seed_audit'] = {'path': 'audit.json', 'sha256': sha(path)}

    def test_fresh_pair_is_accepted(self):
        self.assertTrue(validate_cohort(self.root, self.plan))

    def test_training_validation_parent_and_reserved_seeds_are_rejected(self):
        for seed in [101, 202, 303, 304, 1434999901, 1434999902]:
            with self.subTest(seed=seed):
                self.plan['seeds'] = self.audit['seeds'] = [seed, 405]
                self.save_audit()
                with self.assertRaises(AssertionError):
                    validate_cohort(self.root, self.plan)

    def test_missing_initialization_cannot_shrink_the_audit(self):
        self.audit['manifests'] = self.references[2:]
        self.save_audit()
        with self.assertRaises(AssertionError):
            validate_cohort(self.root, self.plan)

    def test_manifest_tampering_invalidates_audit(self):
        path = Path(self.references[0]['path'])
        manifest = json.loads(path.read_text())
        manifest['source_seed_ledger']['training'] = []
        path.write_text(json.dumps(manifest))
        with self.assertRaises(AssertionError):
            validate_cohort(self.root, self.plan)

    def test_stale_audit_hash_is_rejected(self):
        path = self.root / 'audit.json'
        path.write_text(path.read_text() + '\n')
        with self.assertRaises(AssertionError):
            validate_cohort(self.root, self.plan)

    def test_duplicate_or_invalid_seeds_are_rejected(self):
        for seeds in [[], [404, 404], [-1, 405], [True, 405], [2**64, 405]]:
            with self.subTest(seeds=seeds):
                self.plan['seeds'] = self.audit['seeds'] = seeds
                self.save_audit()
                with self.assertRaises(AssertionError):
                    validate_cohort(self.root, self.plan)

    def test_failed_search_cannot_certify_freshness(self):
        for result in [{'exit_code': 0, 'stdout': 'old-report.json', 'stderr': ''},
                       {'exit_code': 2, 'stdout': '', 'stderr': 'unreadable directory'}]:
            self.audit['prior_experiment_search'] = result
            self.save_audit()
            with self.assertRaises(AssertionError):
                validate_cohort(self.root, self.plan)

    def test_unqualified_arm_cannot_borrow_audit(self):
        self.plan['arms'] = copy.deepcopy(self.arms)
        self.plan['arms']['control']['manifest_sha256'] = 'unlisted-manifest'
        with self.assertRaises(AssertionError):
            validate_cohort(self.root, self.plan)

    def test_fresh_cohort_cannot_claim_old_canonical_prefix(self):
        self.plan['canonical_prefix_trials'] = {}
        with self.assertRaises(AssertionError):
            validate_cohort(self.root, self.plan)

    def test_swapped_arms_or_initialization_are_rejected(self):
        self.plan['arms'] = {'control': self.arms['treatment'],
                             'treatment': self.arms['control'], 'parent': self.arms['parent']}
        with self.assertRaises(AssertionError):
            validate_cohort(self.root, self.plan)
        self.plan['arms'] = self.arms
        self.plan['initialization_seed'] = 1435400302
        with self.assertRaises(AssertionError):
            validate_cohort(self.root, self.plan)


if __name__ == '__main__':
    unittest.main()
