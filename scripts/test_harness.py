#!/usr/bin/env python3
"""Fail-closed reporting contracts, independent of Rust or archived experiments."""
from contextlib import redirect_stdout
import io
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from audit_checks import AuditFailure, equal
from harness.cargo import InvalidOutput, parse
from harness.provenance import sha
from harness.reporting import render
from harness.runner import execute, failed, run
from harness.study import PROTOCOL, scientific, validate


def transcript(name='sample', passed=1, failed_count=0, ignored=0):
    lines = [f'     Running unittests src/lib.rs (target/release/deps/{name}-a12b)',
             f'running {passed+failed_count+ignored} tests']
    for status, count in [('ok', passed), ('FAILED', failed_count), ('ignored', ignored)]:
        lines += [f'test {status}_{i} ... {status}' for i in range(count)]
    if failed_count:
        lines += ['failures:', '---- FAILED_0 stdout ----', "assertion failed: expected 64, actual 63", 'failures:', '    FAILED_0']
    status = 'FAILED' if failed_count else 'ok'
    lines.append(f'test result: {status}. {passed} passed; {failed_count} failed; {ignored} ignored; 0 measured; 0 filtered out; finished in 0.01s')
    return '\n'.join(lines)+'\n'


class CargoTests(unittest.TestCase):
    def test_accounts_for_each_suite_and_ignored_case(self):
        result = parse(transcript('one', 3, 0, 2)+transcript('two', 4), ['one', 'two'])
        self.assertEqual(result['counts'], dict(passed=7, failed=0, ignored=2, selected=9))
        self.assertEqual(len(result['suites'][0]['tests']), 5)

    def test_integration_header(self):
        self.assertEqual(parse(transcript().replace('unittests src/lib.rs', 'tests/sample.rs'), ['sample'])['counts']['passed'], 1)

    def test_failure_preserves_witness(self):
        result = parse(transcript(failed_count=1), ['sample'])
        self.assertIn('expected 64, actual 63', result['suites'][0]['tests'][1]['detail'])

    def test_rejects_missing_suite_even_with_a_passing_tail(self):
        with self.assertRaises(InvalidOutput):
            parse(transcript('one'), ['one', 'two'])

    def test_rejects_zero_and_ignored_only(self):
        for ignored in [0, 3]:
            with self.subTest(ignored=ignored), self.assertRaises(InvalidOutput):
                parse(transcript(passed=0, ignored=ignored), ['sample'])

    def test_rejects_truncated_duplicate_foreign_and_unknown(self):
        good = transcript()
        mutations = [good[:good.index('test result:')], good+good,
                     transcript('foreign'), good.replace('test result: ok.', 'test result: SUCCESS.'),
                     good.replace(' ... ok', ' ... WHAT'), 'unrecognized output\n',
                     good.replace('Running unittests', 'Running unknown-target'),
                     good.replace('1 passed;', '2 passed;'), good.replace('running 1 tests', 'running 2 tests'),
                     good.replace('0 filtered out', '1 filtered out'),
                     good.replace('test result: ok.', 'test result: FAILED.'),
                     good.replace('test ok_0 ... ok', 'test ok_0 ... ok\ntest ok_0 ... ok')]
        for text in mutations:
            with self.subTest(text=text), self.assertRaises(InvalidOutput):
                parse(text, ['sample'])


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def invoke(self, code, timeout=3, **extra):
        step = {'id': 'sample', 'adapter': 'cargo', 'expected_suites': ['sample'],
                'argv': [sys.executable, '-c', code], **extra}
        return execute(step, self.root, self.root, dict(os.environ), timeout, lambda *_: None)

    def test_passing_process_preserves_all_log_bytes(self):
        text = transcript()
        result = self.invoke('print('+repr(text)+', end="")')
        self.assertFalse(failed(result, False))
        self.assertEqual((self.root/'sample.log').read_text(), text)
        self.assertEqual(result['log_sha256'], sha(self.root/'sample.log'))

    def test_passing_output_cannot_hide_nonzero_exit(self):
        result = self.invoke('import sys;print('+repr(transcript())+');sys.exit(7)')
        self.assertTrue(failed(result, False))
        self.assertEqual(result['exit'], 7)

    def test_failing_test_cannot_hide_zero_exit(self):
        result = self.invoke('print('+repr(transcript(failed_count=1))+')')
        self.assertTrue(failed(result, False))
        self.assertEqual(result['verification'], 'passed')
        self.assertEqual(result['failures'][0]['invariant'], 'test.failed')

    def test_unknown_output_cannot_pass(self):
        result = self.invoke('print("all good")')
        self.assertTrue(failed(result, False))
        self.assertEqual(result['verification'], 'failed')

    def test_compile_failure_has_bounded_diagnostic_context(self):
        result = self.invoke('import sys;print("error[E0308]: mismatched types\\n --> source.rs:2:3");sys.exit(1)')
        self.assertTrue(failed(result, False))
        self.assertIn('error[E0308]', result['failures'][0]['excerpt'])

    def test_signal_cannot_be_reported_as_pass(self):
        result = self.invoke('import os,signal;print('+repr(transcript())+',flush=True);os.kill(os.getpid(),signal.SIGTERM)')
        self.assertTrue(failed(result, False))
        self.assertEqual(result['signal'], 15)

    def test_timeout_kills_descendants_and_retains_output(self):
        marker = self.root/'late'
        child = ('import time;from pathlib import Path;p=Path('+repr(str(marker))+')\n'
                 'for i in range(200):\n p.write_text(str(i));time.sleep(0.05)')
        code = 'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",'+repr(child)+']);print("started",flush=True);time.sleep(10)'
        result = self.invoke(code, timeout=2)
        self.assertEqual(result['execution'], 'timeout')
        self.assertIn('started', (self.root/'sample.log').read_text())
        self.assertTrue(marker.exists(), 'the descendant must run before cleanup is exercised')
        stopped_value = marker.read_text()
        time.sleep(0.2)
        self.assertEqual(marker.read_text(), stopped_value)

    def test_missing_study_artifact(self):
        result = self.invoke('pass', adapter='study', kind='hard', plan_sha256='a'*64,
                             artifact=str(self.root/'missing.json'), study_root=str(self.root))
        self.assertEqual(result['verification'], 'failed')
        self.assertTrue(failed(result, False))

    def test_stale_study_identity(self):
        artifact = self.root/'study.json'
        artifact.write_text(json.dumps({'protocol': PROTOCOL, 'kind': 'hard', 'plan_sha256': 'b'*64}))
        result = self.invoke('pass', adapter='study', kind='hard', plan_sha256='a'*64, artifact=str(artifact), study_root=str(self.root))
        self.assertEqual(result['verification'], 'failed')
        self.assertEqual(result['failures'][0]['invariant'], 'study.plan_identity')

    def test_unknown_adapter(self):
        result = self.invoke('pass', adapter='unsupported')
        self.assertEqual(result['verification'], 'failed')


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.artifact = Path(self.temp.name).resolve()/'plan.json'
        self.artifact.write_text(json.dumps({'sampling_seeds': [301], 'arms': ['observation', 'memory'], 'checkpoints': [4096]}))
        self.digest = sha(self.artifact)
        self.result = {'complete': True, 'plan_sha256': self.digest, 'checkpoint_predictions': 256,
                       'results': [{'seed': 301, 'arm': arm, 'step': 4096, 'combat': 60,
                                    'feeding': 64, 'exact_fit': False} for arm in ['observation', 'memory']]}
        self.envelope = {'protocol': PROTOCOL, 'kind': 'hard', 'plan_sha256': self.digest,
                         'verification': 'passed', 'result': self.result, 'study_root': str(self.artifact.parent),
                         'scientific': scientific(self.result, 'hard'),
                         'inputs': {str(self.artifact): self.digest}}

    def test_valid_negative_is_not_verification_failure(self):
        value = validate(self.envelope, 'hard', self.digest, self.artifact.parent)
        self.assertEqual(value['outcome'], 'rejected')
        step = {'execution': 'passed', 'verification': 'passed', 'scientific': value}
        self.assertFalse(failed(step, False))
        self.assertTrue(failed(step, True))
        self.assertEqual(value['arms'][0]['worst']['margin'], -4)

    def test_forged_summary_is_rejected(self):
        self.envelope['scientific']['outcome'] = 'passed'
        with self.assertRaises(AuditFailure) as caught:
            validate(self.envelope, 'hard', self.digest, self.artifact.parent)
        self.assertEqual(caught.exception.invariant, 'study.summary')

    def test_changed_input_is_rejected(self):
        self.artifact.write_text('{"changed":true}')
        with self.assertRaises(AuditFailure) as caught:
            validate(self.envelope, 'hard', self.digest, self.artifact.parent)
        self.assertEqual(caught.exception.invariant, 'study.plan_identity')

    def test_missing_checkpoint_is_rejected_even_if_summary_is_recomputed(self):
        self.result['results'].pop()
        with self.assertRaises(AuditFailure) as caught:
            validate(self.envelope, 'hard', self.digest, self.artifact.parent)
        self.assertEqual(caught.exception.invariant, 'study.checkpoint_coverage')

    def test_missing_bound_input_is_rejected(self):
        extra = self.artifact.parent / 'weights.f32'
        extra.write_bytes(b'original')
        self.envelope['inputs'][str(extra)] = sha(extra)
        extra.write_bytes(b'changed')
        with self.assertRaises(AuditFailure) as caught:
            validate(self.envelope, 'hard', self.digest, self.artifact.parent)
        self.assertEqual(caught.exception.invariant, 'study.stale_input')

    def test_missing_completion_is_rejected(self):
        self.result['complete'] = False
        with self.assertRaises(AuditFailure):
            validate(self.envelope, 'hard', self.digest, self.artifact.parent)

    def test_structured_check_is_bounded_and_locates_first_difference(self):
        with self.assertRaises(AuditFailure) as caught:
            equal({'predictions': [1, 2, 3]}, {'predictions': [1, 4, 3]}, 'predictions', seed=301)
        self.assertEqual(caught.exception.details['field'], 'predictions/1')
        self.assertEqual(caught.exception.details['expected'], '4')

    def test_optimized_python_is_rejected(self):
        for script in ['verify_interaction_hard.py', 'verify_interaction_standardized.py', 'verify_interaction_zero_state.py', 'run_harness.py']:
            result = subprocess.run([sys.executable, '-O', str(Path(__file__).parent/script), '--help'], capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn(b'audit.optimized_python_unsupported', result.stderr)


class FullCorpusTests(unittest.TestCase):
    def result(self):
        rows = []
        for seed in (1, 2, 3):
            for arm in ('memory', 'zero-state'):
                for step in (512, 2048):
                    good = arm == 'zero-state'
                    rows.append({'seed': seed, 'arm': arm, 'step': step, 'combat': .96, 'attack': .9,
                                 'feeding': 1. if good else .98, 'training_fit_thresholds_met': good,
                                 'counts': {'combat': dict(rows=100, correct=96, attack_labels=50, attack_correct=45),
                                            'feeding': dict(rows=100, correct=100 if good else 98, attack_labels=0, attack_correct=0)}})
        return dict(results=rows, thresholds=dict(combat=.95, attack=.9, feeding=.99), checkpoint_predictions=2400,
                    raw_nine_checkpoints_and_weights_identical=True)

    def test_joint_gate_keeps_all_initializations_and_denominators(self):
        r = self.result()
        s = scientific(r, 'zero-state')
        self.assertEqual(s['outcome'], 'passed')
        self.assertEqual(s['arms'][0]['passed'], 0)
        self.assertEqual(s['arms'][1]['passed'], 3)
        r['results'][-1]['feeding'] = .98
        r['results'][-1]['counts']['feeding']['correct'] = 98
        r['results'][-1]['training_fit_thresholds_met'] = False
        s = scientific(r, 'zero-state')
        self.assertEqual(s['outcome'], 'rejected')
        self.assertEqual(s['arms'][1]['passed'], 2)
        self.assertEqual(s['arms'][1]['worst']['margin'], -1)

    def test_forged_rates_gates_counts_and_duplicate_records_rejected(self):
        base = self.result()
        for mutation in ('rate', 'gate', 'counts', 'duplicate', 'threshold'):
            r = deepcopy(base)
            if mutation == 'rate': r['results'][0]['combat'] = 1.
            elif mutation == 'gate': r['results'][0]['training_fit_thresholds_met'] = True
            elif mutation == 'counts': r['results'][0]['counts']['combat']['correct'] = True
            elif mutation == 'duplicate': r['results'].append(r['results'][0])
            else: r['thresholds']['feeding'] = .98
            with self.subTest(mutation=mutation), self.assertRaises(AuditFailure):
                scientific(r, 'zero-state')

    def test_plan_binds_full_corpus_prediction_and_domain_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan = root / 'plan.json'
            plan.write_text(json.dumps(dict(sampling_seeds=[1, 2, 3], arms=['memory', 'zero-state'],
                                            checkpoints=[512, 2048], training_counts=[dict(combat=100, feeding=100)],
                                            thresholds=dict(combat=.95, attack=.9, feeding=.99))))
            digest = sha(plan)
            result = self.result()
            result.update(complete=True, plan_sha256=digest, training_rows=200)
            envelope = dict(protocol=PROTOCOL, kind='zero-state', plan_sha256=digest, study_root=str(root),
                            verification='passed', result=result, scientific=scientific(result, 'zero-state'),
                            inputs={str(plan): digest})
            self.assertEqual(validate(envelope, 'zero-state', digest, root)['outcome'], 'passed')
            result['checkpoint_predictions'] = 128 * 12
            with self.assertRaises(AuditFailure) as caught:
                validate(envelope, 'zero-state', digest, root)
            self.assertEqual(caught.exception.invariant, 'study.prediction_coverage')
            result['checkpoint_predictions'] = 2400
            result['results'][0]['counts']['feeding']['rows'] = 99
            with self.assertRaises(AuditFailure) as caught:
                validate(envelope, 'zero-state', digest, root)
            self.assertEqual(caught.exception.invariant, 'study.domain_coverage')


class RunTests(unittest.TestCase):
    def run_fake(self, outputs, source_changed=False, require_scientific=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        directory = root/'report'
        before, after = {'sha256': 'a'}, {'sha256': 'b' if source_changed else 'a'}
        steps = [{'id': str(i), 'adapter': 'cargo', 'argv': ['fake']} for i in range(len(outputs))]
        outputs = [{**steps[i], 'scientific': {'outcome': 'not_applicable'}, 'failures': [], 'exit': 0, **out}
                   for i, out in enumerate(outputs)]
        with patch('harness.runner.snapshot', side_effect=[before, after]), patch('harness.runner.environment', return_value={}), patch('harness.runner.execute', side_effect=outputs), redirect_stdout(io.StringIO()) as stdout:
            code = run(root, 'test', directory, steps, 'test-only', 10, require_scientific)
        return code, json.loads((directory/'report.json').read_text()), stdout.getvalue()

    def test_source_changes_cannot_pass(self):
        code, report, _ = self.run_fake([{'execution': 'passed', 'verification': 'passed'}], True)
        self.assertEqual(code, 1)
        self.assertEqual(report['failures'][0]['invariant'], 'runner.source_changed')

    def test_interrupt_accounts_for_unexecuted_steps(self):
        code, report, _ = self.run_fake([{'execution': 'interrupted', 'verification': 'failed'}, {}])
        self.assertEqual(code, 130)
        self.assertEqual(report['coverage'], dict(requested=2, completed=0, not_run=1, incomplete=1))

    def test_independent_steps_continue_after_failure(self):
        code, report, _ = self.run_fake([{'execution': 'failed', 'verification': 'failed'}, {'execution': 'passed', 'verification': 'passed'}])
        self.assertEqual(code, 1)
        self.assertEqual(report['coverage']['completed'], 2)

    def test_required_scientific_gate_controls_final_exit(self):
        result = {'execution': 'passed', 'verification': 'passed', 'scientific': {'outcome': 'rejected'}}
        for required in [False, True]:
            with self.subTest(required=required):
                code, report, output = self.run_fake([result], require_scientific=required)
                self.assertEqual(code, int(required))
                self.assertIn('science=rejected', output)
                self.assertEqual(report['status'], 'failed' if required else 'passed')

    def test_passing_display_bound(self):
        _, report, output = self.run_fake([{'execution': 'passed', 'verification': 'passed'}]*30)
        self.assertLessEqual(len(output.encode()), 2048)
        self.assertIn('report.json', output)
        self.assertEqual(report['display_bytes'], len(output.encode()))

    def test_failure_display_bound_and_omissions(self):
        failures = [{'invariant': 'example', 'message': 'é'*10000} for _ in range(20)]
        _, report, output = self.run_fake([{'execution': 'failed', 'verification': 'failed', 'failures': failures}])
        self.assertLessEqual(len(output.encode()), 6144)
        self.assertIn('omitted=15', output)
        self.assertEqual(len(report['failures']), 20)

    def test_long_path_and_scope_cannot_break_display_budget(self):
        _, report, _ = self.run_fake([{'execution': 'passed', 'verification': 'passed'}])
        report['directory'] = '/' + 'long/'*500
        report['scope'] = 'é'*10000
        self.assertLessEqual(len(render(report).encode()), 2048)


if __name__ == '__main__':
    unittest.main()
