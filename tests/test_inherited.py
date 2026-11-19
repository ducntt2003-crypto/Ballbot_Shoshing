"""Retained, unchanged historical tests; select self-contained native checks."""
from sloshbot.bootstrap import setup,ROOT
setup()
import importlib.util,unittest

def load_tests(loader,tests,pattern):
    suite=unittest.TestSuite()
    for file,cls,names in [('test_research.py','ResearchTests',None),('test_contact.py','ContactTests',['test_capsule_sphere_distance_matches_rotated_native_contacts','test_force_frames_virtual_work_and_anisotropic_axes','test_diagnostic_calls_do_not_mutate_state_or_warmstart','test_positive_negative_motor_generalized_torque','test_reset_reproducible_and_default_contact_parameters'])]:
        spec=importlib.util.spec_from_file_location('inherited_'+cls,ROOT/'tests/historical'/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);c=getattr(m,cls)
        suite.addTests(loader.loadTestsFromTestCase(c) if names is None else [c(n) for n in names])
    return suite
