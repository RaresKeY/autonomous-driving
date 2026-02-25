from __future__ import annotations


def test_role_test_files_are_separate_and_present(role_test_files):
    expected_names = {
        "inference_tests.py",
        "training_tests.py",
        "download_tests.py",
        "parse_tests.py",
    }
    actual_names = {path.name for path in role_test_files.values()}

    assert actual_names == expected_names
    for path in role_test_files.values():
        assert path.exists(), f"Missing role test file: {path}"


def test_pytest_config_supports_star_tests_pattern(project_root):
    pytest_ini = project_root / "pytest.ini"
    assert pytest_ini.exists(), "Add pytest.ini so *_tests.py files are discovered by default"

    content = pytest_ini.read_text(encoding="utf-8")
    assert "*_tests.py" in content, "pytest.ini must include *_tests.py discovery pattern"


def test_role_target_mapping_matches_current_specs(role_targets):
    # Meta-test for the agreed owner -> target-file mapping.
    assert role_targets["Paul"].as_posix().endswith("/src/inference.py")
    assert role_targets["Anca"].as_posix().endswith("/src/training.py")
    assert role_targets["Mihaela"].as_posix().endswith("/src/parse_kitti_label.py")
    assert role_targets["Claudia"].as_posix().endswith("/src/download.py")
