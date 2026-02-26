from tests.role_contracts.training_contract_checks import *  # noqa: F401,F403

# tests/training_tests.py
# Teste automate pentru src/training.py
# Rares le va rula la integrare

import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Asigura ca Python gaseste modulul src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import build_model, train_model, parse_args, main


# ============================================================
# TEST 1: build_model returneaza un obiect valid
# ============================================================

def test_build_model_returns_model():
    model = build_model(architecture="yolov8n", pretrained=True)
    assert model is not None, "build_model() a returnat None"
    print("  [PASS] build_model() returneaza model valid")


def test_build_model_invalid_architecture():
    """Trebuie sa arunce exceptie la arhitectura invalida."""
    try:
        build_model(architecture="arhitectura_inexistenta_xyz")
        print("  [FAIL] build_model() nu a aruncat exceptie pentru arhitectura invalida")
        assert False
    except Exception:
        print("  [PASS] build_model() arunca exceptie corect pentru arhitectura invalida")


# ============================================================
# TEST 2: train_model returneaza None daca dataset lipseste
# ============================================================

def test_train_model_missing_dataset():
    result = train_model(
        dataset_path="C:/cale/inexistenta/dataset.yaml",
        epochs=1,
    )
    assert result is None, "train_model() trebuia sa returneze None pentru dataset lipsa"
    print("  [PASS] train_model() returneaza None pentru dataset inexistent")


def test_train_model_returns_artifacts():
    """Testeaza cu un mock ca sa nu facem training real."""
    mock_results = MagicMock()
    mock_results.results_dict = {
        "metrics/mAP50(B)": 0.85,
        "metrics/mAP50-95(B)": 0.60,
    }

    mock_model = MagicMock()
    mock_model.train.return_value = mock_results

    with patch("src.training.build_model", return_value=mock_model), \
         patch("src.training.Path.exists", return_value=True):

        artifacts = train_model(
            dataset_path="fake/dataset.yaml",
            epochs=1,
            run_name="test_run",
        )

    assert artifacts is not None, "train_model() trebuia sa returneze artifacts"
    assert "best_model" in artifacts, "artifacts trebuie sa contina 'best_model'"
    assert "final_model" in artifacts, "artifacts trebuie sa contina 'final_model'"
    assert "run_dir" in artifacts, "artifacts trebuie sa contina 'run_dir'"
    print("  [PASS] train_model() returneaza artifacts cu cheile corecte")


# ============================================================
# TEST 3: parse_args parseaza corect argumentele
# ============================================================

def test_parse_args_defaults():
    """Testeaza valorile default ale argumentelor."""
    with patch("sys.argv", ["training.py", "--dataset", "fake/dataset.yaml"]):
        args = parse_args()

    assert args.dataset == "fake/dataset.yaml"
    assert args.epochs == 10
    assert args.batch_size == 4
    assert args.lr == 0.01
    assert args.architecture == "yolov8n"
    assert args.device == "cpu"
    print("  [PASS] parse_args() are valorile default corecte")


def test_parse_args_custom():
    """Testeaza argumente custom."""
    with patch("sys.argv", [
        "training.py",
        "--dataset", "C:/date/kitti.yaml",
        "--epochs", "25",
        "--batch-size", "8",
        "--architecture", "yolov8s",
        "--run-name", "test_experiment",
    ]):
        args = parse_args()

    assert args.epochs == 25
    assert args.batch_size == 8
    assert args.architecture == "yolov8s"
    assert args.run_name == "test_experiment"
    print("  [PASS] parse_args() parseaza argumente custom corect")


# ============================================================
# TEST 4: main() returneaza 0 sau 1
# ============================================================

def test_main_returns_1_on_missing_dataset():
    """main() trebuie sa returneze 1 daca dataset lipseste."""
    with patch("sys.argv", ["training.py", "--dataset", "inexistent/dataset.yaml"]):
        result = main()
    assert result == 1, f"main() trebuia sa returneze 1, a returnat {result}"
    print("  [PASS] main() returneaza 1 pentru dataset inexistent")


def test_main_returns_0_on_success():
    """main() trebuie sa returneze 0 la succes (cu mock)."""
    mock_artifacts = {
        "best_model": "runs/test/weights/best.pt",
        "final_model": "runs/test/weights/last.pt",
        "run_dir": "runs/test/",
    }

    with patch("sys.argv", ["training.py", "--dataset", "fake/dataset.yaml"]), \
         patch("src.training.train_model", return_value=mock_artifacts):
        result = main()

    assert result == 0, f"main() trebuia sa returneze 0, a returnat {result}"
    print("  [PASS] main() returneaza 0 la succes")


# ============================================================
# RUNNER
# ============================================================

if __name__ == "__main__":
    tests = [
        test_build_model_returns_model,
        test_build_model_invalid_architecture,
        test_train_model_missing_dataset,
        test_train_model_returns_artifacts,
        test_parse_args_defaults,
        test_parse_args_custom,
        test_main_returns_1_on_missing_dataset,
        test_main_returns_0_on_success,
    ]

    passed = 0
    failed = 0

    print("\n" + "=" * 45)
    print("RUNNING TESTS - training_tests.py")
    print("=" * 45)

    for test in tests:
        print(f"\n{test.__name__}")
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print("\n" + "=" * 45)
    print(f"REZULTAT: {passed} passed, {failed} failed")
    print("=" * 45)

    sys.exit(0 if failed == 0 else 1)
    