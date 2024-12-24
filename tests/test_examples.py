from pathlib import Path

from _pytest.capture import CaptureFixture
from _pytest.fixtures import FixtureRequest

from tests.util.pytest import run_module_function

ROOT = Path(__file__).parent.parent
EXAMPLES_FOLDER = ROOT / "examples"


def test_dummy(request: FixtureRequest, capsys: CaptureFixture) -> None:
    outcome = run_module_function(
        request=request, filepath=EXAMPLES_FOLDER / "dummy.py"
    )
    assert isinstance(outcome[0], Path)
    assert outcome[0].name == "dummy.py"
    assert outcome[1] == 0
    assert outcome[2] == "test_dummy.main"

    out, err = capsys.readouterr()
    assert out == "Hallo, Räuber Hotzenplotz.\n"
