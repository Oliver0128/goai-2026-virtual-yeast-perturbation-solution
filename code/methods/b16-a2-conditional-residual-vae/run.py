import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "_shared"))

from runner import main

try:
    raise SystemExit(main())
except Exception as error:
    manifest_path: Path | None = None
    if "--manifest" in sys.argv:
        manifest_index = sys.argv.index("--manifest") + 1
        if manifest_index < len(sys.argv):
            manifest_path = Path(sys.argv[manifest_index])
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path = manifest_path.parent / "failure.json"
        failure_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "failed_at": datetime.now().astimezone().isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "preserved_for_repair_and_rerun": True,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    raise
