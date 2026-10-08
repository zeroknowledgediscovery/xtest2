#!/usr/bin/env python3
import json, subprocess, os
from pathlib import Path
BASE=Path(os.environ.get("WORKSPACE_DIR","/workspace"))
script=BASE/"output"/"solution.py"
assert script.is_file(),"Missing solver script"
subprocess.run(["python3",str(script)],cwd=BASE,timeout=165,check=True)
submitted=json.loads((BASE/"output"/"solution.json").read_text())
expected=json.loads((Path(os.environ.get("TESTS_DIR","/tests"))/"answer.json").read_text())
assert isinstance(submitted,dict) and sorted(submitted)==["polarities"]
assert type(submitted["polarities"]) is list and len(submitted["polarities"])==48
assert all(type(x) is int and x in (-1,1) for x in submitted["polarities"])
assert submitted["polarities"][0]==1
assert submitted["polarities"]==expected["polarities"],"Incorrect polarity assignment"
assert (BASE/"output"/"analysis.md").is_file(),"Missing analysis.md"
print("PASS: exact canonical polarity assignment")
