# %% [markdown]
# # RSNA Knee Real Data Inspection
#
# Private, CPU-only, internet-disabled evidence gathering for issue #15.
# This notebook samples headers and pixels; it never decodes the full corpus.

# %%
import base64
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
package_archive = Path("/kaggle/working/rsna_knee_source.zip")
package_archive.write_bytes(base64.b64decode("__RSNA_KNEE_PACKAGE_ARCHIVE__"))
sys.path.insert(0, str(package_archive))

# %%
from rsna_knee import inspect_competition

outcome = inspect_competition(
    input_root=Path("/kaggle/input"),
    work_root=Path("/kaggle/working"),
)
print(outcome.artifact_path)

# %%
import json

print(json.dumps(outcome.findings, indent=2, sort_keys=True))

# %%
from IPython.display import Image, display  # type: ignore[import-not-found]

for relative_path in outcome.findings["visualizations"]["paths"]:
    display(Image(filename=outcome.artifact_path.parent / relative_path))
