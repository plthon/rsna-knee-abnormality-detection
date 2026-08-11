# %% [markdown]
# # RSNA Knee Data Index
#
# Private, CPU-only, internet-disabled indexing and representative preprocessing
# for issue #5. DICOM headers are fully indexed before representative pixels are
# decoded. No source or decoded image is written to the output artifact.

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
from rsna_knee import (
    PreprocessingConfig,
    build_competition_index,
    verify_index_artifact,
)

outcome = build_competition_index(
    input_root=Path("/kaggle/input"),
    work_root=Path("/kaggle/working"),
    config=PreprocessingConfig(
        slice_count=8,
        lower_percentile=1,
        upper_percentile=99,
        image_size=224,
    ),
)
verified_statistics = verify_index_artifact(outcome.artifact_dir)
print(outcome.manifest_path)

# %%
import json

print(json.dumps(verified_statistics, indent=2, sort_keys=True))
