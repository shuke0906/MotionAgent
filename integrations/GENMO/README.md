# External GENMO Integration

GENMO is external, not a vendored repository. The compatibility patch is generated against official upstream revision `16bebf402d8893184249ee206d957b8248cd8310` and contains existing GEM changes: reproducible seed forwarding, constraint conditioning, and inference validation. The pure-text adapter is in `gem/motionagent.py`. This does not change MotionAgent architecture or train GEM.

From the MotionAgent root, after cloning upstream GENMO:

```bash
git -C vendor/GENMO checkout 16bebf402d8893184249ee206d957b8248cd8310
git -C vendor/GENMO apply ../../integrations/GENMO/compatibility.patch
cp -r integrations/GENMO/gem/. vendor/GENMO/gem/
cp -r integrations/GENMO/scripts/. vendor/GENMO/scripts/
python vendor/GENMO/scripts/download_gvhmr_support.py
```

Apply the patch only to a clean matching revision. Download GEM and licensed SMPL-X separately. No full upstream checkout, checkpoints, or downloaded support tensors are included.

GENMO-derived files retain NVIDIA notices and the [NVIDIA OneWay Noncommercial license](LICENSE). Integration files do not remove upstream restrictions. The downloader fetches checksum-verified external GVHMR tensors; they are not included in this patch set.
