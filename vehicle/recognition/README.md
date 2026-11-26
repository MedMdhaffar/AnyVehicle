# Recognition package layout

The normal console loads recognition code and default assets from this package:

```text
attributes/
  models/              Vehicle detector, LFM GGUF/projector and optional HF model
  trackers/            Vehicle ByteTrack settings
  manufacturer/        Active LFM recognition and crop selection
plate/
  models/              CUDA ONNX detector, marker detector and registration OCR
  trackers/            Plate ByteTrack settings
  license_plate_plugin/
    configuration/     Package-relative category configuration
    assets/            DINOv3, category OCR, reference images/index and vendor source
```

Shared earlier policies and smoke-tool compatibility live in the sibling
`vehicle/migration/` package, with `attributes/`, `plate/` and `colour/`
subpackages. They are not Django applications. Active plate and LFM recognition
policies are unchanged. Explicit environment model and tracker overrides remain
supported.

The former `vehicle/license_plate_service/` and
`vehicle/vehicle_classification_manufacturer_service/` and
`vehicle/vehicle_color_service/` Django apps are no longer
local import or asset dependencies of the console. Historical pinned external
source checkouts used by opt-in audit tools retain their original source layout.
