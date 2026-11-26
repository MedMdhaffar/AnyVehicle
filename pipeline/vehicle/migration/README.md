# Migration compatibility package

This package holds shared code retained from the former Django recognition apps:

| Package | Responsibility |
|---|---|
| `attributes/` | Earlier classification, profile, consensus and observation helpers |
| `plate/` | Earlier plate scalar policies, evidence types and smoke-tool interfaces |
| `colour/` | Colour evidence, scalar/temporal policies and legacy smoke-tool interfaces |

The shared engine, DB1 development fixture and regression tools still import some
of these modules. Keep this package even after removing the former Django app
folders. This is unrelated to Django's schema-migration mechanism.

Active recognition code, LFM/ONNX/OCR assets and default tracker configuration
remain under `vehicle/recognition/`. The compatibility packages retain their
legacy test/sample assets where needed. The normal console continues to obtain
manufacturer, body type and colour from LFM; relocating colour helpers does not
enable the old HSV or external-agent inference paths.

The former `vehicle/license_plate_service/`,
`vehicle/vehicle_classification_manufacturer_service/` and
`vehicle/vehicle_color_service/` packages are no longer dependencies of the normal
console. Historical pinned external source checkouts used by opt-in audit tools
keep their original layout.
