export function cropPath(runId, trackId, kind, crop) {
  const expected = `/api/t1/runs/${runId}/tracks/${trackId}/crops/${kind}/`;
  return crop?.status === "available" && crop.url === expected ? expected : null;
}

export function plateReasonLabel(plate) {
  const reasons = { no_plate_detected: "No plate detected", no_detection: "No plate detected", not_associated: "Not associated", crop_rejected: "Crop rejected",
    ocr_not_attempted: "Plate detected, OCR not attempted", no_eligible_crop: "Plate detected, no eligible crop",
    ocr_rejected: "OCR attempted, reading rejected", recognition_error: "Recognition failed",
    stopped_before_ocr: "Stopped before OCR", crop_retention_limit: "Plate detected, crop retention limit reached", recognized: "Plate read" };
  return reasons[plate?.diagnostic?.reason] || "Unknown";
}

export function sourceFrame(frame) {
  return Number.isInteger(frame) && frame >= 0 ? `Source frame ${frame}` : "Source frame unknown";
}

export function cropOrigin(kind, crop) {
  return kind === "plate" ? "Selected plate" : ["manufacturer", "engine_selected"].includes(crop?.source) ? "Manufacturer input" : "Vehicle observation";
}

export function brandName(result) {
  if (result?.status === "accepted") {
    const make = result.make || "unknown";
    return ["bmw", "gmc", "mg", "byd"].includes(make) ? make.toUpperCase() : make.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }
  const reasons = { insufficient_quality: "No usable view", stopped_before_submission: "Stopped", InputStopped: "Stopped" };
  const names = { disabled: "Disabled", unknown: "Unknown", recognized: "Unknown", error: "Unavailable", timed_out: "Timed out", malformed: "Unavailable", failed: "Unavailable", execution_uncertain: "Uncertain", cancelled: "Cancelled", pending: "Pending" };
  return reasons[result?.reason] || names[result?.status] || "Unknown";
}
