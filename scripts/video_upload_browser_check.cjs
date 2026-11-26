// Content validation of multiple containers via the real frontend; no inference.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.CONSOLE_TEST_PLAYWRIGHT);
(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CONSOLE_TEST_CHROMIUM,
    headless: true, args: ["--no-sandbox"] });
  const results = [];
  try {
    const page = await browser.newPage({ baseURL: process.env.CONSOLE_TEST_URL || "http://127.0.0.1:5182/" });
    await page.addInitScript(() => {
      window.uploadTestResults = [];
      const originalSend = XMLHttpRequest.prototype.send;
      XMLHttpRequest.prototype.send = function (...args) {
        this.addEventListener("load", () => {
          if (this.responseURL.endsWith("/api/t1/uploads/")) window.uploadTestResults.push({ status: this.status, body: JSON.parse(this.responseText) });
        });
        return originalSend.apply(this, args);
      };
    });
    await page.goto("/");
    const directory = process.env.WAYCON_UPLOAD_TEST_VIDEOS;
    assert(directory);
    for (const extension of ["mp4", "mov", "avi", "mkv"]) {
      await page.getByLabel("Video file").setInputFiles(path.join(directory, `tiny.${extension}`));
      await page.getByRole("button", { name: "Upload Video", exact: true }).click();
      await page.getByText("Validated · Ready for analysis").waitFor();
      const uploaded = await page.evaluate(() => window.uploadTestResults.at(-1));
      assert.equal(uploaded.status, 201);
      const item = uploaded.body;
      assert.equal(item.total_frames, 2);
      results.push(item);
    }
    assert.equal(new Set(results.map(r => r.upload_id)).size, 4);
    fs.writeFileSync(path.join(directory, "browser-uploads.json"), JSON.stringify({ status: "passed", results }, null, 2));
    console.log(JSON.stringify({ status: "passed", formats: 4, unique_uploads: 4 }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
