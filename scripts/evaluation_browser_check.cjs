// Real HTTP/UI verification. Test verdicts are reset to UNREVIEWED in finally.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.CONSOLE_TEST_PLAYWRIGHT);

(async () => {
  const run = process.env.WAYCON_EVALUATION_RUN;
  const output = process.env.WAYCON_EVALUATION_OUTPUT;
  assert(run && output);
  fs.mkdirSync(output, { recursive: true, mode: 0o700 });
  const browser = await chromium.launch({ executablePath: process.env.CONSOLE_TEST_CHROMIUM,
    headless: true, args: ["--no-sandbox"] });
  const page = await browser.newPage({ baseURL:process.env.CONSOLE_TEST_URL || "http://127.0.0.1:5182/",
    viewport: { width: 1500, height: 1080 } });
  page.setDefaultTimeout(20000);
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const endpoint = `/api/t1/evaluation/runs/${run}/`;
  const data = () => page.evaluate(async (url) => (await fetch(url)).json(), endpoint);
  const touched = new Set();
  let completed = false;
  try {
    await page.goto(process.env.CONSOLE_TEST_URL || "http://127.0.0.1:5182/");
    await page.getByRole("button", { name: "Evaluation / Review" }).click();
    await page.getByLabel("Evaluation run").selectOption(run);
    await page.locator(".evaluation-card").first().waitFor();
    const original = await data();
    assert(original.records.every((r) => r.review.verdict === "UNREVIEWED"));
    const maker = original.records.find((r) => r.domain === "manufacturer" && r.kind === "prediction");
    const plate = original.records.find((r) => r.domain === "plate" && r.predicted_label?.includes("تونس"));
    assert(maker && plate && maker.crop_reference && maker.inference_time_ms > 0);
    let card = page.locator(`[data-evidence-id="${maker.id}"]`);
    const image = card.locator("img").first();
    await image.waitFor();
    await image.evaluate((node) => node.decode());
    assert(await image.evaluate((node) => node.naturalWidth > 50));
    await card.getByRole("button", { name: /Enlarge/ }).click();
    await page.getByRole("dialog").waitFor();
    await page.screenshot({ path: path.join(output,"image-inspection.png") });
    await page.getByRole("button", { name: "Close image inspector" }).click();
    touched.add(maker.id);
    await card.getByRole("radio", { name: "FALSE", exact: true }).check();
    await card.getByRole("button", { name: "Save review" }).click();
    await page.waitForFunction(async ({ url,id }) => (await (await fetch(url)).json()).records.find((r) => r.id === id).review.verdict === "FALSE", { url:endpoint,id:maker.id });
    const csvResponse = await page.request.get(endpoint+"download.csv");
    const csvText = (await csvResponse.body()).toString("utf8");
    assert(csvText.includes("FALSE") && csvText.includes(maker.predicted_label));
    await page.reload();
    await page.getByRole("button", { name: "Evaluation / Review" }).click();
    await page.waitForFunction((id) => document.querySelector('[aria-label="Evaluation run"]')?.value === id,run);
    assert.equal(await page.getByLabel("Evaluation run").inputValue(),run);
    card=page.locator(`[data-evidence-id="${maker.id}"]`);
    await card.waitFor();
    await page.waitForFunction((id) => document.querySelector(`[data-evidence-id="${id}"] input[value="FALSE"]`)?.checked,maker.id);
    await card.getByLabel(`Correct label for ${maker.id}`).fill('Review test, "corrected" label');
    await card.getByRole("button", { name: "Save review" }).click();
    await page.waitForFunction(async ({ url,id }) => (await (await fetch(url)).json()).records.find((r) => r.id === id).review.correct_label?.startsWith("Review test"), { url:endpoint,id:maker.id });
    await card.getByRole("radio", { name: "NOT CLEAR", exact: true }).check();
    await card.getByRole("button", { name: "Save review" }).click();
    await page.waitForFunction(async ({ url,id }) => (await (await fetch(url)).json()).records.find((r) => r.id === id).review.verdict === "NOT CLEAR", { url:endpoint,id:maker.id });
    assert(!(await (await page.request.get(endpoint+"download.csv")).text()).includes(maker.predicted_label));
    const plateCard=page.locator(`[data-evidence-id="${plate.id}"]`);
    touched.add(plate.id);
    await plateCard.getByRole("radio", { name: "TRUE", exact: true }).check();
    await plateCard.getByRole("button", { name: "Save review" }).click();
    await page.waitForFunction(async ({ url,id }) => (await (await fetch(url)).json()).records.find((r) => r.id === id).review.verdict === "TRUE", { url:endpoint,id:plate.id });
    const download=await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("link", { name: "Download CSV · TRUE + FALSE" }).click(),
    ]);
    await download[0].saveAs(path.join(output,"browser-export.csv"));
    assert(fs.readFileSync(path.join(output,"browser-export.csv"),"utf8").includes("تونس"));
    await page.getByLabel("Review filter").selectOption("NOT CLEAR");
    assert.equal(await page.locator(".evaluation-card").count(),1);
    await page.getByLabel("Review filter").selectOption("all");
    await page.getByRole("tab", { name: "Quality review", exact: true }).click();
    await page.locator(".evaluation-card").first().waitFor();
    assert(await page.locator(".evaluation-card").count()>2);
    const quality=original.records.find((r) => r.kind === "quality" && r.crop_reference);
    assert(quality);
    const qualityImage=await page.request.get(quality.crop_reference);
    assert.equal(qualityImage.status(),200);
    await page.getByRole("tab", { name: "Predictions", exact: true }).click();
    await page.screenshot({ path:path.join(output,"automated-review-test.png"),fullPage:true });
    assert.deepEqual(errors,[]);
    completed=true;
  } finally {
    try {
    if (touched.size) {
      const latest=await data();
      for (const row of latest.records.filter((r) => touched.has(r.id))) {
        const response=await page.request.post(endpoint+`records/${row.id}/review/`,{
          data:{verdict:"UNREVIEWED",revision:row.review.revision},
        });
        assert.equal(response.status(),200);
      }
    }
    const final=await data();
    assert(final.records.every((r) => r.review.verdict === "UNREVIEWED"));
    await page.reload();
    await page.getByRole("button", { name:"Evaluation / Review" }).click();
    await page.locator(".evaluation-card").first().waitFor();
    await page.locator(".evaluation-card img").evaluateAll(async (nodes) => {
      await Promise.all(nodes.map(async (node) => { node.loading="eager";await node.decode(); }));
    });
    await page.evaluate(() => window.scrollTo(0,0));
    await page.screenshot({ path:path.join(output,"evaluation-dashboard.png"),fullPage:true });
    fs.writeFileSync(path.join(output,"result.json"),JSON.stringify({
      status:completed ? "passed" : "failed",run_id:run,real_crop_images:true,
      saved_and_corrected_reviews:true,csv_download:true,arabic_preserved:true,
      final_reviews:"UNREVIEWED",automated_test_reviews_are_not_human_accuracy:true,page_errors:errors,
    },null,2));
    } finally { await browser.close(); }
  }
})().catch((error) => { console.error(error.stack);process.exitCode=1; });
