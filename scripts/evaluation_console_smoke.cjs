// Opt-in real prerecorded video through the existing frontend Start workflow.
const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const {chromium}=require(process.env.CONSOLE_TEST_PLAYWRIGHT);
(async () => {
  assert(process.env.WAYCON_EVALUATION_VIDEO && process.env.WAYCON_EVALUATION_OUTPUT);
  const output=process.env.WAYCON_EVALUATION_OUTPUT;fs.mkdirSync(output,{recursive:true,mode:0o700});
  const browser=await chromium.launch({executablePath:process.env.CONSOLE_TEST_CHROMIUM,headless:true,args:["--no-sandbox"]});
  try {
    let page=await browser.newPage({baseURL:process.env.CONSOLE_TEST_URL || "http://127.0.0.1:5182/",viewport:{width:1500,height:1080}});
    page.setDefaultTimeout(20000);const errors=[];page.on("pageerror",e=>errors.push(e.message));
    await page.goto("/");
    if (!process.env.WAYCON_EVALUATION_EXISTING_RUN) {
    await page.getByLabel("Video file").setInputFiles(process.env.WAYCON_EVALUATION_VIDEO);
    await page.getByRole("switch").check();
    await page.getByRole("button",{name:"Upload Video",exact:true}).click();
    await page.getByText("Validated · Ready for analysis").waitFor();
    assert.equal(await page.getByLabel("Timeout",{exact:true}).inputValue(), "");
    await page.getByRole("button",{name:"Start Analysis"}).click();
    // Detach the browser page immediately; the server owns the worker.
    await page.waitForFunction(async () => (await (await fetch("/api/t1/runs/")).json()).runs[0]?.active_slot);
    await page.close();
    page=await browser.newPage({baseURL:process.env.CONSOLE_TEST_URL || "http://127.0.0.1:5182/",viewport:{width:1500,height:1080}});
    await page.goto("/");
    }
    const deadline=Date.now()+180000;let run;
    while(Date.now()<deadline) {
      const response=await page.request.get("/api/t1/runs/");
      const runs=(await response.json()).runs;run=runs.find(r => r.run_id === process.env.WAYCON_EVALUATION_EXISTING_RUN) || runs[0];
      if(run && ["failed","interrupted"].includes(run.status))throw new Error("console run failed: "+run.error_code);
      if(run && run.status==="completed" && !run.active_slot)break;
      await page.waitForTimeout(500);
    }
    assert(run && run.status==="completed" && !run.active_slot);
    assert.equal(run.end_reason,"eof");assert.equal(run.analyzed_frames,8);
    assert.equal(run.delivery.postgres_commit, "committed");
    assert.equal(run.delivery.redis_ack, "acknowledged");
    assert(run.upload?.upload_id);
    await page.getByLabel("Run history").selectOption(run.run_id);
    await page.getByRole("button",{name:"Open evaluation results"}).click();
    await page.waitForFunction(id => document.querySelector('[aria-label="Evaluation run"]')?.value === id,run.run_id);
    await page.locator(".evaluation-card").first().waitFor();
    const response=await page.request.get(`/api/t1/evaluation/runs/${run.run_id}/`);
    const data=await response.json();
    const predictions=data.records.filter(r=>r.kind==="prediction" && r.domain==="manufacturer");
    assert(predictions.length>0 && predictions.every(r=>r.crop_reference && r.inference_time_ms>0));
    assert(data.calls.some(c=>c.stage==="manufacturer_queue"));
    assert(data.records.every(r=>r.review.verdict==="UNREVIEWED"));
    await page.locator(".evaluation-card img").evaluateAll(async nodes=>Promise.all(nodes.map(async node=>{node.loading="eager";await node.decode();})));
    await page.evaluate(()=>window.scrollTo(0,0));
    await page.screenshot({path:path.join(output,"console-evaluation.png"),fullPage:true});
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(output,"console-result.json"),JSON.stringify({status:"passed",run_id:run.run_id,
      end_reason:run.end_reason,analyzed_frames:run.analyzed_frames,manufacturer_predictions:predictions.length,
      records:data.records.length,upload:run.upload,browser_disconnection:true,delivery:run.delivery,queue_timing:true,all_unreviewed:true,metrics:data.metrics,cleanup:run.cleanup},null,2));
    console.log(JSON.stringify({status:"passed",run_id:run.run_id,records:data.records.length}));
  } finally {await browser.close();}
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
