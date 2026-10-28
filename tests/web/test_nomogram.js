// Headless-browser checks for SPACE_Dynamic_Nomogram.html (input validation, model switching, Fig. 6 presets).
// Usage: npm install puppeteer-core && CHROME_PATH=/path/to/chrome node tests/web/test_nomogram.js
const puppeteer = require('puppeteer-core');
const path = require('path');
const file = path.resolve(process.argv[2] || path.join(__dirname, '..', '..', 'SPACE_Dynamic_Nomogram.html'));
let fails = 0;
const ok = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) fails++; };
(async () => {
  const b = await puppeteer.launch({ executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome', headless: 'new', args: ['--no-sandbox'] });
  const pg = await b.newPage();
  const errs = []; pg.on('pageerror', e => errs.push(e.message));
  await pg.goto('file://' + file);
  const st = () => pg.evaluate(() => ({
    results: !document.getElementById('results').hidden,
    risk: document.getElementById('risk').textContent, group: document.getElementById('group').textContent,
    err: document.getElementById('error').hidden ? '' : document.getElementById('error').textContent,
    v: Object.fromEntries(['mshscore','t_category','nodal_status','diameter_cm','histology','hemoglobin_g_dl'].map(k => [k, document.getElementById(k).value])),
  }));
  const setModel = m => pg.evaluate(m => { const s = document.getElementById('model'); s.value = m; s.dispatchEvent(new Event('change')); }, m);
  const type = async (id, val) => { await pg.click('#' + id, { clickCount: 3 }); await pg.keyboard.press('Backspace'); if (val) await pg.type('#' + id, val); };
  const pick = (id, val) => pg.select('#' + id, val);
  const calc = () => pg.click('button[type=submit]');

  // Presets x models
  const exp = { SPACE: ['8.9%', '19.5%', '41.2%'], 'SPACE-T2': ['11.4%', '15.6%', '26.4%'] };
  for (const m of ['SPACE', 'SPACE-T2']) {
    await setModel(m);
    for (const [i, L] of ['A', 'B', 'C'].entries()) {
      await pg.click('#preset' + L); const s = await st();
      ok(s.results && s.risk === exp[m][i], `${m} Patient ${L}: ${s.risk} ${s.group} (expected ${exp[m][i]})`);
    }
  }
  // Preset persists across model switch (preset still active)
  await setModel('SPACE'); await pg.click('#presetA'); await setModel('SPACE-T2');
  let s = await st(); ok(s.results && s.risk === '11.4%' && s.v.mshscore === '0.497', `Preset A then switch to SPACE-T2 reloads preset with MSHscore_T2W: ${s.risk}`);

  // Bug 1: blank MSHscore
  await setModel('SPACE'); await pg.click('#presetA');
  await type('mshscore', ''); await calc(); s = await st();
  ok(!s.results && /Incomplete input/.test(s.err), `Blank MSHscore -> no result; prompt: "${s.err}"`);
  await type('mshscore', '   '); await calc(); s = await st();
  ok(!s.results, 'Whitespace MSHscore -> no result');
  await pg.click('#presetA'); await type('diameter_cm', ''); await calc(); s = await st();
  ok(!s.results && /diameter/.test(s.err), `Blank diameter -> no result; prompt: "${s.err}"`);
  await pg.click('#presetA'); await type('hemoglobin_g_dl', ''); await calc(); s = await st();
  ok(!s.results && /haemoglobin/.test(s.err), 'Blank haemoglobin -> no result');

  // Bug 2: manual input then model switch
  await setModel('SPACE'); await pg.click('#presetA');
  await pick('t_category', 'T3'); await pick('nodal_status', 'Pelvic');
  await type('diameter_cm', '8'); await type('hemoglobin_g_dl', '9'); await type('mshscore', '0.55');
  await calc(); s = await st(); ok(s.results, `Manual input SPACE result shown: ${s.risk} ${s.group}`);
  await setModel('SPACE-T2'); s = await st();
  ok(s.v.t_category === 'T3' && s.v.nodal_status === 'Pelvic' && s.v.diameter_cm === '8' && s.v.hemoglobin_g_dl === '9',
     `After switch to SPACE-T2 clinical inputs kept: ${JSON.stringify(s.v)}`);
  ok(s.v.mshscore === '' && !s.results && /MSHscore_T2W/.test(s.err), `MSHscore cleared, no result, prompt: "${s.err}"`);
  await calc(); s = await st(); ok(!s.results, 'Calculate without new MSHscore_T2W -> still no result');
  await type('mshscore', '0.6'); await calc(); s = await st();
  ok(s.results && s.v.diameter_cm === '8', `After entering MSHscore_T2W=0.6 -> ${s.risk} ${s.group}`);
  // Editing after result hides stale result
  await type('hemoglobin_g_dl', '10'); s = await st(); ok(!s.results, 'Editing a field hides stale result');
  // Explicit preset still fully loads
  await pg.click('#presetB'); s = await st();
  ok(s.results && s.risk === '15.6%' && s.v.diameter_cm === '4.5' && s.v.t_category === 'T1b3-T2b', `Explicit preset B on SPACE-T2 fully loads: ${s.risk}`);
  ok(errs.length === 0, 'No page JS errors ' + errs.join('; '));
  await b.close();
  console.log(fails ? `${fails} FAILED` : 'ALL PASSED'); process.exit(fails ? 1 : 0);
})();
