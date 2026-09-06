const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '@playwright/test');
const fs = require('node:fs');
const assert = require('node:assert/strict');
(async () => {
 const browser = await chromium.launch({headless:true});
 const errors = [], report = [];
 fs.mkdirSync('renders', {recursive:true});
 for (const width of [320,390,1024,1440]) for (const colorScheme of ['light','dark']) {
  const context = await browser.newContext({viewport:{width,height:900},colorScheme,reducedMotion:'reduce'});
  const page = await context.newPage();
  page.on('pageerror',e=>errors.push(e.message));
  page.on('console',m=>{if(m.type()==='error')errors.push(m.text())});
  page.on('requestfailed',r=>errors.push(r.url()));
  await page.goto('http://127.0.0.1:4879/',{waitUntil:'load'});
  await page.evaluate(()=>document.fonts.ready);
  assert.equal(await page.locator('h1').count(),1);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'horizontal overflow '+width+' '+JSON.stringify(await page.evaluate(()=>[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth && getComputedStyle(e).position!=='absolute' && getComputedStyle(e).position!=='fixed').map(e=>({tag:e.tagName,cl:e.className,right:e.getBoundingClientRect().right})).slice(0,20))));
  await page.screenshot({path:`renders/r1-home-${width}-${colorScheme}.png`,fullPage:true});
  if(width===1440&&colorScheme==='dark')await page.screenshot({path:'renders/r1-hero.png'});
  const broken = await page.evaluate(()=>[...document.querySelectorAll('a[href^="#"]')].filter(a=>!document.querySelector(a.getAttribute('href'))).map(a=>a.href));
  assert.deepEqual(broken,[]);
  await page.getByRole('tab',{name:'CLI',exact:true}).click();
  assert(await page.locator('#example-cli').isVisible());
  await page.getByRole('tab',{name:'CLI',exact:true}).press('ArrowRight');
  assert(await page.locator('#example-api').isVisible());
  await page.getByRole('tab',{name:'HTTP API',exact:true}).press('Home');
  assert(await page.locator('#example-go').isVisible());
  await context.grantPermissions(['clipboard-read','clipboard-write']);
  await page.locator('.code-copy').click();
  assert((await page.evaluate(()=>navigator.clipboard.readText())).includes('package main'));
  await page.locator('.install-copy').click();
  assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),'go get github.com/zerfoo/zerfoo');
  await page.locator('.faq summary').first().click();
  assert(await page.locator('.faq details').first().getAttribute('open')!==null);
  if(width<=390){
   await page.locator('.menu-toggle').click();
   assert.equal(await page.evaluate(()=>document.body.style.overflow),'hidden');
   await page.keyboard.press('Escape');
   assert.equal(await page.locator('.menu-toggle').getAttribute('aria-expanded'),'false');
   await page.locator('.menu-toggle').click();
   await page.locator('.nav-links a').first().click();
   assert.equal(await page.evaluate(()=>document.body.style.overflow),'');
  }
  const links=await page.locator('a[href^="/"]').evaluateAll(as=>[...new Set(as.map(a=>a.getAttribute('href')))]);
  if(width===1440&&colorScheme==='dark')for(const link of links){const r=await page.request.get('http://127.0.0.1:4879'+link);assert.equal(r.status(),200,link);}
  await page.goto('http://127.0.0.1:4879/docs/getting-started/quickstart/',{waitUntil:'load'});
  await page.evaluate(()=>document.fonts.ready);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'docs overflow '+width);
  await page.screenshot({path:`renders/r1-docs-${width}-${colorScheme}.png`,fullPage:true});
  report.push({width,colorScheme,checks:'render, overflow, tabs, keyboard, clipboard, FAQ, docs; mobile menu where applicable'});
  await context.close();
 }
 const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'no-preference'});
 const page=await context.newPage();await page.goto('http://127.0.0.1:4879/');
 const x=await page.locator('#beam-gradient').getAttribute('x1');await page.waitForTimeout(200);
 assert.notEqual(await page.locator('#beam-gradient').getAttribute('x1'),x,'beam animates');
 await page.emulateMedia({reducedMotion:'reduce'});
 assert.equal(await page.locator('.beam-svg').isVisible(),false,'reduced motion');
 await context.close();
 assert.deepEqual(errors,[]);
 fs.writeFileSync('renders/check-results.json',JSON.stringify({report,errors,animation:'passed'},null,2));
 console.log(JSON.stringify({captureConfigurations:report.length,consoleErrors:errors.length,interactions:'passed',localLinks:'passed',motion:'passed'}));
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
