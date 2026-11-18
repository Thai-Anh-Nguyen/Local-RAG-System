// Build report/report.pdf from report/report.md.
// Usage (from the repo root):
//   npm install --prefix report/tools --no-save marked@15.0.12 puppeteer-core@23.11.1
//   node report/tools/build-pdf.mjs [path/to/chrome]
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { marked } from 'marked';
import puppeteer from 'puppeteer-core';

const here = dirname(fileURLToPath(import.meta.url));
const reportDir = resolve(here, '..');
const chrome = process.argv[2] || '/usr/bin/google-chrome';

const raw = readFileSync(join(reportDir, 'report.md'), 'utf8');
const fm = raw.match(/^---\n([\s\S]*?)\n---\n/);
const meta = {};
if (fm) for (const line of fm[1].split('\n')) {
  const m = line.match(/^(\w+):\s*"?(.*?)"?$/);
  if (m) meta[m[1]] = m[2];
}
const body = marked.parse(fm ? raw.slice(fm[0].length) : raw);
// Turn standalone images into captioned figures.
const html = body.replace(/<p><img src="([^"]+)" alt="(.*?)"><\/p>/g,
  '<figure><img src="$1" alt=""><figcaption>$2</figcaption></figure>');

const css = `
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
:root { --ink:#1f2937; --muted:#64748b; --accent:#1e3a8a; --rule:#e2e8f0; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Inter", "Noto Sans", "Helvetica Neue", Arial, sans-serif; color: var(--ink);
  font-size: 10.5pt; line-height: 1.5; margin: 0; background: #fff; }
.title { border-bottom: 3px solid var(--accent); padding-bottom: 10px; margin-bottom: 18px; }
.title .course { color: var(--accent); font-weight: 600; font-size: 10pt; letter-spacing: .02em; }
.title h1 { font-size: 21pt; line-height: 1.2; margin: 6px 0 6px; border: 0; padding: 0; }
.title .meta { color: var(--muted); font-size: 10pt; }
h1 { font-size: 15pt; color: var(--accent); margin: 22px 0 8px; padding-bottom: 4px; border-bottom: 1px solid var(--rule);
  break-after: avoid; }
h2 { font-size: 12pt; margin: 16px 0 6px; break-after: avoid; }
p, li { text-align: justify; hyphens: auto; }
ul, ol { padding-left: 20px; margin: 6px 0; }
li { margin: 2px 0; }
code { font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace; font-size: 8.8pt; background: #f1f5f9;
  padding: 0 3px; border-radius: 3px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 9pt; }
tr { break-inside: avoid; }
th, td { border: 1px solid var(--rule); padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #f1f5f9; }
tr:nth-child(even) td { background: #fafbfc; }
figure { margin: 10px 0 14px; break-inside: avoid; text-align: center; }
figure img { max-width: 88%; max-height: 82mm; border: 1px solid var(--rule); border-radius: 4px; }
figure.diagram svg { width: 100%; height: auto; }
figcaption { color: var(--muted); font-size: 9pt; margin-top: 4px; text-align: center; }
a { color: var(--accent); }
`;

const doc = `<!doctype html><html><head><meta charset="utf-8">
<base href="${pathToFileURL(reportDir + '/').href}"><title>${meta.title || 'Report'}</title>
<style>${css}</style></head><body>
<div class="title"><div class="course">${meta.subtitle || ''}</div><h1>${meta.title || ''}</h1>
<div class="meta">${[meta.author, meta.date].filter(Boolean).join(' · ')}</div></div>
${html}</body></html>`;

const htmlPath = join(here, 'report.html');
writeFileSync(htmlPath, doc);
const browser = await puppeteer.launch({ executablePath: chrome, headless: true });
const page = await browser.newPage();
await page.goto(pathToFileURL(htmlPath).href, { waitUntil: 'networkidle0' });
await page.pdf({
  path: join(reportDir, 'report.pdf'), format: 'A4', printBackground: true, preferCSSPageSize: true,
  displayHeaderFooter: true,
  headerTemplate: '<div></div>',
  footerTemplate: `<div style="width:100%;font-size:8px;color:#64748b;padding:0 18mm;display:flex;justify-content:space-between;">
    <span>CS311 Advanced Deep Learning · Final Project</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>`,
});
await browser.close();
console.log('wrote', join(reportDir, 'report.pdf'));
