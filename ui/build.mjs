// Bundles src/card.js and src/card.css into dist/card.html, one file that the
// Go server embeds. Run: npm run build (or go generate in the repo root).
import { build } from "esbuild";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";

const version = (readFileSync("../main.go", "utf8").match(/serverVersion\s*=\s*"([^"]+)"/) || [])[1] || "dev";
const out = await build({
  entryPoints: ["src/card.js"],
  bundle: true,
  format: "iife",
  minify: true,
  write: false,
  target: "es2022",
  legalComments: "eof",
  define: { __VERSION__: JSON.stringify(version) },
});
const js = out.outputFiles[0].text.replace(/<\/script/gi, "<\\/script");
const css = readFileSync("src/card.css", "utf8");
const html = readFileSync("src/card.html", "utf8")
  .replace("/*CSS*/", () => css)
  .replace("/*JS*/", () => js);
mkdirSync("dist", { recursive: true });
writeFileSync("dist/card.html", html);
console.log(`dist/card.html ${(html.length / 1024).toFixed(0)} KB (v${version})`);
