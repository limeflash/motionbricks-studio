// Bundles src/card.js + src/card.css into dist/card.html (the chat card) and src/editor.js + src/editor.css into
// dist/editor.html (the pose editor page): one file each, which the Go server embeds. Run: npm run build (or go
// generate in the repo root).
import { build } from "esbuild";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";

const version = (readFileSync("../main.go", "utf8").match(/serverVersion\s*=\s*"([^"]+)"/) || [])[1] || "dev";
mkdirSync("dist", { recursive: true });
for (const name of ["card", "editor"]) {
  const out = await build({
    entryPoints: [`src/${name}.js`],
    bundle: true,
    format: "iife",
    minify: true,
    write: false,
    target: "es2022",
    legalComments: "eof",
    define: { __VERSION__: JSON.stringify(version) },
  });
  const js = out.outputFiles[0].text.replace(/<\/script/gi, "<\\/script");
  const css = readFileSync(`src/${name}.css`, "utf8");
  const html = readFileSync(`src/${name}.html`, "utf8")
    .replace("/*CSS*/", () => css)
    .replace("/*JS*/", () => js);
  writeFileSync(`dist/${name}.html`, html);
  console.log(`dist/${name}.html ${(html.length / 1024).toFixed(0)} KB (v${version})`);
}
