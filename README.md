# 3apps

A tiny static landing site hosting Tom's small self-made web apps. No build step, no external CDNs.

## Structure
```
index.html          landing page (project list lives in the PROJECTS array)
assets/             card images (screenshots)
3notion/index.html  3notion — Notion-like offline notes app
3sticker/           3sticker — image → sticker maker (on-device AI, models self-hosted in 3sticker/models)
```

## Add a project
1. Create a folder, e.g. `myapp/`, with an `index.html` (self-contained).
2. Put a screenshot in `assets/myapp.png` (16:10 looks best).
3. Add an entry to the `PROJECTS` array in `index.html`:
```js
{ name: "myapp", url: "myapp/", image: "assets/myapp.png",
  description: "One-line description.", tags: ["Tag 1", "Tag 2"] }
```

## Run locally
`python3 -m http.server 8000` in this folder, then open http://localhost:8000.

## Deploy
Any static host works: upload the whole folder (GitHub Pages, Netlify Drop, Cloudflare Pages).
