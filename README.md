# Variance reduction and adaptive sampling at fixed inference and training compute

The blog post lives in `index.md`. This repository is a small Jekyll site (Markdown + MathJax), built and
served by GitHub Pages. The layout and styles follow `subliminal-watermarks.github.io`. The experiment
code lives in the `adaptive-sampling` repository.

## Preview locally

```bash
bundle install                 # once
bundle exec jekyll serve       # then open http://localhost:4000/science-adaptive-sampling/
```

## Layout

| Path | What it is |
|---|---|
| `index.md` | the post |
| `_layouts/article.html` | title, subtitle, authors, date, draft banner |
| `_layouts/default.html` | page shell, MathJax, auto-sizing for embedded figures |
| `assets/css/publication.css` | all styles; post-specific additions are at the end |
| `assets/anim/water-filling.html` | Figure 2, a self-contained interactive embedded with an `<iframe>` |
| `assets/images/` | static figures |

## Writing

- Math: `$$...$$` works both inline and as a display block (blank lines around it).
  Inside raw HTML (figure captions, table cells) use `\( ... \)` instead, because kramdown does not
  convert math there.
- Inside a table, write `|` in math as `\vert` or `\mid`, or the table breaks.
- Pending numbers are `<span class="todo-inline">XXX</span>`; find them all with `grep -n todo index.md`.
- Figures go in `assets/images/` and are linked as `{{ '/assets/images/fig.png' | relative_url }}`.
- An embedded interactive posts its own height to the page
  (`parent.postMessage({type: 'anim-height', h}, '*')`) so the iframe fits its content.

## Publishing checklist

1. Fill every `XXX`: critical batch size (128-tape replay) and the GRPO comparison.
2. Settle the author list in the front matter of `index.md`.
3. Remove `draft: true` and set `date:`.
4. Push to `main` in `Chang-Han-Chen/science-adaptive-sampling`. GitHub Pages deploys from the repository root.
   The site appears at https://chang-han-chen.github.io/science-adaptive-sampling/.
5. Link to it from the personal site's blog.
