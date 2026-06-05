# Editorial UI Restyle — Design

**Date:** 2026-06-05
**Scope:** Web UI restyle of the two user-facing views — **Ask** and **Check** — in the
Esteem DocLens FastAPI app (`static/index.html`, `static/style.css`, `static/app.js`).
Admin views (Documents, Ingest, Jobs, Inspect) are explicitly **out of scope** for this pass.

## Goal

A bolder, "Editorial / document-forward" visual restyle of the user-facing app: treat it
as a reading instrument for engineers rather than a generic SaaS tool. Replace the cool,
flat, somewhat sterile teal-on-grey look with a warm-paper editorial aesthetic, fix the
dead-space empty state, and give chat answers real typographic and citation treatment.

This is a **CSS-led restyle plus small markup/JS touches** — no framework, no bundler. The
existing application logic, API calls, routing, and event wiring stay as they are.

## Decisions (locked during brainstorming)

- **Views:** Ask + Check only. Admin views untouched (and remain admin-gated).
- **Build:** Stay buildless — plain HTML/CSS/JS + CDN assets. No framework/bundler.
- **Direction:** Editorial / document-forward.
- **Theme:** Light only for this pass. Dark mode deferred, but tokens are structured so a
  later dark theme is a token swap, not a rewrite.
- A real throwaway mockup of the Editorial direction (Ask empty/chat + Check) was built and
  approved before writing this spec. It is the visual reference for implementation.

## Design system (the real deliverable)

The screens demonstrate a token system that must be implemented as CSS custom properties on
`:root`, so the look is consistent and a future dark theme is a token swap.

### Palette (warm paper, light)
| Token | Value | Use |
|-------|-------|-----|
| `--paper` | `#faf8f3` | App background |
| `--surface` | `#fffdf9` | Cards, rail, inputs |
| `--surface-2` | `#f4f1e9` | Hover fills, subtle wells |
| `--ink` | `#20201d` | Primary text |
| `--ink-soft` | `#56524a` | Secondary text |
| `--muted` | `#8c8678` | Meta, placeholders, counts |
| `--line` | `#e9e3d6` | Hairline borders/dividers |
| `--line-strong` | `#ddd5c4` | Input borders, dashed dropzones |
| `--accent` | `#0f5d54` | Primary accent (deep teal-green) |
| `--accent-deep` | `#0a463f` | Buttons, brand mark, send |
| `--accent-tint` | `#e7efec` | Active fills, focus rings |
| `--gold` | `#9a7a1e` | Source/citation accent |
| `--gold-tint` | `#f3ecd8` | Clause pill background |
| `--danger` | `#9c3326` | Errors/destructive |
| `--shadow` | `0 1px 2px rgba(40,34,20,.05), 0 8px 24px -16px rgba(40,34,20,.18)` | Card elevation |

Spacing, radii (cards ~14px, controls ~9–11px, pills 999px) and the shadow are also tokens.

### Typography (Google Fonts, no build)
- **Display** — `Fraunces` (opsz variable): brand name, folder title, hero headline, card headings.
- **Reading / serif body** — `Newsreader`: chat questions & answers, source snippets, ledes, placeholders.
- **UI chrome** — `Inter`: buttons, labels, sidebar items, counts, kickers.
- Tabular figures for numeric counts (`font-variant-numeric: tabular-nums`).

### Iconography
Inline SVG only (stroke icons: folder, search, plus, arrow-right, upload, send, check). No
icon-font or JS icon dependency. Icons live in the markup or a tiny JS helper.

## Layout & component changes

### Shell / sidebar (`rail`)
- Warm `--surface` rail, 288px.
- Brand: rounded `--accent-deep` mark with `ED` in display face; name in display, meta in
  uppercase tracked Inter.
- "New folder" becomes an outlined button with a `+` glyph.
- Folder search gets an inline search glyph.
- Uppercase "FOLDERS" section label.
- Each folder row: folder glyph + ellipsized name + tabular count. Active row uses
  `--accent-tint` fill and `--accent-deep` text. (Existing per-folder `+` add affordance is
  preserved from current behavior.)

### Topbar
- Uppercase "FOLDER" kicker, folder title in Fraunces display, stats as labelled figures
  (`1 document · 80 pages · 0 pending`) with emphasised numbers.
- Ask / Check as understated **underline tabs** (accent underline on active), right-aligned.
- A hairline divider below the topbar.

### Ask — empty state (biggest win)
Replaces the current centered grey one-liner sitting in a large void.
- Kicker "ASK THE FOLDER".
- Serif display headline addressed to the active folder ("What would you like to know about
  <folder>?").
- A grounding lede referencing the folder's page count and the citations guarantee.
- The four example prompts become a **clickable index**: hairline-ruled rows, serif text,
  an arrow that nudges on hover; clicking populates/sends the composer. (Examples may be
  static defaults or pulled from existing example logic — wire to current "ask" handler.)
- The composer stays pinned at the bottom (card-style input, see below).

### Ask — conversation
- A centered reading column, max-width ~760px.
- **User message:** small "You" avatar chip + serif question text (18px).
- **Assistant message:** accent avatar (`E`), "DocLens" name, optional "· N sources" meta;
  answer rendered in Newsreader at a comfortable reading size/leading. Markdown continues to
  render via the existing marked + DOMPurify + KaTeX pipeline; only the typographic envelope
  changes.
- **Citations:** inline superscript markers (`.ref`) in accent, keyed to source cards.
- **Source cards:** gold left-rule card with the source number in display, a clause pill
  (`§ 6.2.1`) in gold-tint, document title + page meta, and the quoted snippet in serif.
  These map to whatever source/citation data the chat response already provides.
- **Composer:** rounded card textarea (serif, 16px) with an accent focus ring and a square
  accent icon-send button; a centered trust line beneath ("DocLens answers only from
  documents in this folder, with citations").

### Check
- Same structure and form fields as today; restyled only.
- Section/card headings in display face with a serif sub-label ("your submission", "tender /
  PO / PR", "in your words").
- Warm dashed dropzone with an upload glyph and "Drop files or **browse**".
- Custom-styled radio group (accent filled dot for selected).
- `What to check` textarea in serif.
- Confident `✓ Run check` button in `--accent-deep`.
- "Recent checks" list restyled to match (cards/rows on warm surface).

## Out of scope (YAGNI)
- Dark mode (deferred; tokens make it cheap later).
- Admin views (Documents, Ingest, Jobs, Inspect).
- Any backend/API/MCP change. This is presentation only.
- New JS framework, bundler, or build step.
- Mobile/responsive redesign beyond not breaking the current single-column fallback (the app
  is a desktop tool; we will avoid regressions but not design new breakpoints).

## Approach & risks

- **CSS-first.** Most of the work is rewriting `static/style.css` around the token system and
  the components above. The markup in `index.html` gets light additions (icons, kicker spans,
  section label). `app.js` changes are limited to: rendering the new empty-state markup,
  rendering assistant messages with the source-card/citation structure, and wiring example
  rows to the existing ask handler.
- **Risk — JS-generated markup.** Much of the Ask/Check DOM is built in `app.js`, not static
  HTML. Implementation must read the current render functions and restyle by changing the
  classes/structure they emit, keeping existing data flow and event handlers intact. Verify
  against the running app, not just the mockup.
- **Risk — citation/source data shape.** Source cards assume the chat response exposes clause,
  document, page, and snippet. Confirm what `app.js` actually receives; degrade gracefully
  (omit a pill if no clause, etc.) rather than inventing fields.
- **Risk — fonts/CDN.** Google Fonts + existing KaTeX/marked/DOMPurify all load from CDN,
  consistent with the current app. Include a system-serif/sans fallback stack.
- **Verification.** Re-run the live app on `docs.db` and screenshot Ask (empty + a real
  conversation) and Check, comparing against the approved mockup and the baseline.

## Reference artifacts
- Baselines: `baseline-ask.png`, `baseline-check.png` (current app).
- Approved mockup: `/tmp/ocr-mock/mock.html` → `mock-ask-empty.png`, `mock-ask-chat.png`,
  `mock-check.png`. (Throwaway; not committed to the app.)
