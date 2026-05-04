# Working With Claude Code — A Quick Guide

A short reference based on what tripped us up today. Save this; refer back to it whenever you start a session.

---

## 1. The Mental Model

Think of Claude Code as a **collaborator sitting at your computer** who can read, write, and run code in one specific folder. Two implications:

- **Whatever folder you start Claude in, that's the folder it works in.** Always start it from your project root: `/Users/kaeleyoshea/Capstone/A.MRdeeP-Deep-Learning--MRP/`.
- **It can only see what you've shown it.** Each session starts fresh. It doesn't remember what the previous session did unless you tell it (or unless something is saved in memory or `CLAUDE.md`).

---

## 2. Files Claude Looks At Automatically

These files are loaded into Claude's brain at the start of every session, without you asking. Knowing this is a superpower.

| File | What it does |
|---|---|
| `CLAUDE.md` (in project root) | Project context — goals, conventions, file structure, anything you want Claude to know upfront. **Convention: one CLAUDE.md at the top of your project.** |
| `~/.claude/CLAUDE.md` | Personal preferences that apply to *all* your projects (tone, coding style, etc.) |
| `.claude/settings.json` | Permissions, hooks, etc. — the technical config. You usually don't touch this manually. |
| Auto-memory files | Stuff Claude has saved across sessions about you and the project (lives at `~/.claude/projects/<project-name>/memory/`) |

**Where your setup strays slightly:** you have `claude.md_prompts/claude.md` instead of a top-level `CLAUDE.md`. That works because you're including the prompts folder in your messages — but a top-level `CLAUDE.md` would auto-load and save you from having to mention it. Easy fix: copy or symlink that file to the project root.

---

## 3. The Worktree Trap (What Bit Us)

A **worktree** is git's way of letting you have *parallel copies* of your project — useful when an agent is working on Task A while you work on Task B, with no overlap. Claude Code can auto-create them if launched with `--worktree`.

**For a solo researcher doing one thing at a time, worktrees are pure overhead.** They cause exactly the problem we hit: multiple folder copies that drift out of sync.

**Rule of thumb:** unless you have a specific reason to run multiple Claude sessions in parallel on the same project, never use `--worktree`. Just type `claude` in your project folder. The auto-memory I saved today will make future Claude sessions check `pwd` first and refuse to work if they're in a worktree.

---

## 4. Slash Commands You'll Actually Use

Type these in the chat input:

| Command | What it does |
|---|---|
| `/help` | List of available commands |
| `/clear` | Wipe the current conversation and start fresh (keeps the project open) |
| `/init` | Generate a starter `CLAUDE.md` from your existing code — good if you don't have one yet |
| `/model` | Switch which Claude model you're using (Opus, Sonnet, Haiku) |
| `/config` | Open the settings UI |
| `/cost` | See how much you've spent in this session |

You can ignore the rest until you need them.

---

## 5. Communicating Better With Claude

Things that came up today and would have helped:

**Tell Claude what you've changed.** When you replace a CSV or refactor something between sessions, mention it in the first message: "I updated `test_data/raw/...` since we last talked — it's now 1,200 rows." Claude has no idea otherwise.

**Push back when something feels wrong.** You did this today and it caught a real bug ("how the fuck were that many observations dropped?"). That instinct is correct — trust it. If a number looks wrong, make Claude explain it before moving on.

**Say "stop" when you're confused.** I'll keep moving forward unless told otherwise. A short "wait, why?" is always better than letting me run further down the wrong path.

**Short messages are fine.** You don't have to write paragraphs. "make a docs file" works. "explain why" works. Don't feel like you owe me a polished prompt.

**Ask for plain English when I'm being technical.** I'll default to jargon (git, branches, worktrees, MLE, GLMs, ...) because most users want it. If you want me to drop the jargon, just say so once and I'll adjust for the rest of the session.

---

## 6. Where Your Project Setup Differs From Convention

Everything you've built is sound — these are minor stylistic things, not bugs.

| Convention | What you have | Notes |
|---|---|---|
| One `CLAUDE.md` at project root | `claude.md_prompts/claude.md` | Move it up one level so it auto-loads. |
| Prompts kept in your head or in chat | `claude.md_prompts/prompt_NN_*.md` files | This is actually **better than convention** for a research project — it's a reproducible record of what each model script was asked to do. Keep doing this. |
| One project folder | One project folder + worktrees | Today we cleaned up the worktrees. Going forward, just one folder. |
| Git for snapshotting work | Git is here but you've barely touched it | This is fine. You don't need git for the research itself. If you ever want a "save point" before a risky change, ask me to "make a git commit" and I'll handle it. |

---

## 7. The Three Habits That Will Save You Time

1. **Start every session by saying what state things are in.** "Last session I did X. The Y file is updated. Now I want to do Z." Two sentences saves a lot of context-rebuilding.
2. **Verify data before modeling.** A quick `wc -l my_file.csv` or "show me the first 5 rows and the row count" can save hours.
3. **Don't accept output you don't understand.** If a number, file path, or explanation seems off, ask. Cheaper than letting it propagate.

---

## 8. When Things Feel Off

- **Claude is in the wrong folder?** Type `pwd` and ask Claude to confirm. If it's anywhere except `/Users/kaeleyoshea/Capstone/A.MRdeeP-Deep-Learning--MRP/`, close the window and restart from the right place.
- **Claude is doing something you didn't ask for?** Just say "stop" or close the window. No harm done — you can review every file change in your editor before doing anything irreversible.
- **You're confused about what's happening?** "Explain like I haven't used Claude Code before" is a perfectly valid request. I'll drop the jargon.

---

That's the whole thing. None of this is mandatory — you got this far without any of it. But these are the patterns that'll make future sessions smoother.
