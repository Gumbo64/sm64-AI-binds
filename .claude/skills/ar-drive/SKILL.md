---
name: ar-drive
description: Drive an SM64 agent-route session (objective-rrt-v2/agent, the `ar` CLI) interactively. Keep the objective window filled, react to the solver's events, and talk with the user while it runs. Use when the user asks to drive, run, continue or watch an `ar` session, or gives a route goal such as "get BoB's first star".
---

# Driving an `ar` session

You are the route planner for an SM64 run. The `ar` daemon's rolling-horizon GA solves the objectives you queue. The user is watching this window and may type changes or questions at any time. Answer them, then keep driving.

Arguments: `$ARGUMENTS` holds a session name and/or a goal (e.g. `smoke3`, or `bob1 get the first star from the castle grounds`).

## Set-up

1. `cd /workspaces/sm64-combined/sm64-AI-binds/objective-rrt-v2` and use `AR=target-agent/release/ar`. If the binary is missing, build it: `CARGO_TARGET_DIR=target-agent nice -n 10 cargo build --release -p obrrt-agent --bin ar`.
2. Read `agent/AGENT.md` (how to drive) and skim `agent/README.md` (commands, objective selectors).
3. Pick the session:
   - an existing one: `$AR use NAME`;
   - a new one: `$AR new NAME --threads 6`. Check the load first (`uptime`): other searches may be running, so don't take every core.
4. `$AR brief`. If no daemon is running, `$AR serve`.

## The loop

- Keep 2-3 objectives queued. Prefer native ones (`$AR objectives`). Write lambda, spec or goto objectives on the fly when no native one fits.
- Wait for events with the Bash tool and `run_in_background: true`: `$AR wait --timeout 1800`. While it runs you are free, and the user's messages reach you. When it returns, read the event(s) and act as AGENT.md says (`met` → push more, `stalled` → inspect and fix, and so on). Then start the next wait.
- Comment briefly in this window: what happened, what you decided and why (one to three lines per event). Also record the decision in the session with `$AR note "..."` and `--note` on push, scrap and rewind. The user can read that history later (`$AR events`, `$AR log`).
- When the user asks for a change (a different route, a budget, a rewind), say what you'll do, then do it with `ar` commands. When they ask a question, answer it from `$AR brief` / `state` / `objects` / `geo` / `events`, not from memory.
- Don't stop the daemon unless the user asks. If you need to end your turn, leave a background `ar wait` running so the next event wakes you.

## Rules

- No TAS inputs in a route meant to count. The `boot: tas` start is menus only.
- Cap courses: press the switch on the visit.
- Never delete session files or library templates (`ar lib retire`, not deletion).
- The compute floor (population 128, 600-frame plans, 1 frame per step, pink noise) is a minimum. Raising budgets is fine; lowering them below it is refused.
