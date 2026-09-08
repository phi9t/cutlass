<!-- Agentic Engineering Workflow v2026.09.02.1. Canonical source: ultron/docs/agents/agentic-engineering.md. Sibling copies are managed by Ultron. -->
# Agentic Engineering Workflow

> Scale agent execution, not agent authority.

Use this workflow for planning, building, fixing, or changing code. The human
owns architectural authority and merge. Agents receive bounded work, isolated
execution, and an independent verification budget.

When Matt and Superpowers both apply, read
`docs/agents/skill-orchestration.md`. Load every applicable skill, then execute
one combined phase under the owner named there. Do not repeat equivalent
ceremonies or create parallel artifacts merely because both suites describe
them. `using-superpowers` discovers applicable skills; `ask-matt` selects the
route.

Some harnesses expose Matt skills only as explicit human commands. If a named
owner is not model-invocable, do not skip its phase and do not claim to have
invoked it. Apply the route and combined completion criteria in the crosswalk
directly, using every available supporting skill, and record that fallback in
the durable ticket.

## 1. Establish intent and choose the path

Inspect the working tree and applicable repository guidance, then use
`ask-matt` to route the work. Before repository exploration, read
`docs/agents/domain.md`; use its glossary and surface any ADR conflict. Read
`docs/agents/issue-tracker.md` before tracker operations and
`docs/agents/triage-labels.md` before triaging incoming work. Put durable intent
in the configured tracker:

- Single-repository work uses that repository's tracker.
- Cross-repository work starts in Ultron at `.scratch/<effort>/`. Create linked
  child tickets in every affected repository and point each child back to the
  Ultron map.

Every managed repository carries an Ultron-scoped mainline pointer,
`ultron/mainline`, so cross-repo policy and landing work has one stable branch
name to target regardless of whether a repository's native default is `main` or
`master`. `tools/sync_agentic_workflow.py --check` reports `MISSING-BRANCH`
drift when the branch is absent, and `--apply` creates it at the repository's
`base_branch` tip. The convention is ensure-exists only: an existing
`ultron/mainline` is never moved, force-updated, or deleted.

The **short path** is available only when every condition holds: one repository,
one session, reversible, narrowly scoped, and no public API or schema,
migration, security boundary, destructive data path, distributed or concurrency
invariant, production infrastructure, or architectural decision.

If any condition fails, use the **gated path**. Record the classification and
its evidence in the durable ticket before implementation begins.

## 2. Short path

Keep the primary context. Define the behavior, implement red-green slices, run
focused tests, perform the standard two-axis code review, and inspect the final
diff. Completion requires fresh verification evidence and human-owned merge.

## 3. Bug path

Before broad investigation, Matt's diagnosis loop must establish a fast command
that can turn red and minimize the failure. Then use Superpowers root-cause and
pattern analysis: rank three to five hypotheses, test one variable at a time,
and encode the verified cause as a failing regression test before fixing it.
After three failed fixes, stop patching and escalate to architectural review.
The crosswalk defines the full hard-bug recipe.

## 4. Gated path

Complete these gates in order:

1. **Intent:** Record the outcome, acceptance criteria, constraints, and explicit
   authority boundary in the tracker.
2. **Design:** Run Matt's stateful repository design flow with human
   participation. Resolve decisions in frontier rounds rather than forcing a
   one-question cadence. Completion means the repository has been explored,
   alternatives and tradeoffs are explicit, the human has approved the design,
   ambiguities have been self-reviewed, and every material decision is settled
   or represented by a decision ticket. This satisfies the Superpowers
   brainstorming phase.
3. **Independent critique:** Send the design to a separate context that did not
   inherit the design or implementation conversation. Prefer another model
   family when available. Reconcile every finding before advancing.
4. **Specification:** Write the precise spec in the tracker, subject it to
   adversarial review, and split it into blocker-linked tracer-bullet tickets.
   Each ticket declares requirements, exclusions, blockers, and verification
   evidence. The tracker spec and tickets are canonical; exact file-and-code
   micro-plans remain session-local. Do not duplicate them under
   `docs/superpowers/specs` or `docs/superpowers/plans`.
5. **Isolated execution:** Implement each ticket in its own worktree. Parallel
   agents receive non-overlapping tickets and never share a mutable checkout.
6. **Behavioral proof:** Approve the public test seam once during design or
   specification, then drive every behavior slice with visible red-green
   evidence. Only behavior-preserving cleanup follows green. Run focused tests
   first, then the repository's broader required checks.
7. **Ticket conformance:** Review every ticket on both Matt axes: repository
   Standards and ticket Spec. This satisfies the Superpowers request-review
   phase only when neither axis has unresolved blocking findings.
8. **Independent verification:** Run `roborev review --branch`. If branch review
   is unavailable, use a separate-context reviewer against the fixed base and
   spec. Passing tests do not replace this gate.
9. **Repair:** Fix findings and re-review for at most three iterations. Run
   `roborev refine --max-iterations 3` only on a clean isolated branch or
   worktree; otherwise repair through the implementing agent and submit a fresh
   review. Escalate non-convergence, uncertainty, or exhausted budget as
   attention-needed rather than silently extending authority or spend.
10. **Human control:** Present the reviewed branch, remaining risks, fresh
    completion evidence, and intent coverage. Commit, push, and pull-request
    creation each require explicit authorization. Only a human performs the
    merge; agents stop at the reviewed branch even when otherwise authorized.
11. **Distillation:** Update the living glossary, ADRs, and architecture docs
    with knowledge that remains useful after the task. Code and tests remain the
    durable truth alongside tracker specs and tickets; session-local
    micro-plans, execution checklists, and review traces are non-authoritative
    context.

## Operating boundaries

- Durable intent is external to prompts, conversations, agent memory, and
  commits.
- Background and parallel execution stays isolated from the human's checkout.
- Verification has its own budget and may cost as much as implementation when
  risk warrants it, but extending the three-iteration repair budget requires a
  human decision.
- Agents surface decisions, findings, patches, and evidence. Humans retain
  architecture authority and perform every merge.
- Skill text never grants Git authority. Commit, push, and pull-request actions
  require explicit authorization for the specific action.
- Preserve existing work and repository-specific rules. Direct user
  instructions and more specific repository guidance take precedence.
