<!-- Skill Orchestration Crosswalk v2026.08.14.1. Canonical source: ultron/docs/agents/skill-orchestration.md. Sibling copies are managed by Ultron. -->
# Matt–Superpowers Skill Orchestration

This policy applies whenever skills from both suites cover the same work. Load
every applicable skill, but execute each engineering phase once using the union
of its completion criteria. Precedence is:

1. Direct user instruction.
2. Repository policy, including more-specific nested guidance.
3. The phase owner named below.
4. Supporting skills.

A supporting skill adds checks or mechanics; it does not create a duplicate
phase, artifact, approval, or Git authority.

## Invocation availability

Skill installation does not guarantee model invocation: a harness may expose a
Matt owner only as an explicit human command or omit it from the model's skill
catalog. When the owner is invocable, use it. When it is not, do not skip the
phase or pretend the skill ran. This crosswalk and the Agentic Engineering
Workflow become the executable adapter: apply the named route and combined
completion criteria directly with all available supporting skills, and record
the fallback in durable intent. Pause only when the missing owner leaves a
material decision that the written criteria cannot resolve safely.

## Phase ownership and equivalence

| Phase | Owner | Supporting skills | Combined completion criteria |
| --- | --- | --- | --- |
| Skill discovery | `using-superpowers` | `ask-matt` | Applicable skills are loaded before action; discovery does not choose the engineering route. |
| Routing and durable intent | `ask-matt` | tracker and domain guidance | The route, durable store, scope, and authority boundary are explicit. |
| Domain language and design decisions | Matt domain and design skills | `codebase-design` and repository architecture guidance | Terms are defined in the living glossary; material decisions are approved and distilled into the appropriate ADR or architecture document. |
| Design | `grill-with-docs` and Matt design skills | `brainstorming` | The repository was explored; frontier decisions, alternatives, and tradeoffs are explicit; the human approved the design; ambiguities were self-reviewed. |
| Foggy-problem mapping | `wayfinder` | research, prototype, and grilling skills | Unknowns and decision dependencies are mapped into bounded learning work before implementation. |
| Specification and decomposition | `to-spec`, then `to-tickets` | `writing-plans` | An adversarially reviewed tracker spec and blocker-linked tracer-bullet tickets define requirements, exclusions, and evidence. |
| Isolation and execution | `using-git-worktrees`, then `subagent-driven-development` or `executing-plans` | Matt phase-boundary routing | Gated tickets run in isolated checkouts; concurrent work is non-overlapping; progress is tied to tickets. |
| Behavioral proof | `test-driven-development` | Matt-approved test seams and repository test guidance | Each behavior slice visibly fails before implementation and passes afterward; only behavior-preserving cleanup follows green. |
| Bug diagnosis | `diagnosing-bugs` | `systematic-debugging` | A tight red-capable command exists, the failure is minimized, and the verified root cause—not a symptom—is addressed. |
| Ticket review | Matt `code-review` | `requesting-code-review` | Both Standards and Spec axes have been reviewed with no unresolved blocking findings. |
| Review response | `receiving-code-review` | Matt ticket review | Findings are verified before changes; valid Critical and Important findings are resolved, and disagreements use technical evidence. |
| Branch defect gate | `roborev` | separate-context review fallback | The assembled branch has independent defect review distinct from ticket conformance. |
| Completion verification | `verification-before-completion` | repository checks | Claims use fresh command output from the final state; focused and broader required checks pass. |
| Git finish | Direct human authorization | `finishing-a-development-branch` | Only explicitly authorized commit, push, or pull-request actions run; a human performs every merge. |

## Normative conflict rules

- `using-superpowers` discovers and loads skills. `ask-matt` selects the route,
  durable location, and context boundary.
- Matt's design flow owns design. Its frontier rounds replace Superpowers'
  one-question cadence. It satisfies `brainstorming` only after repository
  exploration, alternatives and tradeoffs, explicit human approval, and an
  ambiguity self-review.
- `wayfinder`, `to-spec`, and `to-tickets` replace committed Superpowers design
  and plan documents. Tracker specs and tickets are canonical. Do not create
  duplicate `docs/superpowers/specs` or `docs/superpowers/plans` artifacts when
  the combined criteria are already satisfied.
- Exact files, code steps, commands, and execution checklists form a
  session-local micro-plan. They may implement a durable ticket but do not
  become a competing source of intent.
- Superpowers worktrees and subagent or plan-execution mechanics implement Matt
  tickets. They do not redefine the design, spec, ticket graph, or authority.
- Approve a public test seam once during design or specification. Every behavior
  slice must then show red before implementation and green afterward. Refactor
  only after green, and keep cleanup behavior-preserving. Architectural changes
  return to design rather than hiding inside refactoring.
- For bugs, Matt's tight loop comes first: obtain a fast red-capable command and
  minimize the failure. Then apply Superpowers root-cause and pattern analysis,
  rank three to five hypotheses, and test one variable at a time. After three
  failed fixes, stop patching and escalate to architectural review.
- Matt's two-axis Standards and Spec review satisfies Superpowers requesting
  review for each ticket. Receiving-review discipline still applies when
  evaluating findings. Branch-level `roborev` remains a separate defect gate,
  followed by fresh completion verification.
- Skill text never grants Git authority. Commit, push, and pull-request creation
  each require explicit authorization. Agents never perform merges.

## Artifact ownership

| Lifetime | Artifacts | Rule |
| --- | --- | --- |
| Durable | Tracker intent, reviewed specs, blocker-linked tickets, `CONTEXT.md`, ADRs, architecture docs, code, and tests | Maintain these as the authoritative record and distill lasting knowledge into them. |
| Ephemeral | Design conversation, exact file-and-code micro-plans, execution checklists, reviewer scratch, and raw tool output | Keep these session-local; discard or summarize them after their durable result is captured. |

Do not promote execution traces into architecture by default, and do not leave
lasting decisions only in ephemeral material.

## Route recipes

### Short feature

1. Discover skills, then let `ask-matt` confirm the short path and durable
   intent location.
2. Explore and design to the combined approval criteria; approve the test seam.
3. Implement visible red-green slices in the current context unless isolation
   is otherwise required.
4. Run two-axis review, inspect the diff, and collect fresh completion evidence.
5. Stop before any Git action that lacks explicit authorization and before
   merge in all cases.

### Gated feature

1. Record durable intent and acceptance criteria, complete Matt design in
   frontier rounds, and reconcile an independent separate-context critique.
2. Produce the tracker spec and blocker-linked tracer-bullet tickets.
3. Give each ticket a fresh isolated worktree and non-overlapping execution
   context. Implement using visible red-green slices.
4. Complete Standards and Spec review per ticket, then run branch-level
   `roborev` over the assembled result.
5. Permit at most three repair and re-review rounds, collect fresh completion
   evidence, distill lasting knowledge, and hand the reviewed branch to a human.

### Hard bug

1. Route through Matt diagnosis and establish the smallest fast command that
   can turn red; minimize and localize the failure.
2. Investigate root cause and analogous patterns. Rank three to five hypotheses
   and test one variable at a time.
3. After three failed fixes, stop and request architectural review.
4. Encode the verified cause as a failing regression test, make it green, then
   perform behavior-preserving cleanup and risk-scaled review.
5. Re-run the tight command and broader required checks from the final state.

### Wayfinder effort

1. Use `wayfinder` to map unknowns, decisions, and dependencies without treating
   the map as an implementation plan.
2. Create bounded research, prototype, or grilling tickets for unresolved
   questions.
3. When the frontier is clear, pass through `to-spec` and `to-tickets` before
   implementation unless `ask-matt` reclassifies the remaining work as short.

### Incoming triage

1. Put unshaped external requests through the repository triage rules.
2. Promote ready work into the appropriate short, gated, bug, or wayfinder
   route.
3. Matt-generated tickets that already satisfy tracker requirements bypass raw
   intake triage; they do not bypass design, evidence, review, or authority
   gates.

## Context boundaries

At every phase boundary, `ask-matt` chooses whether to continue, compact, hand
off, or start a fresh isolated context. Prefer fresh contexts for gated tickets
and independent review. Context boundaries reduce contamination; they never
erase durable intent or loosen acceptance criteria.
