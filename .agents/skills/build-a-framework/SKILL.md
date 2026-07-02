---
name: build-a-framework
description: Design, implement, integrate, and validate a coherent framework composed of multiple interdependent features, using architect, feature-leader, and integration-tester subagents.
---

# build-a-framework

Build a framework: a coherent collection of features that work organically together to achieve a set of high-level objectives.

This skill extends the `build-a-feature` workflow. A framework is not merely a bundle of independent features. It requires:

- a shared design intent
- bounded high-level objectives
- scoped and specific requirements
- clear feature boundaries
- explicit coupling/interface definitions between features
- feature-level implementation and validation
- framework-level integration testing

Use this skill when the requested work involves multiple interacting features, subsystems, workflows, or architectural components.

---

## Agent roles and model requirements

### Main agent

The main agent owns the entire framework-building process.

Responsibilities:

- spawn and coordinate architect agents
- decide when the architecture is mature enough
- spawn feature leader agents
- spawn the integration tester agent
- maintain the framework-level source of truth
- decide whether the framework satisfies the user’s intent
- guard against scope creep, gold plating, or premature optimization
- produce the final report

The main agent must not collapse framework work into a single feature implementation burst.

---

### Architect A

Architect A is the primary system designer.

Model requirement:

- use the latest full model
- use **x-high reasoning**
- fall back to **high reasoning** only if x-high reasoning is unavailable

Responsibilities:

- meticulously analyze the user’s intent
- explore available contextual information, including code, papers, docs, previous plans, and `.agents/knowledge/`
- define at most **three high-level objectives**
- translate those objectives into at most **five scoped, specific requirements**
- define one feature for each requirement when appropriate
- define each feature’s boundary
- identify where features interleave, depend on each other, or couple through shared interfaces
- revise the design after critique from Architect B when useful

Architect A is not obligated to accept every critique from Architect B, but must explicitly explain whether each critique is accepted, partially accepted, or rejected.

---

### Architect B

Architect B is the architecture critic.

Model requirement:

- use the latest full model
- use **x-high reasoning**
- fall back to **high reasoning** only if x-high reasoning is unavailable

Responsibilities:

- independently explore available contextual information, including code, papers, docs, previous plans, and `.agents/knowledge/`
- critique Architect A’s design for:
  - feasibility
  - modularity
  - consistency with the user’s intent
  - overengineering
  - under-scoping
  - missing requirements
  - ambiguous interfaces
  - risky feature coupling
  - validation blind spots
- propose concrete changes

Architect B should not design an entirely separate framework unless Architect A’s proposal is fundamentally misaligned. Prefer targeted critique and actionable revisions.

After the main agent is satisfied with the architecture, Architect B is closed.

---

### Feature leader agents

Each feature leader owns one framework feature.

Model requirement:

- use the latest full model
- use **high reasoning**

Responsibilities:

- ingest the framework architecture from Architect A
- own one specific feature and its boundary
- invoke the `build-a-feature` skill to execute feature-level:
  - exploration
  - planning
  - implementation
  - validation
- coordinate with other feature leaders on shared interfaces, coupling points, or integration contracts
- document feature status, validation evidence, unresolved risks, and interface assumptions

Each feature leader may spawn sub-subagents through the `build-a-feature` skill.

After a feature leader completes its feature, it must close its sub-subagents.

The main agent must **not** close feature leader agents after their feature is complete, because they remain responsible for post-integration fixes.

---

### Integration tester agent

The integration tester owns framework-level validation.

Model requirement:

- use the latest full model
- use **high reasoning**

Responsibilities:

- ingest Architect A’s final framework design
- develop an incremental framework integration test plan
- define how completed features should collectively satisfy the high-level objectives
- test the completed set of features after feature leaders finish their initial implementations
- identify integration failures
- map each failure to the responsible feature leader or leaders
- request targeted fixes from feature leaders
- repeat the test → identify → fix → test loop until the framework satisfies the requirements and high-level objectives, or until further progress requires human intervention

The integration tester should focus on emergent behavior across features, not just isolated feature correctness.

---

## Framework process overview

The default sequence is:

1. **Architecture design**
2. **Architecture critique and revision**
3. **Feature leader spawning**
4. **Feature-level build-a-feature execution**
5. **Integration test planning**
6. **Framework-level integration testing**
7. **Failure ownership assignment**
8. **Feature-level fixes**
9. **Repeated integration testing**
10. **Final framework report**

Do not skip architecture.
Do not jump directly into feature implementation.
Do not declare success after individual features pass in isolation.

---

# Phase 1: Architecture design

The main agent first spawns two architect subagents:

- Architect A
- Architect B

Architect A begins by analyzing the user’s intent and available contextual information.

## Architect A objectives

Architect A must produce:

1. **User intent analysis**
   - What is the user actually trying to achieve?
   - What outcomes matter most?
   - What constraints are explicit?
   - What constraints are implied by the codebase, papers, project context, or prior work?

2. **Context exploration**
   - relevant code modules
   - relevant scripts, configs, tests, docs, and examples
   - relevant papers or research notes
   - existing abstractions and extension points
   - existing implementation patterns
   - constraints, invariants, and likely pitfalls

3. **High-level objectives**
   - define at most **three**
   - each objective must be framework-level, not feature-level
   - each objective must be testable or at least evaluable

4. **Scoped requirements**
   - define at most **five**
   - each requirement must be specific enough to map to implementation work
   - each requirement must support one or more high-level objectives

5. **Feature decomposition**
   - define one feature per requirement when appropriate
   - avoid unnecessary feature proliferation
   - each feature must have a clear owner
   - each feature must have clear boundaries
   - each feature must identify inputs, outputs, side effects, and dependencies

6. **Feature coupling map**
   - identify where features interleave
   - identify shared interfaces
   - identify shared data structures
   - identify shared configuration
   - identify sequencing dependencies
   - identify risks of inconsistent assumptions between features

---

## Architect A deliverable

Architect A must write the architecture proposal to:

`.agents/frameworks/FRAMEWORK_<framework_name>.md`

This document is the framework-level source of truth.

It must include:

```md
# Framework: <framework_name>

## User intent

## Available context reviewed

## High-level objectives

1. ...
2. ...
3. ...

## Requirements

1. ...
2. ...
3. ...
4. ...
5. ...

## Feature decomposition

### Feature 1: <name>
- Requirement addressed:
- Boundary:
- Inputs:
- Outputs:
- Side effects:
- Dependencies:
- Non-goals:
- Invokes build-a-feature: yes/no

### Feature 2: <name>
...

## Feature coupling and interfaces

## Integration risks

## Validation strategy

## Assumptions

## Open questions
````

Architect A must not implement code during this phase.

---

# Phase 2: Architecture critique and revision

After Architect A produces the initial framework design, Architect B critiques it.

## Architect B critique objectives

Architect B must evaluate:

* Does the architecture satisfy the user’s actual intent?
* Are the high-level objectives too many, too vague, or too narrow?
* Are the requirements scoped and specific?
* Are features properly bounded?
* Are any features too large and should be split?
* Are any features too small and should be merged?
* Are feature interfaces explicit enough?
* Are there hidden coupling points?
* Is implementation feasible in the current codebase?
* Are validation criteria strong enough?
* Are there missing tests, datasets, scripts, or execution paths?
* Are there likely failure modes that Architect A missed?

---

## Architect B deliverable

Architect B must produce a critique containing:

```md
# Architecture Critique

## Summary judgment

## Alignment with user intent

## Feasibility issues

## Modularity issues

## Requirement issues

## Feature-boundary issues

## Coupling/interface risks

## Validation blind spots

## Proposed changes

1. ...
2. ...
3. ...

## Must-fix before implementation

## Nice-to-have improvements
```

The critique should be concise but concrete.

---

## A-propose / B-criticize loop

Architect A must review Architect B’s critique and revise the framework design.

For each proposed change, Architect A must state:

* accepted
* partially accepted
* rejected

Architect A must explain the reason.

The main agent repeats the A-propose / B-criticize loop until the main agent is satisfied that:

* the architecture reflects the user’s intent
* the objectives are bounded
* the requirements are scoped
* the feature decomposition is coherent
* feature boundaries and coupling points are explicit
* the validation strategy is plausible

When satisfied, the main agent closes Architect B.

Architect A’s final revised document remains the authoritative framework design.

---

# Phase 3: Spawn feature leaders

The main agent spawns one feature leader per approved feature.

Each feature leader receives:

* Architect A’s final framework design
* the specific feature assigned to them
* the feature boundary
* the requirement the feature satisfies
* known dependencies on other features
* known interface/coupling obligations
* expected validation evidence

Feature leaders must not independently redefine the framework architecture.

They may clarify local implementation details, but framework-level changes must be escalated to the main agent.

---

## Feature leader operating rules

Each feature leader must invoke the `build-a-feature` skill for its assigned feature.

The feature leader must preserve the `build-a-feature` phase structure:

1. Explore
2. Plan
3. Implement
4. Validate

Feature leaders must write their feature-level plans to:

`.agents/plans/PLAN_<feature_name>.md`

Feature leaders must write their feature completion reports to:

`.agents/reports/FEATURE_<feature_name>_REPORT.md`

---

## Feature leader deliverable

Each feature leader’s report must include:

```md
# Feature Report: <feature_name>

## Requirement addressed

## Framework objective supported

## Summary of implementation

## Files changed

## Interfaces exposed

## Interfaces consumed

## Coordination with other feature leaders

## Feature-level validation performed

## What passed

## What failed and was fixed

## Remaining limitations

## Integration concerns for tester
```

After completing the feature, each feature leader closes its sub-subagents.

The main agent keeps the feature leader alive for later integration fixes.

---

# Phase 4: Cross-feature coordination

If two or more features interface with each other, the relevant feature leaders must coordinate before implementation reaches the point of irreversible divergence.

Coordination is required when features share:

* APIs
* schemas
* config fields
* file formats
* CLI arguments
* database tables
* simulator state
* rendering outputs
* logging conventions
* test fixtures
* execution order
* common utility modules

---

## Coordination deliverable

When coordination is needed, the involved feature leaders must write an interface agreement to:

`.agents/interfaces/INTERFACE_<feature_a>__<feature_b>.md`

For rare three-feature interfaces, use:

`.agents/interfaces/INTERFACE_<feature_a>__<feature_b>__<feature_c>.md`

The interface agreement must include:

```md
# Interface Agreement: <feature names>

## Features involved

## Shared objective

## Shared requirement or coupling point

## Producer responsibilities

## Consumer responsibilities

## Data/API/schema contract

## Error handling

## Compatibility constraints

## Test expectations

## Known risks
```

Interface agreements are binding unless changed by the main agent.

---

# Phase 5: Integration test planning

While feature leaders are executing feature-level planning, implementation, and validation, the main agent spawns the integration tester agent.

The integration tester ingests Architect A’s final framework design and develops a framework-level integration test plan.

The integration tester does not wait for all features to complete before planning.

---

## Integration test plan objectives

The integration test plan must define:

* the order in which completed features should be tested together
* the smallest meaningful integration slice
* the expected behavior of each integration slice
* how each requirement will be checked
* how each high-level objective will be evaluated
* what scripts, tests, scenes, commands, or manual checks should be used
* what artifacts should be inspected
* how failures should be attributed to feature owners
* what counts as framework-level success

---

## Integration test plan deliverable

The integration tester must write the plan to:

`.agents/plans/INTEGRATION_PLAN_<framework_name>.md`

The plan must include:

```md
# Integration Test Plan: <framework_name>

## Framework objectives

## Requirements under test

## Feature dependency graph

## Incremental test sequence

### Test slice 1
- Features involved:
- Command / procedure:
- Expected result:
- Failure signals:
- Likely owner if failed:

### Test slice 2
...

## Full-framework test

## Artifacts to inspect

## Regression checks

## Success criteria

## Stop conditions
```

---

# Phase 6: Framework-level integration testing

After all feature leaders complete their initial feature work, the integration tester begins testing.

The integration tester must run the planned integration tests incrementally.

Do not only run a final full-framework test.

Start from the smallest meaningful integration slice and gradually add features.

---

## Integration testing objectives

The integration tester must verify:

* features can operate together
* shared interfaces are respected
* outputs from one feature are valid inputs to another
* config and execution flow are coherent
* the framework satisfies each scoped requirement
* the framework collectively serves the high-level objectives
* no obvious regressions were introduced

---

## Integration testing loop

For each integration failure:

1. Identify the failure.
2. Explain the expected behavior.
3. Explain the actual behavior.
4. Determine the most likely owning feature leader or leaders.
5. Send a targeted fix request to the responsible feature leader.
6. The responsible feature leader invokes `build-a-feature` again for the fix.
7. The feature leader validates the fix locally.
8. The integration tester reruns the relevant integration test.
9. Repeat until the test passes or the failure requires human intervention.

The integration tester should not directly implement feature fixes unless explicitly authorized by the main agent.

---

## Failure ownership

A failure may be owned by:

* one feature leader
* two feature leaders if the failure lies at an interface
* rarely, three feature leaders if the failure involves a shared contract among three features
* the main agent if the failure reveals a framework-level design flaw

If a framework-level design flaw is discovered, the main agent must reopen architectural reasoning with Architect A.

Architect B may be respawned if critical architectural critique is needed.

---

# Phase 7: Framework acceptance

The framework is complete only when the integration tester believes:

* all scoped requirements are satisfied
* the high-level objectives are satisfied or credibly supported
* feature interfaces work together
* integration tests pass
* remaining limitations are documented
* any untested assumptions are explicit

A framework must not be declared complete merely because each feature passed its own validation.

---

## Stop conditions

The integration loop should stop and ask for human intervention if:

* more than five integration fix attempts fail on the same issue
* the expected behavior is ambiguous and cannot be resolved from available context
* required data, credentials, hardware, simulator assets, or external services are unavailable
* the implementation requires a major architecture change not covered by the approved framework design
* the framework’s objectives appear infeasible under current constraints

When stopping, the integration tester must clearly summarize:

* what failed
* what was expected
* what was tried
* which feature leaders were involved
* why further autonomous progress is unsafe or inefficient
* what decision or resource is needed from the human user

---

# Phase 8: Final framework report

The main agent produces the final framework report after integration testing is complete.

Write the report to:

`.agents/reports/FRAMEWORK_<framework_name>_FINAL_REPORT.md`

---

## Final report structure

```md
# Framework Final Report: <framework_name>

## User intent addressed

## High-level objectives

1. ...
2. ...
3. ...

## Requirements

1. ...
2. ...
3. ...
4. ...
5. ...

## Implemented features

### Feature 1: <name>
- Owner:
- Requirement:
- Summary:
- Validation status:

### Feature 2: <name>
...

## Feature coupling and interfaces

## Integration tests performed

## What passed

## What failed and was fixed

## Remaining limitations

## Known risks

## Evidence of success

## Recommended next steps
```

---

# Operating rules

## 1. Respect framework-level phase boundaries

The default sequence is:

Architecture design → Critique/revision → Feature leader execution → Integration planning → Integration testing → Fix loop → Final report

Do not skip directly to implementation.

---

## 2. Keep objectives and requirements bounded

Architect A must define:

* at most three high-level objectives
* at most five scoped requirements

If the user asks for more, group related goals into coherent objectives and explain the grouping.

---

## 3. Prefer explicit artifacts

For nontrivial framework work, create and use explicit artifacts:

* `.agents/frameworks/FRAMEWORK_<framework_name>.md`
* `.agents/plans/PLAN_<feature_name>.md`
* `.agents/interfaces/INTERFACE_<feature_a>__<feature_b>.md`
* `.agents/plans/INTEGRATION_PLAN_<framework_name>.md`
* `.agents/reports/FEATURE_<feature_name>_REPORT.md`
* `.agents/reports/FRAMEWORK_<framework_name>_FINAL_REPORT.md`

These artifacts are the source of truth for coordination.

---

## 4. Preserve feature boundaries

Feature leaders must not silently expand their feature scope.

If implementation requires changing another feature’s boundary or shared interface, escalate to the main agent.

---

## 5. Validate locally and globally

Each feature must pass feature-level validation through `build-a-feature`.

The framework must also pass integration testing.

Both are mandatory.

---

## 6. Be honest about uncertainty

If requirements are ambiguous, state the interpretation clearly.

If assumptions are made, record them in the relevant framework, feature, or integration artifact.

---

## 7. Avoid unnecessary refactors

Do not perform broad refactors, renames, or cleanup unless they are necessary for the framework’s correctness, maintainability, or integration.

---

## 8. Keep changes scoped and reversible

Prefer incremental, reversible changes.

Do not introduce large architectural shifts unless Architect A’s approved design explicitly requires them.

---

## 9. Escalate design flaws

If implementation or integration reveals that the architecture is flawed, the main agent must reopen architectural reasoning with Architect A.

Respawn Architect B when critique is needed.

---

## 10. Do not confuse feature success with framework success

A framework succeeds only when the features collectively satisfy the high-level objectives.

Individual feature completion is necessary but not sufficient.

---

# Default output structure

When using this skill, structure the final response roughly as:

## Architecture summary

* User intent
* High-level objectives
* Requirements
* Feature decomposition
* Coupling/interface map

## Feature implementation summary

* Feature leaders
* Features completed
* Key implementation changes
* Feature-level validation

## Integration validation summary

* Integration tests run
* Failures found
* Fixes requested
* Fixes completed
* Remaining issues

## Final status

* Whether the framework satisfies the requirements
* Whether the framework satisfies the high-level objectives
* Evidence
* Limitations
* Next recommended actions

```
```
