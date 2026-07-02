---
name: review-code-by-other
description: This skill should be triggered when the agent is given explicit instruction to review commit(s) made by an individual other than the current user on the current branch.
---

For this skill, the agent should act as a reviewer to review code commited by another contributor, assess the code's correctness and provide feedback to the human user. The review should be conducted following the below steps:
1. Identify Git commits made by the reviewed individual after the last commit made by the user.
2. The user should have prompted earlier what functionalities the commits should achieve; if not prompted yet, ask the user to prompt.
3. Spawn agent A to read throught the commits and find out if the commits can achieve the user-specified functionalities. Also, check if test suites have been included explicitly to validate correctness, or if instructions have been provided to perform (manual) tests.
4. The main agent takes in what agent A has learned from the code-reading, and produces a brief summary for the human user, with the following information: 1) if the commited code covers the functionalities specified by the user; 2) if not fully covered, what functionalities are left uncovered; 3) if test suites have been included explicitly to validate correctness, or if instructions have been provided to perform (manual) tests; 4) if not, devise a strategy to test the commits; 5) potential issues in the code; 6) overall quality of the commits - do you believe this's quality work contributed by a team member with good grades, some level of knowledge of physics-based simulation, and with help of an AI-based coding agent (e.g. Gemini, Codex)?

No changes or corrections need to be performed. Code doesn't have to be compiled and run.