# Submission and Peer Review

This page defines the standard submission workflow for **Projects 1–6** in Ontology Tradecraft.

The course is remote and asynchronous. GitHub is the primary environment for submitting technical work, receiving feedback, reviewing peer work, and documenting revisions.

Unless a project README explicitly states otherwise, use the workflow below.

---

## Course Project Structure

The course contains six major project milestones:

- **Projects 1–5** are Pull Request Projects.
- **Project 6** is the Final Project.

Each project builds on skills developed earlier in the course.

---

## Before Starting a Project

First, synchronize your fork.

From your local repository:

```bash
git switch main
git fetch upstream
git pull upstream main
git push origin main
```

Your `main` branch should remain a clean copy of the current course repository.

Do **not** complete assignment work directly on `main`.

---

## 1. Create a Project Branch

Create a new branch for the project.

For example:

```bash
git switch -c project-1
```

For later projects:

```bash
git switch -c project-2
git switch -c project-3
```

and so on.

Use one branch per project.

---

## 2. Read the Entire Project README

Before changing files:

1. Open the relevant project directory.
2. Read its `README.md` from beginning to end.
3. Identify the files provided as starter material.
4. Identify the files you are expected to create or modify.
5. Identify any commands or tests you are expected to run.
6. Identify the required deliverables.

Do not assume that every file in a project folder should be edited.

---

## 3. Complete the Work Locally

Work in your project branch.

Check your branch at any time with:

```bash
git branch --show-current
```

Check changed files with:

```bash
git status
```

Commit meaningful stages of work rather than making one enormous final commit.

Example:

```bash
git add projects/project-2/
git commit -m "Complete RDF enrichment for Project 2"
```

Use commit messages that describe what changed.

Good:

```text
Add SHACL constraints for required identifiers
```

Poor:

```text
stuff
```

---

## 4. Test Before Submitting

Run every test or validation command specified in the project README.

Depending on the project, this may include:

- Python scripts;
- Jupyter notebooks;
- RDF parsing;
- SPARQL queries;
- OWL reasoning;
- ROBOT commands;
- SHACL validation;
- automated tests; or
- GitHub Actions.

Do not knowingly submit a project that fails its documented tests without explaining the failure in the pull request.

Before submitting, also run:

```bash
git status
```

Make sure that:

- required files are present;
- accidental temporary files are not included;
- generated files are included only when required;
- credentials and secrets are not present; and
- your intended changes have been committed.

---

## 5. Push Your Branch

Push the branch to your fork:

```bash
git push -u origin project-1
```

Replace `project-1` with the name of your current branch.

After the first push, later updates can normally be pushed with:

```bash
git push
```

---

## 6. Open a Pull Request

On GitHub, open a pull request from:

```text
YOUR-USERNAME:project-X
```

into the course repository's:

```text
main
```

Use a clear title:

```text
Project 3 — Your Name
```

In the pull request description, include:

```markdown
## Project

Project X — Project Title

## Summary

Briefly describe what you completed.

## Deliverables

- [ ] Required deliverable 1
- [ ] Required deliverable 2
- [ ] Required deliverable 3

## Validation

Describe the tests, reasoners, queries, validation commands, or other checks you ran.

## Known Issues

List any known limitations or unresolved problems. Write "None known" if appropriate.

## Questions

List any questions for reviewers.
```

A pull request is part of the assignment. It should make your work understandable to someone who did not watch you create it.

---

## 7. Check GitHub Actions

If the repository runs automated checks, look at the **Checks** or **Actions** section of your pull request.

### If the checks pass

Proceed to peer review.

### If a check fails

1. Open the failed check.
2. Read the error output.
3. Reproduce the problem locally when possible.
4. Fix the problem on the same project branch.
5. Commit the fix.
6. Push again.

The pull request will update automatically.

Do not open a second pull request merely to fix the first one.

---

## 8. Complete Peer Review

Projects 1–5 include peer-review activity unless the project instructions say otherwise.

A substantive peer review should do more than say that the work looks good or bad.

A useful review might:

- identify a modeling ambiguity;
- question a class or relation choice;
- identify a missing competency question;
- point out an RDF, OWL, SPARQL, or SHACL issue;
- suggest a clearer definition;
- identify a failed or missing test;
- question whether ontology reuse is appropriate;
- identify an interoperability problem;
- suggest a simpler implementation; or
- explain why a particular design decision is strong.

Be charitable, specific, and technically useful.

Examples of insufficient review comments:

```text
Looks good.
```

```text
I don't think this is right.
```

Examples of useful review comments:

```text
The competency question asks which organizations participate in the process,
but the current model connects only persons to the process. Is the intended
restriction deliberate? If not, the domain of `has participant` may need to be
used more generally here.
```

```text
The SHACL shape requires exactly one identifier, but the ontology does not
appear to state that identifiers are unique. I would distinguish the validation
requirement from the OWL claim and document why the closed-world constraint is
being enforced in SHACL.
```

---

## 9. Respond to Feedback

Peer review is not complete merely because someone commented on your pull request.

You should:

1. read the feedback;
2. decide whether a revision is warranted;
3. make appropriate changes;
4. push those changes to the same branch; and
5. respond to the reviewer.

If you disagree with a suggestion, explain why. You are not required to accept every recommendation, but you are expected to engage seriously with substantive feedback.

Useful response:

```text
I kept the relation unchanged because the competency questions require
participation by both persons and organizations. I added documentation explaining
that choice and a test case showing both uses.
```

---

## 10. Final Submission State

A project is ready for final evaluation when:

- [ ] all required deliverables are present;
- [ ] required local tests have been run;
- [ ] applicable automated checks pass;
- [ ] the pull request description is complete;
- [ ] required peer review has been completed;
- [ ] you have responded to substantive feedback; and
- [ ] your final revisions have been pushed to the pull request.

Unless instructed otherwise, **do not close your pull request after submitting it**.

---

## Updating a Pull Request After Submission

You do not need to create a new pull request when making revisions.

Make the change locally:

```bash
git add .
git commit -m "Address peer review feedback"
git push
```

The existing pull request will update automatically.

This is intentional: the pull request should preserve the history of the project, review, and revision.

---

## After a Project Is Complete

Return to `main`:

```bash
git switch main
```

Update it:

```bash
git fetch upstream
git pull upstream main
git push origin main
```

Then create a new branch for the next project:

```bash
git switch -c project-X
```

Do not build the next project on top of an old project branch unless the project instructions explicitly require that workflow.

---

## Project 6: Final Project

Project 6 is the Final Project and integrates the major techniques developed during the course.

Follow the same branch, testing, push, and pull-request workflow described above.

The final project should make the complete workflow understandable and reproducible. Follow the Project 6 README for the exact required artifacts.

The final presentation is **recorded and submitted asynchronously**. No live presentation is required unless explicitly arranged separately.

Your final presentation should make clear:

- the problem being addressed;
- the competency questions or intended use;
- the modeling strategy;
- ontology reuse and alignment decisions;
- the data or RDF workflow;
- OWL axioms and reasoning;
- SPARQL querying;
- validation, including SHACL where applicable;
- automation or pipeline design;
- ML or LLM components where applicable;
- results;
- limitations; and
- what you would improve with additional development time.

---

## Common Git Problems

### I accidentally edited `main`

Do not panic.

Before committing, create a branch from your current state:

```bash
git switch -c project-X
```

Then continue normally.

### My branch is behind the course repository

Update `main` first:

```bash
git switch main
git fetch upstream
git pull upstream main
git push origin main
```

Then return to your project branch.

If you need the new upstream changes inside an existing project branch, merge or rebase carefully. If you are unsure how to resolve conflicts, ask before deleting or overwriting work.

### Git says I have a merge conflict

Do not choose changes blindly.

Open the conflicting files, identify the competing edits, and decide what the final file should contain. After resolving the conflict:

```bash
git add PATH-TO-RESOLVED-FILE
git commit
git push
```

### I committed a file that should not be in the repository

Remove it from Git tracking:

```bash
git rm --cached PATH-TO-FILE
```

Then add an appropriate rule to `.gitignore`, commit, and push.

### I accidentally committed a password or token

Removing it in a later commit is **not sufficient**, because the secret remains in Git history.

Immediately revoke or rotate the exposed credential. Then report the incident so the repository history can be handled appropriately.

---

## Asking for Technical Help

Before opening an Issue:

1. read the project README;
2. run the documented test or validation command;
3. read the complete error;
4. check existing Issues; and
5. confirm that the problem occurs on your current project branch.

When opening an Issue, include:

```markdown
## Project

Project X

## Environment

- Operating system:
- Python version:
- Java version:
- ROBOT version, if relevant:

## What I Was Trying to Do

Describe the task.

## Command

```text
paste the exact command here
```

## Error

```text
paste the complete error here
```

## What I Tried

Describe the troubleshooting steps already attempted.

## Link

Link to your branch or pull request if relevant.
```

Never include passwords, access tokens, API keys, private keys, or other credentials.

---

## The Short Version

For every project:

```text
sync main
   ↓
create project branch
   ↓
read project README
   ↓
complete work
   ↓
run tests
   ↓
commit
   ↓
push
   ↓
open pull request
   ↓
check automated tests
   ↓
complete peer review
   ↓
respond to feedback
   ↓
revise and push
   ↓
project complete
```

If you are unsure what to do next, return to this workflow.
