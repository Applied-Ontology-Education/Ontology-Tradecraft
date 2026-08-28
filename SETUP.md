# Setup

This page describes the baseline technical environment for **Ontology Tradecraft**.

The course is remote and asynchronous. You should complete this setup before beginning Project 1. Later projects introduce additional libraries and tools; those project-specific requirements will be documented in the relevant project folder.

If you already have a working development environment, you do not need to reinstall everything. You do need to verify that the commands below work.

---

## 1. Create a GitHub Account

You will use GitHub throughout the course to:

- obtain course materials;
- maintain your own copy of the repository;
- create branches;
- submit work through pull requests;
- review the work of peers; and
- respond to feedback.

Create an account at:

https://github.com/


If you already have an account, use that account.

---

## 2. Install Git

Install Git from:

https://git-scm.com/downloads

Verify the installation from a terminal:

```bash
git --version
```

You should see a Git version number.

Configure your name and email if you have not done so previously:

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

Use an email address associated with your GitHub account if possible.

---

## 3. Install VS Code

Install Visual Studio Code:

https://code.visualstudio.com/

Recommended extensions:

- Python
- Jupyter
- GitHub Pull Requests and Issues
- RDF / Turtle syntax support, if desired

You may use another editor, but course examples assume VS Code.

---

## 4. Install Python

Install a current Python 3 release:

https://www.python.org/downloads/

Python 3.10 or 3.11 is a safe default for the baseline course environment unless a later project specifies a different version.

Verify:

```bash
python --version
```

On some systems, use:

```bash
python3 --version
```

You should also verify `pip`:

```bash
python -m pip --version
```

or:

```bash
python3 -m pip --version
```

---

## 5. Fork the Course Repository

Open:

https://github.com/Applied-Ontology-Education/Ontology-Tradecraft

Select **Fork** and create a fork under your own GitHub account.

Your fork is your working copy of the course repository. Do not complete assignments directly on the upstream repository's `main` branch.

---

## 6. Clone Your Fork

On your GitHub fork, select **Code**, copy the HTTPS URL, and run:

```bash
git clone https://github.com/YOUR-USERNAME/Ontology-Tradecraft.git
cd Ontology-Tradecraft
```

Replace `YOUR-USERNAME` with your GitHub username.

Verify your remotes:

```bash
git remote -v
```

You should see your fork listed as `origin`.

Add the course repository as `upstream`:

```bash
git remote add upstream https://github.com/Applied-Ontology-Education/Ontology-Tradecraft.git
```

Verify again:

```bash
git remote -v
```

You should now have:

- `origin` — your fork;
- `upstream` — the course repository.

---

## 7. Create a Python Virtual Environment

From the repository root:

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

When the environment is active, your terminal should normally display something like:

```text
(.venv)
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

---

## 8. Install Baseline Python Tools

Install the tools used in the early course modules:

```bash
python -m pip install rdflib jupyter
```

Verify RDFLib:

```bash
python -c "import rdflib; print(rdflib.__version__)"
```

Verify Jupyter:

```bash
jupyter --version
```

Later projects may require additional packages. Install those packages only when the relevant project instructions call for them.

---

## 9. Install Java

Several ontology tools and reasoners require Java.

Install a current Long-Term Support Java Development Kit (JDK). Java 17 is a reasonable default unless a project specifies otherwise.

Verify:

```bash
java -version
```

You should receive version information rather than an error.

---

## 10. Install ROBOT

ROBOT is used for ontology automation, reasoning, reporting, and related command-line tasks.

Installation documentation:

https://robot.obolibrary.org/

After installation, verify:

```bash
robot --version
```

If your shell reports that `robot` cannot be found, confirm that the ROBOT executable is on your system `PATH`.

---

## 11. Install Protégé

Download the Protégé Desktop ontology editor:

https://protege.stanford.edu/

Launch Protégé once to confirm that it opens correctly.

You will use Protégé for inspecting ontologies, editing OWL content, and working with automated reasoners.

---

## 12. Open the Repository in VS Code

From the repository root:

```bash
code .
```

If the `code` command is unavailable, open VS Code manually and select **File → Open Folder**, then select the `Ontology-Tradecraft` directory.

If VS Code asks you to select a Python interpreter, choose the interpreter inside `.venv`.

---

## 13. Test Jupyter

Start Jupyter:

```bash
jupyter lab
```

or:

```bash
jupyter notebook
```

Create or open a notebook and run:

```python
import rdflib

g = rdflib.Graph()
print("RDFLib is working.")
```

If the cell executes without an error, the baseline Python/Jupyter environment is working.

Stop Jupyter when finished.

---

## 14. Verify GitHub Authentication

From inside your cloned repository, run:

```bash
git status
```

Then create a temporary setup branch:

```bash
git switch -c setup-test
```

Create a harmless temporary file:

```bash
echo "setup test" > setup-test.txt
```

Commit it:

```bash
git add setup-test.txt
git commit -m "Test GitHub setup"
```

Push the branch:

```bash
git push -u origin setup-test
```

If the branch appears on your GitHub fork, your Git/GitHub workflow is functioning.

You may then delete the test branch and file:

```bash
git switch main
git branch -D setup-test
```

Delete the remote `setup-test` branch from GitHub as well.

---

## 15. Keep Your Fork Synchronized

Before beginning a new project, update your local `main` branch:

```bash
git switch main
git fetch upstream
git pull upstream main
git push origin main
```

Then create a **new branch for the project**.

For example:

```bash
git switch -c project-2
```

Do not complete multiple projects on one long-running branch.

---

## Setup Checklist

Before beginning Project 1, verify that all of the following are true:

- [ ] I have a GitHub account.
- [ ] `git --version` works.
- [ ] VS Code is installed.
- [ ] `python --version` or `python3 --version` works.
- [ ] I created and activated `.venv`.
- [ ] RDFLib imports successfully.
- [ ] Jupyter starts successfully.
- [ ] `java -version` works.
- [ ] `robot --version` works.
- [ ] Protégé launches successfully.
- [ ] I forked the course repository.
- [ ] I cloned my fork.
- [ ] `origin` points to my fork.
- [ ] `upstream` points to the course repository.
- [ ] I successfully pushed a test branch to my fork.

If every item is checked, proceed to **Project 1**.

---

## Getting Help With Setup

Technical problems are expected. Before asking for help:

1. Read the complete error message.
2. Confirm that you are running the command from the correct directory.
3. Confirm that your Python virtual environment is active.
4. Run the relevant verification command above.
5. Search existing repository Issues to see whether the problem has already been addressed.

If the problem remains, open a GitHub Issue and include:

- your operating system;
- the step you were attempting;
- the exact command you ran;
- the complete error message;
- what you expected to happen; and
- what you have already tried.

Do not post passwords, access tokens, private keys, or other credentials.
