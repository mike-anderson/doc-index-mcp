"""Install the doc-index-mcp skill for a supported agent.

Claude Code reads skills from ``.claude/skills/<name>/SKILL.md`` relative to
the project. Hermes Agent reads them from
``~/.hermes/skills/<category>/<name>/SKILL.md``, keyed by category rather than
by project. Both consume the same SKILL.md, so this just puts the bundled file
where the chosen agent will look for it.
"""

import argparse
import os
import shutil
import sys


SKILL_DIR_NAME = "doc-search"
SKILL_FILENAME = "SKILL.md"
HERMES_CATEGORY = "research"


def _bundled_skill_path():
    """Absolute path to the SKILL.md shipped inside the package."""
    return os.path.join(
        os.path.dirname(__file__), "skills", SKILL_DIR_NAME, SKILL_FILENAME
    )


def _claude_target_dir():
    return os.path.join(os.getcwd(), ".claude", "skills", SKILL_DIR_NAME)


def _hermes_target_dir(category):
    # Hermes resolves its skill tree from HERMES_HOME when set, so honour it
    # rather than assuming ~/.hermes — installs into a non-default profile
    # would otherwise land somewhere the agent never reads.
    hermes_home = os.environ.get("HERMES_HOME") or os.path.join(
        os.path.expanduser("~"), ".hermes"
    )
    return os.path.join(hermes_home, "skills", category, SKILL_DIR_NAME)


def main():
    parser = argparse.ArgumentParser(
        description="Install the doc-search skill for Claude Code or Hermes Agent."
    )
    parser.add_argument(
        "--target",
        choices=["claude", "hermes"],
        default="claude",
        help="Which agent to install for (default: claude)",
    )
    parser.add_argument(
        "--category",
        default=HERMES_CATEGORY,
        help=(
            "Hermes skill category directory "
            f"(default: {HERMES_CATEGORY}; ignored for --target claude)"
        ),
    )
    args = parser.parse_args()

    pkg_skill = _bundled_skill_path()
    if not os.path.exists(pkg_skill):
        print(f"Error: skill file not found in package at {pkg_skill}", file=sys.stderr)
        sys.exit(1)

    if args.target == "hermes":
        target_dir = _hermes_target_dir(args.category)
    else:
        target_dir = _claude_target_dir()

    target_path = os.path.join(target_dir, SKILL_FILENAME)

    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(target_path):
        print(f"Skill already exists at {target_path} — overwriting.")

    shutil.copy2(pkg_skill, target_path)
    print(f"Installed doc-search skill to {target_path}")

    if args.target == "hermes":
        print("Restart your Hermes session so the skill is picked up.")


if __name__ == "__main__":
    main()
