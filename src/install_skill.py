"""Install the knowledge-index-mcp Claude skill into the current directory."""

import os
import shutil
import sys


SKILL_DIR_NAME = "knowledge-search"
SKILL_FILENAME = "SKILL.md"


def main():
    target_dir = os.path.join(os.getcwd(), ".claude", "skills", SKILL_DIR_NAME)
    target_path = os.path.join(target_dir, SKILL_FILENAME)

    # Locate the bundled skill file inside the package
    pkg_skill = os.path.join(
        os.path.dirname(__file__), "skills", SKILL_DIR_NAME, SKILL_FILENAME
    )
    if not os.path.exists(pkg_skill):
        print(f"Error: skill file not found in package at {pkg_skill}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(target_path):
        print(f"Skill already exists at {target_path} — overwriting.")

    shutil.copy2(pkg_skill, target_path)
    print(f"Installed knowledge-search skill to {target_path}")


if __name__ == "__main__":
    main()
