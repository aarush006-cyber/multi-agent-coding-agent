from langchain_core.tools import tool
from pathlib import Path
import shutil
import subprocess

WORKSPACE = Path(".").resolve()

def safe_path(path: str) -> Path:
    """Resolve a path and ensure it stays inside the workspace."""
    target = Path(path).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise ValueError(f"Path is outside workspace: {path}")
    return target


@tool
def read_file(path: str) -> str:
    """Read the contents of a text file."""
    try:
        target = safe_path(path)
    except ValueError as e:
        return f"Error: {e}"
        
    if not target.exists():
        return f"Error: File does not exist at {path}"
        
    with open(target, "r", encoding="utf-8") as file:
        return file.read()


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a text file."""
    try:
        target = safe_path(path)
    except ValueError as e:
        return f"Error: {e}"

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as file:
        file.write(content)

    return f"Successfully wrote to {target}"


@tool
def list_files(path: str = ".") -> str:
    """List files and folders inside a directory."""
    try:
        folder = safe_path(path)
    except ValueError as e:
        return f"Error: {e}"

    if not folder.exists():
        return f"Path does not exist: {path}"
    if not folder.is_dir():
        return f"Not a directory: {path}"

    items = []
    for item in folder.iterdir():
        if item.is_dir():
            items.append(f"[DIR]  {item.name}")
        else:
            items.append(f"[FILE] {item.name}")

    return "\n".join(items)


@tool
def run_command(command: str) -> str:
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,cwd=str(WORKSPACE)
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += "\n" + result.stderr
            MAX_OUTPUT = 10000

            if len(output) > MAX_OUTPUT:
                output = output[:MAX_OUTPUT] + "\n[Output truncated]"

        output += f"\nExit code: {result.returncode}"
        return output

    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error running command: {e}"


@tool
def delete_file(path: str) -> str:
    """Delete a file."""
    try:
        file = safe_path(path)
    except ValueError as e:
        return f"Error: {e}"

    if not file.exists():
        return f"File does not exist: {path}"
    if not file.is_file():
        return f"Not a file: {path}"

    file.unlink()
    return f"Deleted: {path}"


@tool   
def search_files(query: str, path: str = ".") -> str:
    """Search for text inside files."""
    try:
        folder = safe_path(path)
    except ValueError as e:
        return f"Error: {e}"

    results = []
    for file in folder.rglob("*"):
        if file.is_file():
            try:
                text = file.read_text(encoding="utf-8")
                for line_number, line in enumerate(text.splitlines(), 1):
                    if query.lower() in line.lower():
                        results.append(f"{file}:{line_number}: {line.strip()}")
            except (UnicodeDecodeError, PermissionError):
                continue

    if not results:
        return f"No matches found for: {query}"

    return "\n".join(results)


@tool
def create_directory(path: str) -> str:
    """Create a directory."""
    try:
        directory = safe_path(path)
    except ValueError as e:
        return f"Error: {e}"

    if directory.exists():
        return f"Directory already exists: {path}"

    directory.mkdir(parents=True)
    return f"Created directory: {path}"


@tool
def move_file(source: str, destination: str) -> str:
    """Move a file or directory."""
    try:
        src = safe_path(source)
        dest = safe_path(destination)
    except ValueError as e:
        return f"Error: {e}"

    if not src.exists():
        return f"Source does not exist: {source}"
    if dest.exists():
        return f"Destination already exists: {destination}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))

    return f"Moved {source} → {destination}"
