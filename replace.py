from pathlib import Path


class Replacer:
    fields = {
        "project_name": "Project name",
        "project_description": "Project description",
        "year": "Year",
        "author_name": "Author name",
        "author_email": "Author email",
        "author_github": "Author GitHub",
    }

    def __init__(self):
        self.fields = {k: input(f"{v}: ") for k, v in self.fields.items()}

    def as_placeholder(self, key: str) -> str:
        return f"{{{{ {key} }}}}"

    def make_project_dir(self):
        name = self.fields["project_name"].replace("-", "_")
        Path("src", name).mkdir(parents=True, exist_ok=True)

    def replace_text(self, text: str) -> str:
        for k, v in self.fields.items():
            text = text.replace(self.as_placeholder(k), v)
        return text

    def replace_file(self, file: Path):
        text = file.read_text()
        text = self.replace_text(text)
        file.write_text(text)


if __name__ == "__main__":
    replacer = Replacer()
    replacer.make_project_dir()
    for file in [
        "pyproject.toml",
        "LICENSE",
        ".github/workflows/ci.yml",
        ".github/workflows/test-pre-releases.yml",
        "environment.yml",
    ]:
        replacer.replace_file(Path(file))
    Path(__file__).unlink()
