"""ReflexionMemory: ordered list of textual reflections accumulated across
trials of the same seed. Capped at `max_size` (paper default = 3)."""

from dataclasses import dataclass, field


@dataclass
class ReflexionMemory:
    reflections: list = field(default_factory=list)
    max_size: int = 3

    def add(self, reflection: str) -> None:
        if not reflection:
            return
        self.reflections.append(reflection.strip())
        if len(self.reflections) > self.max_size:
            self.reflections = self.reflections[-self.max_size:]

    def is_empty(self) -> bool:
        return len(self.reflections) == 0

    def format_as_block(self) -> str:
        """Return the reflection block to prepend onto the system prompt for
        the next trial. Empty string when no reflections are present so trial 1
        sees an unmodified prompt (clean baseline)."""
        if not self.reflections:
            return ""
        lines = ["", "## Past Lessons (from prior failed attempts on this seed)"]
        for i, r in enumerate(self.reflections, 1):
            lines.append(f"{i}. {r}")
        lines.append("Use these lessons to choose a different approach this trial. Do not repeat the same failed strategy.")
        return "\n".join(lines) + "\n"
