"""``mcda elicit tradeoff``: pairwise trade-off questions and the weights they imply.

Read-only on the project. Answers live in a ``tradeoff-answers`` sidecar (or the filled sheet)
outside it. ``fit`` is registered from ``cli.py``, the one module outside the calibration package
that may import it (noisy mode runs on the calibration sampler). Design:
``docs/design/tradeoff-elicitation.md``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape

from mcda_studio.domain import tradeoff_answers as ta
from mcda_studio.domain.errors import McdaError, ValidationFailed
from mcda_studio.methods import tradeoff as tm
from mcda_studio.storage.json_repository import JsonProjectRepository

elicit_app = typer.Typer(help="Elicit judgements from a person.", no_args_is_help=True)
tradeoff_app = typer.Typer(
    help="Pairwise trade-off questions and the weights a person's answers imply. Read-only; "
    "answers stay outside the project. See docs/design/tradeoff-elicitation.md.",
    no_args_is_help=True,
)
elicit_app.add_typer(tradeoff_app, name="tradeoff")

_err = Console(stderr=True)
_PRIORS = ("auto", "ranking", "unknown", "stated", "low", "medium", "high")

ProjectArg = Annotated[Path, typer.Argument(help="Project JSON file.", exists=True)]
ProfileOpt = Annotated[str | None, typer.Option(help="Weight profile id (default: the analysis').")]
PriorOpt = Annotated[
    str,
    typer.Option(
        help="auto (the profile's elicitation record: its conviction, else its order), ranking, "
        "unknown, stated (the project's SMAA-2 weight groups), low, medium or high."
    ),
]
SeedOpt = Annotated[int, typer.Option(help="Seed for the PCG64 stream.")]
ParticlesOpt = Annotated[int, typer.Option(help="Particles in exact mode.")]


def _fail(message: str) -> NoReturn:
    _err.print(f"[red]error:[/red] {escape(message)}")
    raise typer.Exit(1)


def _call[T](fn: Callable[[], T]) -> T:
    try:
        return fn()
    except (McdaError, ValidationFailed) as exc:
        _fail(str(exc))


def _context(file: Path, profile: str | None, prior: str):  # type: ignore[no-untyped-def]
    from mcda_studio.services import tradeoff_service as ts

    if prior not in _PRIORS:
        _fail("--prior must be one of " + ", ".join(_PRIORS))
    try:
        project = JsonProjectRepository().load(file)
    except Exception as exc:  # noqa: BLE001 - the repository's own messages are shown as they are
        _fail(str(exc))
    return _call(lambda: ts.context(project, profile, prior))  # type: ignore[arg-type]


def _answers(ctx, path: Path, answered_by: str | None, answered_on: str | None):  # type: ignore[no-untyped-def]
    """A tradeoff-answers JSON file, or a filled sheet (``.csv``)."""
    from mcda_studio.services import tradeoff_service as ts

    try:
        raw = path.read_bytes()
    except OSError as exc:
        _fail(f"Cannot read the answers: {exc}")
    if path.suffix.lower() == ".csv":
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            _fail(f"the sheet is not valid UTF-8 ({exc.reason}); save it as 'CSV UTF-8'")
        answers, blank = _call(
            lambda: ts.answers_from_sheet(
                ctx, text, answered_by=answered_by or "unknown", answered_on=answered_on
            )
        )
        if blank:
            _err.print(f"{blank} row(s) have no choice and count as skipped")
        return answers
    try:
        return ta.TradeoffAnswers.model_validate_json(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValidationError) as exc:
        _fail(f"Cannot load the answers: {exc}")


def _settings(particles: int, seed: int) -> tm.SmcSettings:
    settings = tm.SmcSettings(particles=particles, seed=seed)
    try:
        settings.check()
    except ValueError as exc:
        _fail(str(exc))
    return settings


@tradeoff_app.command("next")
def tradeoff_next(
    file: ProjectArg,
    answers: Annotated[
        Path | None, typer.Option("--answers", help="tradeoff-answers JSON or a filled sheet.")
    ] = None,
    profile: ProfileOpt = None,
    prior: PriorOpt = "auto",
    real_pairs: Annotated[
        bool, typer.Option("--real-pairs", help="Also consider pairs of the project's ventures.")
    ] = False,
    random_question: Annotated[
        bool, typer.Option("--random", help="Pick at random among open questions (a control).")
    ] = False,
    particles: ParticlesOpt = 4000,
    seed: SeedOpt = 0,
) -> None:
    """Print the next question as JSON, or why to stop. Add the question to the answers file
    with a "choice" of "A", "B" or "skip"."""
    from mcda_studio.services import tradeoff_service as ts

    ctx = _context(file, profile, prior)
    given = _answers(ctx, answers, None, None) if answers is not None else None
    out = _call(
        lambda: ts.next_question(
            ctx,
            given,
            _settings(particles, seed),
            real_pairs=real_pairs,
            mode="random" if random_question else "adaptive",
        )
    )
    typer.echo(json.dumps(out.model_dump(mode="json"), indent=2))


@tradeoff_app.command("sheet")
def tradeoff_sheet(
    file: ProjectArg,
    out_dir: Annotated[Path, typer.Option("-o", "--out-dir", help="Write tradeoff.csv and .html.")],
    profile: ProfileOpt = None,
    prior: PriorOpt = "auto",
    questions: Annotated[int, typer.Option(help="Number of questions.")] = 20,
    stem: Annotated[str, typer.Option(help="File name stem (default tradeoff).")] = "tradeoff",
    particles: ParticlesOpt = 4000,
    seed: SeedOpt = 0,
) -> None:
    """Write a fixed sheet of questions (UTF-8 CSV with a byte-order mark, every choice blank)
    and a page that explains them. Questions are chosen adaptively against a simulated
    respondent who answers each one A or B with probability 1/2."""
    from mcda_studio.services import tradeoff_service as ts

    if not stem or any(c in stem for c in '/\\:*?"<>|') or stem in (".", ".."):
        _fail("--stem must be a plain file name without a folder or extension")
    if not 1 <= questions <= tm.MAX_QUESTIONS:
        _fail(f"--questions must be between 1 and {tm.MAX_QUESTIONS}")
    sheet, page = out_dir / f"{stem}.csv", out_dir / f"{stem}.html"
    for target in (sheet, page):
        if target.exists():
            _fail(f"Refusing to overwrite {target}")
    ctx = _context(file, profile, prior)
    settings = _settings(particles, seed)
    chosen = _call(lambda: ts.sheet_questions(ctx, settings, questions))
    out_dir.mkdir(parents=True, exist_ok=True)
    with sheet.open("x", encoding="utf-8-sig", newline="") as handle:
        handle.write(ts.sheet_csv(ctx, chosen))
    with page.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(ts.sheet_html(ctx, chosen, settings))
    _err.print(f"wrote {sheet} ({len(chosen)} questions) and {page}")
