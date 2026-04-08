# utils/detect/arguments.py
from __future__ import annotations      # allows forward references in type hints without quoting types
import argparse                         # used to build & parse the detect CLI interface
import sys                              # used to access raw CLI arguments & exit after custom help output
from ..help_text import print_detect_help  # custom detect help printer used instead of default argparse help

HELP_FLAGS = {"--help", "-h", "help"}   # tokens that should trigger the custom detect help screen
MODEL_FLAGS = {"--model", "-m"}         # flags that introduce single model values
VALUE_FLAGS = {"--model", "-m"}         # known flags that explicitly consume one following CLI value
BOOLEAN_FLAGS = {"--test"}              # flags that act as booleans & do not consume a following value

# --------- HELP & TOKEN HELPERS ---------
def _wants_custom_help(raw_args: list[str]) -> bool:
    """
    Return True if the user requested detect-specific help output.
    """
    return any(arg in HELP_FLAGS for arg in raw_args)  # custom help should trigger before argparse handles anything else

def _is_flag(token: str) -> bool:
    """
    Return True when a token looks like a CLI flag.
    """
    return token.startswith("--") or token in MODEL_FLAGS  # catches long flags plus the short model alias used in parsing helpers

def _split_key_value(token: str) -> tuple[str, str] | None:
    """
    Split a key=value token into (key, value).

    Returns None when the token is not in key=value form.
    """
    if "=" not in token:
        return None                                          # plain tokens are not treated as inline key=value arguments

    key, value = token.split("=", 1)                         # only split once so values can still contain '=' safely
    return key.strip().lower(), value.strip()                # normalize key casing & trim surrounding whitespace from both parts

# --------- MODEL NORMALIZATION ---------
def _normalize_models(raw_args: list[str]) -> tuple[list[str], list[str]]:
    """
    Extract model arguments from all supported forms and return:

        (args_without_model_tokens, models_list)

    Supported forms:
        --model X
        -m X
        --models A B C
        model=X
        m=X
        models=A,B,C
    """
    remaining_args: list[str] = []                           # collects all non-model tokens that should continue to argparse
    models: list[str] = []                                   # collects extracted model names from all supported input forms

    i = 0
    while i < len(raw_args):
        token = raw_args[i]

        if token in MODEL_FLAGS:
            next_index = i + 1
            if next_index < len(raw_args) and not _is_flag(raw_args[next_index]):
                models.append(raw_args[next_index].strip())  # consume the next token as the model value for --model/-m
                i += 2
                continue

            remaining_args.append(token)                     # malformed/incomplete model flags are left for argparse to surface properly
            i += 1
            continue

        if token == "--models":
            j = i + 1
            while j < len(raw_args):
                next_token = raw_args[j]

                if _is_flag(next_token):
                    break                                    # stop once the next real flag begins

                if "=" in next_token:
                    break                                    # avoid swallowing unrelated key=value tokens like sources=...

                models.append(next_token.strip())            # collect free tokens after --models as model names
                j += 1

            i = j
            continue

        split_token = _split_key_value(token)
        if split_token is not None:
            key, value = split_token

            if key in {"model", "m"}:
                if value:
                    models.append(value)                     # support single inline model assignments like model=sparrow-v2
                i += 1
                continue

            if key == "models":
                models.extend(part.strip() for part in value.split(",") if part.strip())  # support comma-separated inline model lists
                i += 1
                continue

        remaining_args.append(token)                         # any token not consumed as a model stays in the argparse input stream
        i += 1

    return remaining_args, models

# --------- SOURCE NORMALIZATION ---------
def _inject_sources_if_needed(args_list: list[str]) -> list[str]:
    """
    Allow free positional sources, such as:

        avian detect video.mp4 usb0

    and normalize them into:

        avian detect --sources video.mp4 usb0

    Rules:
        - if --sources is already present, leave args unchanged
        - if there are no flags at all, treat all args as sources
        - if flags exist, inject --sources before the first free positional token
    """
    if not args_list or "--sources" in args_list:
        return args_list                                      # leave already-normalized source input unchanged

    if not any(_is_flag(arg) for arg in args_list):
        return ["--sources", *args_list]                      # all-free tokens are treated as positional sources automatically

    normalized: list[str] = []
    i = 0

    while i < len(args_list):
        token = args_list[i]

        if token in VALUE_FLAGS:
            normalized.append(token)                          # preserve flags that explicitly consume one following value
            if i + 1 < len(args_list):
                normalized.append(args_list[i + 1])
                i += 2
            else:
                i += 1
            continue

        if token in BOOLEAN_FLAGS:
            normalized.append(token)                          # boolean flags pass through without consuming the next token
            i += 1
            continue

        if _is_flag(token):
            normalized.append(token)                          # unknown or already-valid flags are passed through unchanged
            i += 1
            continue

        normalized.append("--sources")                        # first free positional token marks the beginning of source arguments
        normalized.extend(args_list[i:])                      # everything from here onward becomes part of the --sources list
        break

    return normalized

# --------- LIST CLEANUP HELPERS ---------
def _dedupe_models(models: list[str]) -> list[str]:
    """
    De-duplicate model names while preserving order.
    """
    seen: set[str] = set()
    deduped: list[str] = []

    for model in models:
        model_name = str(model).strip()
        if not model_name or model_name in seen:
            continue                                          # skip blank model values & repeated entries while preserving first occurrence

        seen.add(model_name)
        deduped.append(model_name)

    return deduped

# --------- PARSER BUILD ---------
def _build_parser() -> argparse.ArgumentParser:
    """
    Build the detect CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run object detection on video inputs and/or camera sources.",
        add_help=False,                                       # custom help flow is handled manually so detect-specific help text can be shown
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Use your test model directory.",
    )

    parser.add_argument(
        "--sources",
        nargs="*",
        help="One or more camera/video sources.",
    )

    parser.add_argument(
        "--model",
        "-m",
        action="append",
        default=[],
        help=(
            "Select one or more models. Repeatable.\n"
            "Examples:\n"
            "  --model sparrow-v2 --model yolo11n\n"
            "  model=sparrow-v2 model=yolo11n\n"
            "  models=sparrow-v2,yolo11n"
        ),
    )                                                          # repeatable single-model flag for standard CLI use

    parser.add_argument(
        "--models",
        nargs="*",
        help="Alias for providing multiple models: --models A B C.",
    )                                                          # explicit multi-model alias for more natural batch selection

    return parser

# --------- PUBLIC ENTRY POINT ---------
def parse_arguments():
    """
    Parse and normalize detect CLI arguments.

    Supported conveniences:
        - custom detect help output
        - repeatable --model / -m
        - --models A B C
        - model=... / m=... / models=a,b,c
        - free positional sources without needing --sources
        - default source fallback to usb0
    """
    raw_args = sys.argv[1:]                                   # skip the executable/script name & work only with user-supplied arguments

    if _wants_custom_help(raw_args):
        print_detect_help()                                   # custom detect help is printed before argparse gets involved
        sys.exit(0)

    parser = _build_parser()

    args_without_model_tokens, extracted_models = _normalize_models(raw_args)  # peel out model args from all supported shorthand forms first
    final_argv = _inject_sources_if_needed(args_without_model_tokens)           # normalize free positional sources into explicit --sources usage
    args = parser.parse_args(final_argv)

    merged_models = [
        *(args.model or []),
        *(args.models or []),
        *extracted_models,
    ]                                                         # merge parser-collected models with pre-extracted shorthand/key=value model inputs
    args.model = _dedupe_models(merged_models)                # final model list is deduped while preserving original user order

    if not args.sources:
        args.sources = ["usb0"]                               # default to usb0 so detect still runs when the user provides no source explicitly

    return args