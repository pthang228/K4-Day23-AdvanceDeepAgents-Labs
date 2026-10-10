"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    problems = []
    if not isinstance(sources, list):
        return ["sources.json must contain a JSON list"]
    if not sources:
        return ["no sources in sources.json"]

    by_n = {}
    seen_urls = {}
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            problems.append(f"source at index {index} is not an object")
            continue
        number = source.get("n")
        url = source.get("url")
        if type(number) is not int:  # bool is an int subclass but not a valid citation number
            problems.append(f"source at index {index} has non-integer n={number!r}")
        elif number in by_n:
            problems.append(f"source number [{number}] is duplicated")
        else:
            by_n[number] = source
        if not isinstance(url, str) or not re.match(r"^https?://", url):
            problems.append(f"source n={number!r} has invalid url={url!r}")
        elif url in seen_urls:
            problems.append(f"duplicate source url for n={seen_urls[url]!r} and n={number!r}: {url}")
        else:
            seen_urls[url] = number

    heading = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
    matches = list(heading.finditer(str(report_text)))
    if matches:
        split = matches[-1]
        body = report_text[:split.start()]
        references = report_text[split.end():]
    else:
        problems.append("missing ## References heading")
        body = str(report_text)
        references = ""

    cited = _citation_numbers(body)
    source_numbers = set(by_n)
    for number in sorted(cited - source_numbers):
        problems.append(f"[{number}] cited but missing from sources.json")
    for number in sorted(source_numbers - cited):
        problems.append(f"source [{number}] never cited")

    reference_lines = {}
    for line in references.splitlines():
        match = re.match(r"^\[(\d+)\](?:\s+|$)(.*)$", line)
        if not match:
            continue
        number = int(match.group(1))
        reference_lines.setdefault(number, []).append(line)

    for number in sorted(source_numbers):
        lines = reference_lines.get(number, [])
        if not lines:
            problems.append(f"source [{number}] has no reference line")
        elif len(lines) > 1:
            problems.append(f"source [{number}] has {len(lines)} reference lines (expected exactly one)")
    for number in sorted(set(reference_lines) - source_numbers):
        problems.append(f"reference line [{number}] has no source in sources.json")

    for number, lines in sorted(reference_lines.items()):
        source = by_n.get(number)
        for line in lines:
            urls = re.findall(r"https?://[^\s<>]+", line)
            # A plain reference may end with punctuation; finalizer output puts a
            # space before the parenthesised date, so valid URLs are unchanged.
            urls = [url.rstrip(".,;") for url in urls]
            if len(urls) != 1:
                problems.append(f"reference line [{number}] contains {len(urls)} URLs (expected exactly one)")
            elif source is not None and urls[0] != source.get("url"):
                problems.append(
                    f"reference line [{number}] URL {urls[0]!r} does not match sources.json {source.get('url')!r}"
                )
    return problems


_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_CITATION = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")


def _citation_numbers(text):
    """Find singular/group/range citations outside code and Markdown links."""
    numbers = set()
    for index, segment in enumerate(_CODE.split(text)):
        if index % 2:  # fenced or inline code
            continue
        for match in _CITATION.finditer(segment):
            for part in re.split(r"\s*,\s*", match.group(1)):
                span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
                if not span:
                    numbers.add(int(part))
                    continue
                start, end = map(int, span.groups())
                if 0 <= end - start <= 200:
                    numbers.update(range(start, end + 1))
                else:
                    numbers.update((start, end))
    return numbers


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
