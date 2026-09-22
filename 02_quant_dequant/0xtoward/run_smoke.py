"""Launch each local server, send real HTTP requests, retain answers and stop it."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import requests

TASKS = [
    ("arith", "17 + 25 = ? Reply with only the integer.", "42"),
    (
        "extract",
        "Extract the access word after KEY: from this text: red KEY: lantern blue. Reply with only the access word.",
        "lantern",
    ),
    (
        "choice",
        "Which planet is known as the Red Planet? A. Venus B. Mars C. Jupiter D. Earth. Reply with only the letter.",
        "B",
    ),
    (
        "reverse",
        "Reverse the string abcde. Reply with only the reversed string.",
        "edcba",
    ),
    (
        "chinese",
        "请只回答数字：一个盒子里有12个球，拿走5个，又放入3个，剩下几个？",
        "10",
    ),
    (
        "sort",
        "Sort these integers in ascending order: 9, 2, 5. Reply with only the comma-separated list.",
        "2,5,9",
    ),
    ("fact", "What is the chemical symbol for gold? Reply with only the symbol.", "Au"),
    (
        "word",
        'The sentence is: "The cat sleeps on the mat." What is its third word? Reply with only that word.',
        "sleeps",
    ),
    ("explain", "用中文两句话解释为什么低精度权重能节省显存。", None),
]


def norm(s):
    return re.sub(r"[\s`*，。.!！]+", "", s).lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--modes", nargs="+", default=["bf16", "mxfp8", "nvfp4", "nvfp4_4over6"]
    )
    p.add_argument("--output", default="results/smoke_small.json")
    p.add_argument("--port", type=int, default=8127)
    args = p.parse_args()
    root = Path(__file__).parent
    os.chdir(root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    all_results = []
    session = requests.Session()
    session.trust_env = False
    for mode in args.modes:
        log = output.with_name(output.stem + "_" + mode + ".log")
        with log.open("w") as stream:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(root / "smoke_server.py"),
                    "--model",
                    args.model,
                    "--mode",
                    mode,
                    "--port",
                    str(args.port),
                ],
                stdout=stream,
                stderr=stream,
            )
            try:
                deadline = time.time() + 300
                health = None
                while time.time() < deadline:
                    if proc.poll() is not None:
                        raise RuntimeError(
                            f"server exited {proc.returncode}; see {log}"
                        )
                    try:
                        h = session.get(
                            f"http://127.0.0.1:{args.port}/health", timeout=1
                        )
                        if h.ok:
                            health = h.json()
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(0.5)
                if health is None:
                    raise TimeoutError("server startup timed out")
                samples = []
                for name, prompt, expected in TASKS:
                    r = session.post(
                        f"http://127.0.0.1:{args.port}/v1/chat/completions",
                        json=dict(
                            messages=[dict(role="user", content=prompt)],
                            temperature=0,
                            max_tokens=96 if expected is None else 32,
                        ),
                        timeout=180,
                    )
                    r.raise_for_status()
                    body = r.json()
                    answer = body["choices"][0]["message"]["content"]
                    row = dict(
                        task=name,
                        prompt=prompt,
                        expected=expected,
                        answer=answer,
                        exact_match=norm(answer) == norm(expected)
                        if expected is not None
                        else None,
                        **body,
                    )
                    samples.append(row)
                    print(mode, name, repr(answer), row["exact_match"], flush=True)
                result = dict(
                    info=health,
                    samples=samples,
                    exact_matches=sum(s["exact_match"] is True for s in samples),
                    scored_tasks=8,
                )
                all_results.append(result)
                output.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
