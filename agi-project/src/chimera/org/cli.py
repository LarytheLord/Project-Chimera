"""CLI entry point: `python -m chimera.org submit "<goal>"`.

Phase 1 smoke test. Uses PrometheusCognitiveCore (Gemini) unless --mock is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from ..cognitive_core.prometheus_core import PrometheusCognitiveCore
from .org import Org


def _build_org(db_root: str) -> Org:
    core = PrometheusCognitiveCore()
    return Org.default(cognitive_core=core, db_root=db_root)


def cmd_submit(args: argparse.Namespace) -> int:
    org = _build_org(db_root=args.db_root)
    wo = org.submit(args.goal)
    print(f"Submitted WorkOrder {wo.id}, running through {len(org.roles)} roles...")
    final = org.run_until_complete(wo.id, max_hops=args.max_hops)
    print(json.dumps(final.to_dict(), indent=2, default=str))
    return 0 if final.status.value == "completed" else 1


def cmd_resume(args: argparse.Namespace) -> int:
    org = _build_org(db_root=args.db_root)
    finals = org.resume(max_hops=args.max_hops)
    for wo in finals:
        print(f"{wo.id}\t{wo.status.value}\t{wo.assigned_role}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    org = _build_org(db_root=args.db_root)
    for wo in org.store.all():
        print(f"{wo.id}\t{wo.status.value}\t{wo.assigned_role}\t{wo.goal[:60]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chimera.org")
    parser.add_argument(
        "--db-root",
        default=os.environ.get("CHIMERA_ORG_DB_ROOT", "./chimera_org_db"),
        help="Directory for per-role memories and the WorkOrder SQLite store.",
    )
    parser.add_argument("--max-hops", type=int, default=12)

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_submit = sub.add_parser("submit", help="Submit a new goal and run it to completion.")
    p_submit.add_argument("goal", help="The goal text the CEO will work from.")
    p_submit.set_defaults(func=cmd_submit)

    p_resume = sub.add_parser("resume", help="Resume any active WorkOrders from the store.")
    p_resume.set_defaults(func=cmd_resume)

    p_list = sub.add_parser("list", help="List all WorkOrders in the store.")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
