#!/usr/bin/env python3
"""Post an independent local review attestation for a PR's current head."""
import argparse
import json
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('pr')
parser.add_argument('--repo', default='mateu/graphrag-notes')
parser.add_argument('--reviewer', required=True, help='Independent reviewer identity or session ID')
parser.add_argument('--verdict', choices=['clean', 'findings'], required=True)
args = parser.parse_args()
head = subprocess.check_output(['gh', 'pr', 'view', args.pr, '-R', args.repo, '--json', 'headRefOid', '--jq', '.headRefOid'], text=True).strip()
body = '<!-- local-review:v1 ' + json.dumps(dict(commit_sha=head, verdict=args.verdict, reviewer=args.reviewer)) + ' -->'
subprocess.run(['gh', 'pr', 'comment', args.pr, '-R', args.repo, '--body', body], check=True)
