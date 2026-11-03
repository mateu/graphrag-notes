#!/usr/bin/env python3
"""Trusted, stdlib-only exact-head review policy and GitHub check publisher."""
import argparse
import datetime as dt
import json
import os
import re
import urllib.request

BOT = 'chatgpt-codex-connector[bot]'


def timestamp(value):
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00'))


def marker(body, name):
    match = re.search(r'<!-- ' + re.escape(name) + r' (.*?) -->', body, re.S)
    try:
        value = json.loads(match[1]) if match else None
        return value if isinstance(value, dict) else None
    except (ValueError, TypeError):
        return None


def evaluate(head, draft, since, comments, threads, now, minutes=20, attesters=('mateu',), resolved_receipts=None):
    """Return (pending|success|failure, description). Never trust body identity."""
    if draft:
        return 'pending', 'draft'
    unresolved = sum(not t['isResolved'] and any(
        c.get('originalCommit', {}).get('oid') == head or
        c.get('commit', {}).get('oid') == head
        for c in t['comments']['nodes']) for t in threads)
    if unresolved:
        return 'failure', f'{unresolved} unresolved review threads on head'
    verdicts = []
    for c in comments:
        if c['user']['login'] != BOT:
            continue
        body = c['body']
        state = marker(body, 'codex-review-state:v1')
        review = state.get('code_review') if state else None
        if isinstance(review, dict) and review.get('commit_sha') == head and review.get('status') == 'completed':
            if review.get('has_unresolved_findings') is True:
                verdicts.append((c.get('updated_at', ''), 'failure', 'Codex completed: unresolved findings on head'))
            elif review.get('has_unresolved_findings') is False:
                verdicts.append((c.get('updated_at', ''), 'success', 'Codex clean for exact head'))
        receipt = re.search(r'\*\*Reviewed commit:\*\*\s*`([0-9a-f]{7,40})`', body)
        if '<!-- codex-review-receipt:v1 -->' in body and "Codex Review: Didn't find any major issues" in body and receipt and (receipt[1] == head or (resolved_receipts or {}).get(receipt[1]) == head):
            verdicts.append((c.get('updated_at', ''), 'success', 'Codex clean receipt for exact head'))
    if verdicts:
        # Fail closed if any completed evidence still reports unresolved findings.
        failures = [v for v in verdicts if v[1] == 'failure']
        return max(failures or verdicts)[1:]
    opens = since + dt.timedelta(minutes=minutes)
    if now >= opens:
        attestations = []
        for c in comments:
            value = marker(c['body'], 'local-review:v1')
            if c['user']['login'] in attesters and value and value.get('commit_sha') == head and isinstance(value.get('reviewer'), str) and value['reviewer'].strip():
                attestations.append((c.get('updated_at', ''), value))
        if attestations:
            value = max(attestations, key=lambda x: x[0])[1]
            return ('success', 'Local-review fallback: clean for exact head') if value.get('verdict') == 'clean' else ('failure', 'Local-review fallback: non-clean verdict')
    return 'pending', f'Waiting for exact-head Codex verdict; fallback opens {opens.isoformat()}'


class GitHub:
    def __init__(self, repo, token):
        self.repo = repo
        self.token = token

    def api(self, path, data=None, method=None):
        request = urllib.request.Request('https://api.github.com/' + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={'Authorization': f'Bearer {self.token}', 'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'}, method=method)
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    def pages(self, path):
        result = []
        page = 1
        while True:
            batch = self.api(f'{path}{"&" if "?" in path else "?"}per_page=100&page={page}')
            result.extend(batch)
            if len(batch) < 100:
                return result
            page += 1

    def connection(self, number, field, selection):
        nodes, cursor = [], None
        while True:
            query = '''query($owner:String!,$name:String!,$number:Int!,$cursor:String){repository(owner:$owner,name:$name){pullRequest(number:$number){%s(first:100,after:$cursor){nodes{%s} pageInfo{hasNextPage endCursor}}}}}''' % (field, selection)
            owner, name = self.repo.split('/')
            result = self.api('graphql', {'query': query, 'variables': dict(owner=owner, name=name, number=number, cursor=cursor)})
            if result.get('errors'):
                raise RuntimeError(result['errors'])
            connection = result['data']['repository']['pullRequest'][field]
            nodes.extend(connection['nodes'])
            if not connection['pageInfo']['hasNextPage']:
                return nodes
            cursor = connection['pageInfo']['endCursor']

    def run(self, number):
        root = f'repos/{self.repo}'
        pr = self.api(f'{root}/pulls/{number}')
        if pr['state'] != 'open':
            return
        head = pr['head']['sha']
        checks = self.api(f'{root}/commits/{head}/check-runs?check_name=Review%20gate&filter=latest&per_page=100')['check_runs']
        existing = next((c for c in checks if c['name'] == 'Review gate' and c['app']['id'] == 15368), None)
        pending = {'name': 'Review gate', 'status': 'in_progress',
                   'output': {'title': 'Evaluating exact-head review evidence', 'summary': f'PR #{number}, head `{head}`'}}
        if existing:
            check = self.api(f'{root}/check-runs/{existing["id"]}', pending, 'PATCH')
        else:
            check = self.api(f'{root}/check-runs', dict(pending, head_sha=head))
        commit = self.api(f'{root}/commits/{head}')
        since = timestamp(commit['commit']['committer']['date'])
        for event in self.pages(f'{root}/issues/{number}/timeline'):
            if event.get('event') == 'ready_for_review':
                since = max(since, timestamp(event['created_at']))
        threads = self.connection(number, 'reviewThreads', 'isResolved comments(first:100){nodes{originalCommit{oid} commit{oid}} pageInfo{hasNextPage}}')
        # Refuse truncated thread evidence rather than silently ignore it.
        if any(t['comments']['pageInfo']['hasNextPage'] for t in threads):
            raise RuntimeError('Review thread exceeds 100 comments; maintainer inspection required')
        comments = self.pages(f'{root}/issues/{number}/comments')
        resolved_receipts = {}
        for comment in comments:
            if comment['user']['login'] != BOT:
                continue
            match = re.search(r'\*\*Reviewed commit:\*\*\s*`([0-9a-f]{7,39})`', comment['body'])
            if match and match[1] not in resolved_receipts:
                # GitHub rejects ambiguous abbreviations; API errors leave the gate pending.
                resolved_receipts[match[1]] = self.api(f'{root}/commits/{match[1]}')['sha']
        status, description = evaluate(head, pr['draft'], since,
            comments, threads,
            dt.datetime.now(dt.timezone.utc),
            int(os.environ.get('REVIEW_GATE_FALLBACK_MINUTES') or '20'),
            tuple(x.strip() for x in (os.environ.get('REVIEW_GATE_ATTESTERS') or self.repo.split('/')[0]).split(',')), resolved_receipts)
        payload = {'name': 'Review gate', 'head_sha': head, 'status': 'in_progress' if status == 'pending' else 'completed',
            'output': {'title': description[:255], 'summary': f'PR #{number}, head `{head}`. {description}'}}
        if status != 'pending':
            payload['conclusion'] = status
            payload['completed_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        payload.pop('head_sha')
        self.api(f'{root}/check-runs/{check["id"]}', payload, 'PATCH')
        print(number, head, status, description)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pr', type=int)
    args = parser.parse_args()
    gh = GitHub(os.environ['GITHUB_REPOSITORY'], os.environ['GITHUB_TOKEN'])
    event = json.load(open(os.environ['GITHUB_EVENT_PATH'], encoding='utf-8'))
    number = args.pr or event.get('pull_request', {}).get('number')
    if not number and event.get('issue', {}).get('pull_request'):
        number = event['issue']['number']
    if number:
        gh.run(number)
    elif os.environ['GITHUB_EVENT_NAME'] in ('schedule', 'workflow_dispatch'):
        for pr in gh.pages(f'repos/{gh.repo}/pulls?state=open'):
            gh.run(pr['number'])


if __name__ == '__main__':
    main()
