import importlib.metadata
import json
import sys
sys.dont_write_bytecode = True
from envelope import checked_result

from hermes_cli import __version__
from tools.mcp_tool_discovery import register_mcp_servers
from tools.mcp_tool_lifecycle import shutdown_mcp_servers
from tools.registry import registry

request = json.load(sys.stdin)

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

try:
    names = register_mcp_servers({'graphrag': {'url': request['url'],
        'headers': {'Authorization': 'Bearer ' + request['token']}, 'timeout': 300,
        'connect_timeout': 30, 'skip_preflight': True, 'supports_parallel_tool_calls': True}})
    selected = {}
    required = sorted({call['tool'] for call in request['calls']}) if request.get('phase') == 'calls' else ('search_notes', 'get_record', 'build_context', 'capture_note')
    for tool in required:
        matches = [name for name in names if name.endswith('_' + tool)]
        require(len(matches) == 1, 'Hermes native tool catalog mismatch: ' + tool)
        selected[tool] = matches[0]
    def call(tool, arguments):
        return checked_result(registry.dispatch(selected[tool], arguments))
    if request.get('phase') == 'calls':
        results = [checked_result(registry.dispatch(selected[call['tool']], call['arguments']), call.get('expect_error'))
                   for call in request['calls']]
        print('HARNESS_RESULT=' + json.dumps({'version': __version__, 'python_version': sys.version.split()[0],
            'mcp_sdk_version': importlib.metadata.version('mcp'),
            'catalog': sorted(name.removeprefix('mcp__graphrag__') for name in names),
            'exercised_tools': sorted(selected), 'results': results}))
        sys.exit(0)
    found = call('search_notes', {'query': 'nativeclientatlas', 'mode': 'keyword', 'scope': 'all',
                                 'graph': 'off', 'limit': 10})
    require(len(found['records']) == 2, 'Hermes did not read both OpenClaw captures')
    record = request['original']
    inspected = call('get_record', {'id': record['id'], 'revision': record['revision'], 'neighbors': 0})
    require(inspected['content'] == request['capture']['content'] and
            inspected['provenance']['instance_id'] == 'openclaw-a', 'Hermes inspection lost A provenance')
    saved = call('capture_note', request['capture'])
    replay = call('capture_note', request['capture'])
    require(saved['replayed'] is False and replay['replayed'] is True and saved['record'] == replay['record'],
            'Hermes retry did not preserve original capture')
    require(saved['record']['provenance']['instance_id'] == 'hermes' and saved['record']['id'] != record['id'],
            'Hermes trusted principal collided with OpenClaw')
    found = call('search_notes', {'query': 'nativeclientatlas', 'mode': 'keyword', 'scope': 'all',
                                 'graph': 'off', 'limit': 10})
    require(len(found['records']) == 3, 'Hermes capture replay duplicated corpus records')
    context = call('build_context', {'query': 'nativeclientatlas', 'scope': 'all', 'graph': 'off',
                                    'max_chunks': 3, 'max_total_tokens': 1000, 'max_chunk_tokens': 200})
    require(len(context['chunks']) > 0 and context['total_tokens'] <= 1000 and
            all(chunk['id'].startswith('note:') and chunk['citation'] > 0 for chunk in context['chunks']) and
            '[C1]' in context['rendered_context'], 'Hermes native context lost citations or budget')
    print('HARNESS_RESULT=' + json.dumps({'version': __version__, 'python_version': sys.version.split()[0], 'mcp_sdk_version': importlib.metadata.version('mcp'),
        'catalog': sorted(name.removeprefix('mcp__graphrag__') for name in names),
        'exercised_tools': sorted(selected), 'registry_names': sorted(names),
        'native_runtime_api': 'register_mcp_servers/tools.registry.dispatch',
        'hermes_read_openclaw_a': True, 'same_instance_replay': True,
        'native_context_citations': True, 'native_context_tokens': context['total_tokens'],
        'trusted_principal': saved['record']['provenance']['instance_id'],
        'note_count': len(found['records']), 'record_id': saved['record']['id']}))
finally:
    shutdown_mcp_servers()
