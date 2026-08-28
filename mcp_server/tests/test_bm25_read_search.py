import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search import search as core_search
from graphiti_core.search.search_config import (
    EdgeReranker,
    EdgeSearchMethod,
    NodeReranker,
    NodeSearchMethod,
)
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import ComparisonOperator, SearchFilters
from neo4j.time import DateTime as Neo4jDateTime
from pydantic import ValidationError

import graphiti_mcp_server as server
from config.schema import GraphitiAppConfig, GraphitiConfig
from utils.formatting import to_node_result


def _edge(uuid: str) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        name='RELATES_TO',
        fact=f'Fact {uuid}',
        source_node_uuid='source',
        target_node_uuid='target',
        group_id='g',
        created_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
    )


def test_search_mode_defaults_to_hybrid_and_rejects_unknown_value():
    assert GraphitiAppConfig().search_mode == 'hybrid'
    with pytest.raises(ValidationError):
        GraphitiAppConfig(search_mode='semantic-fallback')


def test_node_result_attributes_are_json_safe_for_neo4j_temporal_values():
    node = EntityNode(
        uuid='node-1',
        name='Example',
        group_id='g',
        created_at=datetime(2026, 7, 28, tzinfo=timezone.utc),
        attributes={
            'observed_at': Neo4jDateTime(2026, 7, 28, 12, 34, 56),
            'nested': [Neo4jDateTime(2026, 7, 27, 1, 2, 3)],
        },
    )

    result = to_node_result(node)

    assert result['attributes']['observed_at'] == '2026-07-28T12:34:56.000000000'
    assert result['attributes']['nested'] == ['2026-07-27T01:02:03.000000000']
    json.dumps(result)


def test_default_node_search_configuration_is_unchanged():
    assert server._node_read_search_config('hybrid', 7, None) is NODE_HYBRID_SEARCH_RRF
    assert (
        server._node_read_search_config('hybrid', 7, 'center') is NODE_HYBRID_SEARCH_NODE_DISTANCE
    )


def test_bm25_search_configurations_have_only_bm25_and_preserve_limits():
    node_config = server._node_read_search_config('bm25', 7, None)
    assert node_config.limit == 7
    assert node_config.node_config.search_methods == [NodeSearchMethod.bm25]
    assert node_config.node_config.reranker == NodeReranker.rrf

    centered_node_config = server._node_read_search_config('bm25', 8, 'center')
    assert centered_node_config.node_config.search_methods == [NodeSearchMethod.bm25]
    assert centered_node_config.node_config.reranker == NodeReranker.node_distance

    edge_config = server._edge_bm25_search_config(9, None)
    assert edge_config.limit == 9
    assert edge_config.edge_config.search_methods == [EdgeSearchMethod.bm25]
    assert edge_config.edge_config.reranker == EdgeReranker.rrf

    centered_edge_config = server._edge_bm25_search_config(6, 'center')
    assert centered_edge_config.edge_config.search_methods == [EdgeSearchMethod.bm25]
    assert centered_edge_config.edge_config.reranker == EdgeReranker.node_distance


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'config',
    [
        server._node_read_search_config('bm25', 4, None),
        server._edge_bm25_search_config(4, None),
    ],
)
async def test_bm25_core_search_never_calls_embedder(monkeypatch, config):
    async def no_results(*args, **kwargs):
        return []

    async def fail_if_embedded(*args, **kwargs):
        raise AssertionError('BM25-only search called the embedder')

    monkeypatch.setattr('graphiti_core.search.search.node_fulltext_search', no_results)
    monkeypatch.setattr('graphiti_core.search.search.edge_fulltext_search', no_results)
    clients = SimpleNamespace(
        driver=SimpleNamespace(),
        embedder=SimpleNamespace(create=fail_if_embedded),
        cross_encoder=SimpleNamespace(),
    )

    results = await core_search(
        clients,
        query='keyword',
        group_ids=['g'],
        config=config,
        search_filter=SearchFilters(),
    )

    assert results.nodes == []
    assert results.edges == []


@pytest.mark.asyncio
async def test_bm25_fact_search_uses_explicit_config_and_preserves_filters(monkeypatch):
    client = SimpleNamespace(
        search=AsyncMock(),
        search_=AsyncMock(return_value=SimpleNamespace(edges=[], edge_reranker_scores=[])),
    )
    service = SimpleNamespace(get_client=AsyncMock(return_value=client))
    cfg = GraphitiConfig()
    cfg.graphiti.search_mode = 'bm25'
    monkeypatch.setattr(server, 'graphiti_service', service)
    monkeypatch.setattr(server, 'config', cfg, raising=False)

    response = await server.search_memory_facts(
        'query',
        group_ids=['g'],
        max_facts=4,
        center_node_uuid='center',
        edge_types=['RELATES_TO'],
        valid_at_after='2024-01-01T00:00:00Z',
        temporal_mode='current',
    )

    assert response['facts'] == []
    client.search.assert_not_awaited()
    kwargs = client.search_.await_args.kwargs
    assert kwargs['config'].limit == 4
    assert kwargs['config'].edge_config.search_methods == [EdgeSearchMethod.bm25]
    assert kwargs['group_ids'] == ['g']
    assert kwargs['center_node_uuid'] == 'center'
    assert kwargs['search_filter'].edge_types == ['RELATES_TO']
    assert kwargs['search_filter'].valid_at[0] is not None
    assert (
        kwargs['search_filter'].invalid_at[0][0].comparison_operator == ComparisonOperator.is_null
    )
    assert (
        kwargs['search_filter'].expired_at[0][0].comparison_operator == ComparisonOperator.is_null
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('scores', 'expected_scores'),
    [
        ([0.12345678, 0.5, 0.98765432], [0.123457, 0.5, 0.987654]),
        ([], [None, None, None]),
        ([0.5], [0.5, None, None]),
    ],
)
async def test_bm25_fact_search_exposes_available_reranker_scores(
    monkeypatch, scores, expected_scores
):
    monkeypatch.delenv('GRAPHITI_FACT_SCORE_MODE', raising=False)
    edges = [_edge('edge-1'), _edge('edge-2'), _edge('edge-3')]
    client = SimpleNamespace(
        search=AsyncMock(),
        search_=AsyncMock(return_value=SimpleNamespace(edges=edges, edge_reranker_scores=scores)),
    )
    service = SimpleNamespace(get_client=AsyncMock(return_value=client))
    cfg = GraphitiConfig()
    cfg.graphiti.search_mode = 'bm25'
    monkeypatch.setattr(server, 'graphiti_service', service)
    monkeypatch.setattr(server, 'config', cfg, raising=False)

    response = await server.search_memory_facts('query', max_facts=3)

    for fact, expected_score in zip(response['facts'], expected_scores, strict=True):
        if expected_score is None:
            assert 'score' not in fact
        else:
            assert fact['score'] == expected_score


@pytest.mark.asyncio
async def test_bm25_fact_search_suppresses_scores_for_centered_search(monkeypatch):
    """center_node_uuid selects the node_distance reranker, whose scores are
    per SOURCE NODE. Core expands one node score across every edge sharing that
    source, so the list is not index-aligned with the returned edges — emitting
    it would attach another node's score to a fact. Scores must be omitted."""
    monkeypatch.delenv('GRAPHITI_FACT_SCORE_MODE', raising=False)
    edges = [_edge('edge-1'), _edge('edge-2'), _edge('edge-3')]
    client = SimpleNamespace(
        search=AsyncMock(),
        search_=AsyncMock(
            # one score for a single source node shared by three edges
            return_value=SimpleNamespace(edges=edges, edge_reranker_scores=[0.9])
        ),
    )
    service = SimpleNamespace(get_client=AsyncMock(return_value=client))
    cfg = GraphitiConfig()
    cfg.graphiti.search_mode = 'bm25'
    monkeypatch.setattr(server, 'graphiti_service', service)
    monkeypatch.setattr(server, 'config', cfg, raising=False)

    response = await server.search_memory_facts('query', max_facts=3, center_node_uuid='center')

    assert len(response['facts']) == 3
    for fact in response['facts']:
        assert 'score' not in fact


@pytest.mark.asyncio
async def test_bm25_fact_search_semantic_scores(monkeypatch):
    edges = [_edge('edge-1'), _edge('edge-2')]
    embedder = SimpleNamespace(create=AsyncMock(return_value=[1.0, 2.0, 2.0]))
    driver = SimpleNamespace(
        execute_query=AsyncMock(
            return_value=(
                [
                    {'uuid': 'edge-1', 'emb': [1.0, 2.0, 2.0]},
                    {'uuid': 'edge-2', 'emb': [0.0, 3.0, 4.0]},
                ],
                None,
                None,
            )
        )
    )
    client = SimpleNamespace(
        search=AsyncMock(),
        search_=AsyncMock(return_value=SimpleNamespace(edges=edges, edge_reranker_scores=[])),
        embedder=embedder,
        driver=driver,
    )
    service = SimpleNamespace(get_client=AsyncMock(return_value=client))
    cfg = GraphitiConfig()
    cfg.graphiti.search_mode = 'bm25'
    monkeypatch.setenv('GRAPHITI_FACT_SCORE_MODE', 'semantic')
    monkeypatch.setattr(server, 'graphiti_service', service)
    monkeypatch.setattr(server, 'config', cfg, raising=False)

    response = await server.search_memory_facts('query', max_facts=2)

    assert [fact['score'] for fact in response['facts']] == [1.0, 0.933333]
    embedder.create.assert_awaited_once_with(input_data=['query'])
    query = driver.execute_query.await_args.args[0]
    assert 'WHERE e.uuid IN $uuids' in query
    assert 'ORDER BY' not in query
    assert driver.execute_query.await_args.kwargs['uuids'] == ['edge-1', 'edge-2']


@pytest.mark.asyncio
async def test_bm25_fact_search_semantic_scores_omit_missing_embeddings(monkeypatch):
    edges = [_edge('edge-1'), _edge('edge-2')]
    client = SimpleNamespace(
        search=AsyncMock(),
        search_=AsyncMock(return_value=SimpleNamespace(edges=edges, edge_reranker_scores=[])),
        embedder=SimpleNamespace(create=AsyncMock(return_value=[1.0, 0.0, 0.0])),
        driver=SimpleNamespace(
            execute_query=AsyncMock(
                return_value=([{'uuid': 'edge-1', 'emb': [0.0, 1.0, 0.0]}], None, None)
            )
        ),
    )
    service = SimpleNamespace(get_client=AsyncMock(return_value=client))
    cfg = GraphitiConfig()
    cfg.graphiti.search_mode = 'bm25'
    monkeypatch.setenv('GRAPHITI_FACT_SCORE_MODE', 'semantic')
    monkeypatch.setattr(server, 'graphiti_service', service)
    monkeypatch.setattr(server, 'config', cfg, raising=False)

    response = await server.search_memory_facts('query', max_facts=2)

    assert response['facts'][0]['score'] == 0.0
    assert 'score' not in response['facts'][1]


@pytest.mark.asyncio
async def test_bm25_fact_search_semantic_scores_isolate_embedder_failures(monkeypatch):
    edges = [_edge('edge-1'), _edge('edge-2')]
    embedder = SimpleNamespace(create=AsyncMock(side_effect=RuntimeError('embedder unavailable')))
    driver = SimpleNamespace(execute_query=AsyncMock())
    client = SimpleNamespace(
        search=AsyncMock(),
        search_=AsyncMock(return_value=SimpleNamespace(edges=edges, edge_reranker_scores=[])),
        embedder=embedder,
        driver=driver,
    )
    service = SimpleNamespace(get_client=AsyncMock(return_value=client))
    cfg = GraphitiConfig()
    cfg.graphiti.search_mode = 'bm25'
    monkeypatch.setenv('GRAPHITI_FACT_SCORE_MODE', 'semantic')
    monkeypatch.setattr(server, 'graphiti_service', service)
    monkeypatch.setattr(server, 'config', cfg, raising=False)

    response = await server.search_memory_facts('query', max_facts=2)

    assert len(response['facts']) == 2
    assert all('score' not in fact for fact in response['facts'])
    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_bm25_fact_search_none_mode_omits_scores(monkeypatch):
    edges = [_edge('edge-1'), _edge('edge-2')]
    embedder = SimpleNamespace(create=AsyncMock())
    driver = SimpleNamespace(execute_query=AsyncMock())
    client = SimpleNamespace(
        search=AsyncMock(),
        search_=AsyncMock(
            return_value=SimpleNamespace(edges=edges, edge_reranker_scores=[1.0, 0.5])
        ),
        embedder=embedder,
        driver=driver,
    )
    service = SimpleNamespace(get_client=AsyncMock(return_value=client))
    cfg = GraphitiConfig()
    cfg.graphiti.search_mode = 'bm25'
    monkeypatch.setenv('GRAPHITI_FACT_SCORE_MODE', 'none')
    monkeypatch.setattr(server, 'graphiti_service', service)
    monkeypatch.setattr(server, 'config', cfg, raising=False)

    response = await server.search_memory_facts('query', max_facts=2)

    assert all('score' not in fact for fact in response['facts'])
    embedder.create.assert_not_awaited()
    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_fact_search_surfaces_contradictory_temporal_filter(monkeypatch):
    service = SimpleNamespace(get_client=AsyncMock())
    monkeypatch.setattr(server, 'graphiti_service', service)

    response = await server.search_memory_facts(
        'query',
        invalid_at_after='2026-01-01T00:00:00Z',
        temporal_mode='current',
    )

    assert response['error'].startswith('Invalid date filter:')
    assert 'cannot be combined' in response['error']
    service.get_client.assert_not_awaited()


@pytest.mark.asyncio
async def test_hybrid_fact_search_keeps_existing_client_search_call(monkeypatch):
    client = SimpleNamespace(search=AsyncMock(return_value=[]), search_=AsyncMock())
    service = SimpleNamespace(get_client=AsyncMock(return_value=client))
    cfg = GraphitiConfig()
    monkeypatch.setattr(server, 'graphiti_service', service)
    monkeypatch.setattr(server, 'config', cfg, raising=False)

    response = await server.search_memory_facts('query', group_ids='g', max_facts=3)

    assert response['facts'] == []
    client.search_.assert_not_awaited()
    client.search.assert_awaited_once()
    assert client.search.await_args.kwargs['num_results'] == 3
    assert client.search.await_args.kwargs['group_ids'] == ['g']


@pytest.mark.asyncio
async def test_bm25_errors_remain_explicit(monkeypatch):
    client = SimpleNamespace(search_=AsyncMock(side_effect=RuntimeError('database unavailable')))
    service = SimpleNamespace(get_client=AsyncMock(return_value=client))
    cfg = GraphitiConfig()
    cfg.graphiti.search_mode = 'bm25'
    monkeypatch.setattr(server, 'graphiti_service', service)
    monkeypatch.setattr(server, 'config', cfg, raising=False)

    response = await server.search_memory_facts('query')

    assert response['error'] == 'Error searching facts: database unavailable'
