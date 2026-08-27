import json
import logging
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
from graphiti_core.search.search_filters import SearchFilters
from neo4j.time import DateTime as Neo4jDateTime
from pydantic import ValidationError

import graphiti_mcp_server as server
from config.schema import GraphitiAppConfig, GraphitiConfig
from utils.formatting import to_edge_result, to_node_result


def test_search_mode_defaults_to_hybrid_and_rejects_unknown_value():
    assert GraphitiAppConfig().search_mode == 'hybrid'
    assert GraphitiAppConfig(search_mode='bm25_rerank').search_mode == 'bm25_rerank'
    with pytest.raises(ValidationError):
        GraphitiAppConfig(search_mode='semantic-fallback')


def test_rerank_candidates_default_bounds_and_env_override(monkeypatch, tmp_path):
    assert GraphitiAppConfig().rerank_candidates == 200
    with pytest.raises(ValidationError):
        GraphitiAppConfig(rerank_candidates=0)
    with pytest.raises(ValidationError):
        GraphitiAppConfig(rerank_candidates=1001)

    config_path = tmp_path / 'config.yaml'
    config_path.write_text('graphiti:\n  rerank_candidates: ${GRAPHITI_RERANK_CANDIDATES:200}\n')
    monkeypatch.setenv('CONFIG_PATH', str(config_path))
    monkeypatch.setenv('GRAPHITI_RERANK_CANDIDATES', '321')

    assert GraphitiConfig().graphiti.rerank_candidates == 321


def _entity_edge(
    uuid: str,
    *,
    invalid_at: datetime | None = None,
    expired_at: datetime | None = None,
) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        name='RELATES_TO',
        fact=f'fact {uuid}',
        group_id='g',
        source_node_uuid=f'{uuid}-source',
        target_node_uuid=f'{uuid}-target',
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        invalid_at=invalid_at,
        expired_at=expired_at,
    )


def test_edge_result_similarity_defaults_to_none_and_accepts_score():
    edge = _entity_edge('edge-1')

    assert to_edge_result(edge)['similarity'] is None
    assert to_edge_result(edge, similarity=0.75)['similarity'] == 0.75


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
        server._node_read_search_config('hybrid', 7, 'center')
        is NODE_HYBRID_SEARCH_NODE_DISTANCE
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
    client = SimpleNamespace(search=AsyncMock(), search_=AsyncMock(return_value=SimpleNamespace(edges=[])))
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


@pytest.mark.asyncio
async def test_bm25_rerank_orders_live_candidates_by_cosine(monkeypatch):
    invalidated_at = datetime(2026, 2, 1, tzinfo=timezone.utc)
    candidates = [
        _entity_edge('dead-invalid', invalid_at=invalidated_at),
        _entity_edge('live-low'),
        _entity_edge('dead-expired', expired_at=invalidated_at),
        _entity_edge('live-high'),
    ]
    driver = SimpleNamespace(
        execute_query=AsyncMock(
            return_value=(
                [
                    {'uuid': 'live-low', 'similarity': 0.25},
                    {'uuid': 'live-high', 'similarity': 0.9},
                ],
                None,
                None,
            )
        )
    )
    embedder = SimpleNamespace(create=AsyncMock(return_value=[0.1, 0.2]))
    client = SimpleNamespace(
        driver=driver,
        embedder=embedder,
        search=AsyncMock(),
        search_=AsyncMock(return_value=SimpleNamespace(edges=candidates)),
    )
    service = SimpleNamespace(get_client=AsyncMock(return_value=client))
    cfg = GraphitiConfig()
    cfg.graphiti.search_mode = 'bm25_rerank'
    cfg.graphiti.rerank_candidates = 4
    monkeypatch.setattr(server, 'graphiti_service', service)
    monkeypatch.setattr(server, 'config', cfg, raising=False)

    response = await server.search_memory_facts(
        'line one\nline two',
        group_ids=['g'],
        max_facts=2,
        center_node_uuid='center',
        edge_types=['RELATES_TO'],
    )

    assert [fact['uuid'] for fact in response['facts']] == ['live-high', 'live-low']
    assert [fact['similarity'] for fact in response['facts']] == [0.9, 0.25]
    client.search.assert_not_awaited()
    search_kwargs = client.search_.await_args.kwargs
    assert search_kwargs['config'].limit == 4
    assert search_kwargs['group_ids'] == ['g']
    assert search_kwargs['center_node_uuid'] == 'center'
    assert search_kwargs['search_filter'].edge_types == ['RELATES_TO']
    embedder.create.assert_awaited_once_with(input_data=['line one line two'])
    driver.execute_query.assert_awaited_once()
    query = driver.execute_query.await_args.args[0]
    query_kwargs = driver.execute_query.await_args.kwargs
    assert 'r.invalid_at IS NULL' in query
    assert 'r.expired_at IS NULL' in query
    assert 'r.fact_embedding IS NOT NULL' in query
    assert query_kwargs['uuids'] == [edge.uuid for edge in candidates]
    assert query_kwargs['qv'] == [0.1, 0.2]


@pytest.mark.asyncio
@pytest.mark.parametrize('search_mode', ['bm25', 'hybrid'])
async def test_non_rerank_fact_search_has_none_similarity(monkeypatch, search_mode):
    edge = _entity_edge(f'{search_mode}-edge')
    client = SimpleNamespace(
        search=AsyncMock(return_value=[edge]),
        search_=AsyncMock(return_value=SimpleNamespace(edges=[edge])),
    )
    service = SimpleNamespace(get_client=AsyncMock(return_value=client))
    cfg = GraphitiConfig()
    cfg.graphiti.search_mode = search_mode
    monkeypatch.setattr(server, 'graphiti_service', service)
    monkeypatch.setattr(server, 'config', cfg, raising=False)

    response = await server.search_memory_facts('query', max_facts=1)

    assert response['facts'][0]['similarity'] is None


@pytest.mark.asyncio
async def test_bm25_rerank_embedder_failure_falls_back_with_warning(monkeypatch, caplog):
    candidates = [_entity_edge('first'), _entity_edge('second'), _entity_edge('third')]
    driver = SimpleNamespace(execute_query=AsyncMock())
    embedder = SimpleNamespace(create=AsyncMock(side_effect=RuntimeError('embedder unavailable')))
    client = SimpleNamespace(
        driver=driver,
        embedder=embedder,
        search=AsyncMock(),
        search_=AsyncMock(return_value=SimpleNamespace(edges=candidates)),
    )
    service = SimpleNamespace(get_client=AsyncMock(return_value=client))
    cfg = GraphitiConfig()
    cfg.graphiti.search_mode = 'bm25_rerank'
    monkeypatch.setattr(server, 'graphiti_service', service)
    monkeypatch.setattr(server, 'config', cfg, raising=False)
    caplog.set_level(logging.WARNING, logger=server.__name__)

    response = await server.search_memory_facts('query', max_facts=2)

    assert [fact['uuid'] for fact in response['facts']] == ['first', 'second']
    assert [fact['similarity'] for fact in response['facts']] == [None, None]
    assert 'falling back to plain BM25 order' in caplog.text
    driver.execute_query.assert_not_awaited()


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
