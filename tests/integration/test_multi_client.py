"""
멀티 클라이언트 통합 테스트 (ARK-165)

실제 멀티 클라이언트 시나리오를 재현하여 session 격리가 정상 작동하는지 검증합니다.

테스트 시나리오:
1. test_multi_client_isolation: 두 클라이언트가 다른 노트북 사용
2. test_notebook_switching: 한 클라이언트가 노트북 전환
3. test_concurrent_clients: 100개 클라이언트 동시 실행
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from jupyter_mcp_server.server import session_store
from jupyter_mcp_server.tools.use_notebook_tool import UseNotebookTool
from jupyter_mcp_server.tools.execute_cell_tool import ExecuteCellTool
from jupyter_mcp_server.tools._base import ServerMode


@pytest.mark.asyncio
async def test_multi_client_isolation(monkeypatch):
    """
    Scenario 1: 두 클라이언트가 다른 노트북 사용

    클라이언트 A와 B가 각자 다른 노트북(a.ipynb, b.ipynb)을 사용하고,
    서로의 작업이 독립적으로 진행되는지 검증합니다.
    """
    # Setup
    use_notebook = UseNotebookTool()
    execute_cell = ExecuteCellTool()

    # Mock dependencies for use_notebook
    async def mock_check_path(*args, **kwargs):
        return True, None
    monkeypatch.setattr(use_notebook, '_check_path_local', mock_check_path)

    mock_kernel_manager = Mock()
    mock_kernel_manager.__contains__ = Mock(return_value=True)
    mock_contents_manager = AsyncMock()
    mock_notebook_manager = Mock()
    mock_notebook_manager.__contains__ = Mock(return_value=False)
    mock_notebook_manager.add_notebook = Mock()
    mock_notebook_manager.set_current_notebook = Mock()

    # Mock notebook content retrieval
    mock_contents_manager.get = AsyncMock(return_value={
        'content': {
            'cells': [],
            'metadata': {},
            'nbformat': 4,
            'nbformat_minor': 4
        }
    })

    # Client A: Create session and connect notebook
    session_a = "client-A-uuid-123"
    result_a_connect = await use_notebook.execute(
        mode=ServerMode.JUPYTER_SERVER,
        notebook_path="a.ipynb",
        notebook_name="a",
        session_id=session_a,
        kernel_id="kernel-a",
        kernel_manager=mock_kernel_manager,
        contents_manager=mock_contents_manager,
        notebook_manager=mock_notebook_manager,
        use_mode="connect"
    )

    # Client B: Create session and connect notebook (동시)
    session_b = "client-B-uuid-456"
    result_b_connect = await use_notebook.execute(
        mode=ServerMode.JUPYTER_SERVER,
        notebook_path="b.ipynb",
        notebook_name="b",
        session_id=session_b,
        kernel_id="kernel-b",
        kernel_manager=mock_kernel_manager,
        contents_manager=mock_contents_manager,
        notebook_manager=mock_notebook_manager,
        use_mode="connect"
    )

    # Mock execution for ExecuteCellTool
    monkeypatch.setattr(execute_cell, '_write_outputs_to_cell', AsyncMock())

    async def mock_get_jupyter_ydoc(*args, **kwargs):
        return None

    async def mock_execute_via_execution_stack(*args, **kwargs):
        return ["test output"]

    monkeypatch.setattr('jupyter_mcp_server.tools.execute_cell_tool.get_jupyter_ydoc', mock_get_jupyter_ydoc)
    monkeypatch.setattr('jupyter_mcp_server.tools.execute_cell_tool.execute_via_execution_stack', mock_execute_via_execution_stack)

    # Mock file reading
    import nbformat
    mock_notebook = nbformat.v4.new_notebook()
    mock_notebook.cells = [nbformat.v4.new_code_cell("test"), nbformat.v4.new_code_cell("test2")]

    def mock_open(*args, **kwargs):
        from io import StringIO
        content = StringIO(nbformat.writes(mock_notebook))
        return content

    monkeypatch.setattr('builtins.open', mock_open)

    # Mock ServerContext
    mock_context = Mock()
    mock_serverapp = Mock()
    mock_serverapp.root_dir = "/root"
    mock_serverapp.web_app.settings = {"file_id_manager": Mock(get_id=Mock(return_value=None), index=Mock(return_value="file-id-1"))}
    mock_context.serverapp = mock_serverapp

    def mock_get_server_context():
        return mock_context

    monkeypatch.setattr('jupyter_mcp_server.jupyter_extension.context.get_server_context', mock_get_server_context)

    # Client A: Execute cell (session_id만 전달!)
    result_a1 = await execute_cell.execute(
        mode=ServerMode.JUPYTER_SERVER,
        cell_index=0,
        session_id=session_a,
        kernel_manager=mock_kernel_manager,
        serverapp=mock_serverapp
    )

    # Client B: Execute cell (동시)
    result_b1 = await execute_cell.execute(
        mode=ServerMode.JUPYTER_SERVER,
        cell_index=0,
        session_id=session_b,
        kernel_manager=mock_kernel_manager,
        serverapp=mock_serverapp
    )

    # Client A: Continue working
    result_a2 = await execute_cell.execute(
        mode=ServerMode.JUPYTER_SERVER,
        cell_index=1,
        session_id=session_a,
        kernel_manager=mock_kernel_manager,
        serverapp=mock_serverapp
    )

    # Verify isolation - notebook paths should be correct
    assert result_a1["notebook_path"] == "a.ipynb"
    assert result_b1["notebook_path"] == "b.ipynb"
    assert result_a2["notebook_path"] == "a.ipynb"

    # Verify session_ids in responses
    assert result_a1["session_id"] == session_a
    assert result_b1["session_id"] == session_b
    assert result_a2["session_id"] == session_a

    # Verify session contexts are independent
    ctx_a = session_store.get(session_a)
    ctx_b = session_store.get(session_b)
    assert ctx_a.notebook_path == "a.ipynb"
    assert ctx_b.notebook_path == "b.ipynb"
    assert ctx_a.kernel_id == "kernel-a"
    assert ctx_b.kernel_id == "kernel-b"

    # Cleanup
    session_store._sessions.clear()


@pytest.mark.asyncio
async def test_notebook_switching(monkeypatch):
    """
    Scenario 2: 한 클라이언트가 노트북 전환

    한 클라이언트(session_id)가 여러 노트북을 순차적으로 전환하며,
    SessionStore의 context가 정확히 업데이트되는지 검증합니다.
    """
    use_notebook = UseNotebookTool()
    execute_cell = ExecuteCellTool()

    session_id = "client-uuid-789"

    # Mock dependencies
    async def mock_check_path(*args, **kwargs):
        return True, None
    monkeypatch.setattr(use_notebook, '_check_path_local', mock_check_path)

    mock_kernel_manager = Mock()
    mock_kernel_manager.__contains__ = Mock(return_value=True)
    mock_contents_manager = AsyncMock()
    mock_notebook_manager = Mock()
    mock_notebook_manager.__contains__ = Mock(return_value=False)
    mock_notebook_manager.add_notebook = Mock()
    mock_notebook_manager.set_current_notebook = Mock()

    mock_contents_manager.get = AsyncMock(return_value={
        'content': {
            'cells': [],
            'metadata': {},
            'nbformat': 4,
            'nbformat_minor': 4
        }
    })

    monkeypatch.setattr(execute_cell, '_write_outputs_to_cell', AsyncMock())

    async def mock_get_jupyter_ydoc(*args, **kwargs):
        return None

    async def mock_execute_via_execution_stack(*args, **kwargs):
        return ["output"]

    monkeypatch.setattr('jupyter_mcp_server.tools.execute_cell_tool.get_jupyter_ydoc', mock_get_jupyter_ydoc)
    monkeypatch.setattr('jupyter_mcp_server.tools.execute_cell_tool.execute_via_execution_stack', mock_execute_via_execution_stack)

    import nbformat
    mock_notebook = nbformat.v4.new_notebook()
    mock_notebook.cells = [nbformat.v4.new_code_cell("test"), nbformat.v4.new_code_cell("test2")]

    def mock_open(*args, **kwargs):
        from io import StringIO
        content = StringIO(nbformat.writes(mock_notebook))
        return content

    monkeypatch.setattr('builtins.open', mock_open)

    mock_context = Mock()
    mock_serverapp = Mock()
    mock_serverapp.root_dir = "/root"
    mock_serverapp.web_app.settings = {"file_id_manager": Mock(get_id=Mock(return_value=None), index=Mock(return_value="file-id-2"))}
    mock_context.serverapp = mock_serverapp

    def mock_get_server_context():
        return mock_context

    monkeypatch.setattr('jupyter_mcp_server.jupyter_extension.context.get_server_context', mock_get_server_context)

    # Connect to first notebook
    await use_notebook.execute(
        mode=ServerMode.JUPYTER_SERVER,
        notebook_path="1.ipynb",
        notebook_name="1",
        session_id=session_id,
        kernel_id="kernel-1",
        kernel_manager=mock_kernel_manager,
        contents_manager=mock_contents_manager,
        notebook_manager=mock_notebook_manager,
        use_mode="connect"
    )
    result1 = await execute_cell.execute(
        mode=ServerMode.JUPYTER_SERVER,
        cell_index=0,
        session_id=session_id,
        kernel_manager=mock_kernel_manager,
        serverapp=mock_serverapp
    )

    # Switch to second notebook
    await use_notebook.execute(
        mode=ServerMode.JUPYTER_SERVER,
        notebook_path="2.ipynb",
        notebook_name="2",
        session_id=session_id,
        kernel_id="kernel-2",
        kernel_manager=mock_kernel_manager,
        contents_manager=mock_contents_manager,
        notebook_manager=mock_notebook_manager,
        use_mode="connect"
    )
    result2 = await execute_cell.execute(
        mode=ServerMode.JUPYTER_SERVER,
        cell_index=0,
        session_id=session_id,
        kernel_manager=mock_kernel_manager,
        serverapp=mock_serverapp
    )

    # Go back to first notebook
    await use_notebook.execute(
        mode=ServerMode.JUPYTER_SERVER,
        notebook_path="1.ipynb",
        notebook_name="1",
        session_id=session_id,
        kernel_id="kernel-1",
        kernel_manager=mock_kernel_manager,
        contents_manager=mock_contents_manager,
        notebook_manager=mock_notebook_manager,
        use_mode="connect"
    )
    result3 = await execute_cell.execute(
        mode=ServerMode.JUPYTER_SERVER,
        cell_index=1,
        session_id=session_id,
        kernel_manager=mock_kernel_manager,
        serverapp=mock_serverapp
    )

    # Verify notebook switching
    assert result1["notebook_path"] == "1.ipynb"
    assert result2["notebook_path"] == "2.ipynb"
    assert result3["notebook_path"] == "1.ipynb"

    # Session context updated correctly
    ctx = session_store.get(session_id)
    assert ctx.notebook_path == "1.ipynb"
    assert ctx.kernel_id == "kernel-1"

    # Cleanup
    session_store._sessions.clear()


@pytest.mark.asyncio
async def test_concurrent_clients(monkeypatch):
    """
    Scenario 3: 100개 클라이언트 동시 실행

    100개의 독립적인 클라이언트가 동시에 notebook을 사용하고 cell을 실행하며,
    모든 session이 올바르게 격리되고 응답이 정확한지 검증합니다.
    """
    use_notebook = UseNotebookTool()
    execute_cell = ExecuteCellTool()

    # Mock dependencies
    async def mock_check_path(*args, **kwargs):
        return True, None
    monkeypatch.setattr(use_notebook, '_check_path_local', mock_check_path)

    mock_kernel_manager = Mock()
    mock_kernel_manager.__contains__ = Mock(return_value=True)
    mock_contents_manager = AsyncMock()
    mock_notebook_manager = Mock()
    mock_notebook_manager.__contains__ = Mock(return_value=False)
    mock_notebook_manager.add_notebook = Mock()
    mock_notebook_manager.set_current_notebook = Mock()

    mock_contents_manager.get = AsyncMock(return_value={
        'content': {
            'cells': [],
            'metadata': {},
            'nbformat': 4,
            'nbformat_minor': 4
        }
    })

    monkeypatch.setattr(execute_cell, '_write_outputs_to_cell', AsyncMock())

    async def mock_get_jupyter_ydoc(*args, **kwargs):
        return None

    async def mock_execute_via_execution_stack(*args, **kwargs):
        return [f"output"]

    monkeypatch.setattr('jupyter_mcp_server.tools.execute_cell_tool.get_jupyter_ydoc', mock_get_jupyter_ydoc)
    monkeypatch.setattr('jupyter_mcp_server.tools.execute_cell_tool.execute_via_execution_stack', mock_execute_via_execution_stack)

    import nbformat
    mock_notebook = nbformat.v4.new_notebook()
    mock_notebook.cells = [nbformat.v4.new_code_cell(f"print({i})") for i in range(5)]

    def mock_open(*args, **kwargs):
        from io import StringIO
        content = StringIO(nbformat.writes(mock_notebook))
        return content

    monkeypatch.setattr('builtins.open', mock_open)

    mock_context = Mock()
    mock_serverapp = Mock()
    mock_serverapp.root_dir = "/root"
    mock_serverapp.web_app.settings = {"file_id_manager": Mock(get_id=Mock(return_value=None), index=Mock(return_value="file-id-3"))}
    mock_context.serverapp = mock_serverapp

    def mock_get_server_context():
        return mock_context

    monkeypatch.setattr('jupyter_mcp_server.jupyter_extension.context.get_server_context', mock_get_server_context)

    async def client_task(client_id: int):
        """단일 클라이언트 작업 시뮬레이션"""
        session_id = f"client-{client_id}"
        notebook_path = f"{client_id}.ipynb"
        kernel_id = f"kernel-{client_id}"

        # Connect notebook
        await use_notebook.execute(
            mode=ServerMode.JUPYTER_SERVER,
            notebook_path=notebook_path,
            notebook_name=str(client_id),
            session_id=session_id,
            kernel_id=kernel_id,
            kernel_manager=mock_kernel_manager,
            contents_manager=mock_contents_manager,
            notebook_manager=mock_notebook_manager,
            use_mode="connect"
        )

        # Execute cells
        results = []
        for i in range(5):
            result = await execute_cell.execute(
                mode=ServerMode.JUPYTER_SERVER,
                cell_index=i,
                session_id=session_id,
                kernel_manager=mock_kernel_manager,
                serverapp=mock_serverapp
            )
            results.append(result)

        # Verify all results have correct session_id and notebook_path
        for result in results:
            assert result["session_id"] == session_id, f"Client {client_id}: session_id mismatch"
            assert result["notebook_path"] == notebook_path, f"Client {client_id}: notebook_path mismatch"

        return session_id

    # Run 100 clients concurrently
    tasks = [client_task(i) for i in range(100)]
    session_ids = await asyncio.gather(*tasks)

    # Verify all sessions created
    assert len(session_ids) == 100
    for session_id in session_ids:
        ctx = session_store.get(session_id)
        assert ctx is not None, f"Session {session_id} not found in store"
        assert ctx.notebook_path == f"{session_id.split('-')[1]}.ipynb"
        assert ctx.kernel_id == f"kernel-{session_id.split('-')[1]}"

    # Cleanup
    session_store._sessions.clear()
