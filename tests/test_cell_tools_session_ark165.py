"""
Unit tests for Cell Operation Tools ARK-165 features

Tests session_id support for 5 cell operation tools:
1. ExecuteCellTool
2. DeleteCellTool
3. InsertCellTool
4. OverwriteCellSourceTool
5. ReadCellTool
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from jupyter_mcp_server.tools.execute_cell_tool import ExecuteCellTool
from jupyter_mcp_server.tools.delete_cell_tool import DeleteCellTool
from jupyter_mcp_server.tools.insert_cell_tool import InsertCellTool
from jupyter_mcp_server.tools.overwrite_cell_source_tool import OverwriteCellSourceTool
from jupyter_mcp_server.tools.read_cell_tool import ReadCellTool
from jupyter_mcp_server.tools._base import ServerMode


class TestExecuteCellToolSession:
    """ExecuteCellTool session_id support tests"""

    @pytest.mark.asyncio
    async def test_execute_cell_with_session_id(self, monkeypatch):
        """session_id 있을 때 SessionStore에서 context 조회"""
        from jupyter_mcp_server.server import session_store

        # Setup session
        session_store.update_notebook(
            session_id="session-1",
            notebook_name="nb1",
            notebook_path="path/nb1.ipynb",
            kernel_id="kernel-1"
        )

        tool = ExecuteCellTool()

        # Mock execution
        mock_outputs = ["output1", "output2"]
        monkeypatch.setattr(tool, '_write_outputs_to_cell', AsyncMock())

        # Mock dependencies for JUPYTER_SERVER mode
        async def mock_get_jupyter_ydoc(*args, **kwargs):
            return None  # Use file mode

        async def mock_execute_via_execution_stack(*args, **kwargs):
            return mock_outputs

        monkeypatch.setattr('jupyter_mcp_server.tools.execute_cell_tool.get_jupyter_ydoc', mock_get_jupyter_ydoc)
        monkeypatch.setattr('jupyter_mcp_server.tools.execute_cell_tool.execute_via_execution_stack', mock_execute_via_execution_stack)

        # Mock file reading
        import nbformat
        mock_notebook = nbformat.v4.new_notebook()
        mock_notebook.cells = [nbformat.v4.new_code_cell("print('hello')")]

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

        # Mock kernel_manager
        mock_kernel_manager = Mock()

        # Execute
        result = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            cell_index=0,
            session_id="session-1",
            kernel_manager=mock_kernel_manager,
            serverapp=mock_serverapp
        )

        # Verify structured output
        assert isinstance(result, dict)
        assert result["session_id"] == "session-1"
        assert result["notebook_path"] == "path/nb1.ipynb"
        assert "outputs" in result
        assert "metadata" in result
        assert result["metadata"]["cell_index"] == 0

        # Cleanup
        session_store._sessions.clear()

    @pytest.mark.asyncio
    async def test_execute_cell_without_session_id(self, monkeypatch):
        """session_id 없을 때 config fallback"""
        mock_config = Mock()
        mock_config.document_id = "default.ipynb"
        mock_config.runtime_id = "default-kernel"
        monkeypatch.setattr('jupyter_mcp_server.config.get_config', lambda: mock_config)

        tool = ExecuteCellTool()

        # Mock execution
        mock_outputs = ["output"]
        monkeypatch.setattr(tool, '_write_outputs_to_cell', AsyncMock())

        # Mock dependencies
        async def mock_get_jupyter_ydoc(*args, **kwargs):
            return None

        async def mock_execute_via_execution_stack(*args, **kwargs):
            return mock_outputs

        monkeypatch.setattr('jupyter_mcp_server.tools.execute_cell_tool.get_jupyter_ydoc', mock_get_jupyter_ydoc)
        monkeypatch.setattr('jupyter_mcp_server.tools.execute_cell_tool.execute_via_execution_stack', mock_execute_via_execution_stack)

        # Mock file reading
        import nbformat
        mock_notebook = nbformat.v4.new_notebook()
        mock_notebook.cells = [nbformat.v4.new_code_cell("print('test')")]

        def mock_open(*args, **kwargs):
            from io import StringIO
            content = StringIO(nbformat.writes(mock_notebook))
            return content

        monkeypatch.setattr('builtins.open', mock_open)

        # Mock ServerContext
        mock_context = Mock()
        mock_serverapp = Mock()
        mock_serverapp.root_dir = "/root"
        mock_serverapp.web_app.settings = {"file_id_manager": Mock(get_id=Mock(return_value=None), index=Mock(return_value="file-id-2"))}
        mock_context.serverapp = mock_serverapp

        def mock_get_server_context():
            return mock_context

        monkeypatch.setattr('jupyter_mcp_server.jupyter_extension.context.get_server_context', mock_get_server_context)

        mock_kernel_manager = Mock()

        # Execute without session_id
        result = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            cell_index=0,
            kernel_manager=mock_kernel_manager,
            serverapp=mock_serverapp
        )

        # Verify fallback to config
        assert result["notebook_path"] == "default.ipynb"
        assert result["session_id"] is None


class TestDeleteCellToolSession:
    """DeleteCellTool session_id support tests"""

    @pytest.mark.asyncio
    async def test_delete_cell_with_session_id(self, monkeypatch):
        """session_id 있을 때 정상 작동"""
        from jupyter_mcp_server.server import session_store

        session_store.update_notebook(
            session_id="session-2",
            notebook_name="nb2",
            notebook_path="path/nb2.ipynb",
            kernel_id="kernel-2"
        )

        tool = DeleteCellTool()

        # Mock _delete_cell_file
        mock_cells = [{"cell_type": "code", "source": "deleted code"}]
        monkeypatch.setattr(tool, '_delete_cell_file', AsyncMock(return_value=mock_cells))

        # Mock ServerContext
        mock_context = Mock()
        mock_context.serverapp = None  # Use file mode

        def mock_get_server_context():
            return mock_context

        monkeypatch.setattr('jupyter_mcp_server.jupyter_extension.context.get_server_context', mock_get_server_context)

        # Execute
        result = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            cell_indices=[0],
            session_id="session-2",
            contents_manager=Mock()
        )

        # Verify
        assert isinstance(result, dict)
        assert result["session_id"] == "session-2"
        assert result["notebook_path"] == "path/nb2.ipynb"
        assert "metadata" in result
        assert result["metadata"]["cell_indices"] == [0]

        # Cleanup
        session_store._sessions.clear()


class TestInsertCellToolSession:
    """InsertCellTool session_id support tests"""

    @pytest.mark.asyncio
    async def test_insert_cell_with_session_id(self, monkeypatch):
        """session_id 있을 때 정상 작동"""
        from jupyter_mcp_server.server import session_store

        session_store.update_notebook(
            session_id="session-3",
            notebook_name="nb3",
            notebook_path="path/nb3.ipynb",
            kernel_id="kernel-3"
        )

        tool = InsertCellTool()

        # Mock _insert_cell_file
        from jupyter_mcp_server.models import Notebook
        mock_notebook = Notebook(cells=[])
        monkeypatch.setattr(tool, '_insert_cell_file', AsyncMock(return_value=(mock_notebook, 0, 1)))

        # Mock ServerContext
        mock_context = Mock()
        mock_context.serverapp = None

        def mock_get_server_context():
            return mock_context

        monkeypatch.setattr('jupyter_mcp_server.jupyter_extension.context.get_server_context', mock_get_server_context)

        # Execute
        result = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            cell_index=0,
            cell_type="code",
            cell_source="new code",
            session_id="session-3",
            contents_manager=Mock()
        )

        # Verify
        assert isinstance(result, dict)
        assert result["session_id"] == "session-3"
        assert result["notebook_path"] == "path/nb3.ipynb"
        assert result["metadata"]["cell_type"] == "code"

        # Cleanup
        session_store._sessions.clear()


class TestOverwriteCellSourceToolSession:
    """OverwriteCellSourceTool session_id support tests"""

    @pytest.mark.asyncio
    async def test_overwrite_cell_with_session_id(self, monkeypatch):
        """session_id 있을 때 정상 작동"""
        from jupyter_mcp_server.server import session_store

        session_store.update_notebook(
            session_id="session-4",
            notebook_name="nb4",
            notebook_path="path/nb4.ipynb",
            kernel_id="kernel-4"
        )

        tool = OverwriteCellSourceTool()

        # Mock _overwrite_cell_file
        mock_diff = "+new line"
        monkeypatch.setattr(tool, '_overwrite_cell_file', AsyncMock(return_value=mock_diff))

        # Mock ServerContext
        mock_context = Mock()
        mock_context.serverapp = None

        def mock_get_server_context():
            return mock_context

        monkeypatch.setattr('jupyter_mcp_server.jupyter_extension.context.get_server_context', mock_get_server_context)

        # Execute
        result = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            cell_index=0,
            cell_source="updated code",
            session_id="session-4",
            contents_manager=Mock()
        )

        # Verify
        assert isinstance(result, dict)
        assert result["session_id"] == "session-4"
        assert result["notebook_path"] == "path/nb4.ipynb"
        assert result["metadata"]["cell_index"] == 0
        assert result["metadata"]["diff"] == mock_diff

        # Cleanup
        session_store._sessions.clear()


class TestReadCellToolSession:
    """ReadCellTool session_id support tests"""

    @pytest.mark.asyncio
    async def test_read_cell_with_session_id(self, monkeypatch):
        """session_id 있을 때 정상 작동"""
        from jupyter_mcp_server.server import session_store

        session_store.update_notebook(
            session_id="session-5",
            notebook_name="nb5",
            notebook_path="path/nb5.ipynb",
            kernel_id="kernel-5"
        )

        tool = ReadCellTool()

        # Mock contents_manager.get
        from jupyter_mcp_server.models import Notebook
        import nbformat
        mock_notebook_raw = nbformat.v4.new_notebook()
        mock_notebook_raw.cells = [nbformat.v4.new_code_cell("test code")]

        mock_contents_manager = AsyncMock()
        mock_contents_manager.get = AsyncMock(return_value={
            'content': mock_notebook_raw
        })

        # Execute
        result = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            cell_index=0,
            session_id="session-5",
            contents_manager=mock_contents_manager
        )

        # Verify
        assert isinstance(result, dict)
        assert result["session_id"] == "session-5"
        assert result["notebook_path"] == "path/nb5.ipynb"
        assert "cell_info" in result
        assert "metadata" in result
        assert result["metadata"]["cell_index"] == 0

        # Cleanup
        session_store._sessions.clear()

    @pytest.mark.asyncio
    async def test_read_cell_out_of_range(self, monkeypatch):
        """Cell index out of range - error case"""
        from jupyter_mcp_server.server import session_store

        session_store.update_notebook(
            session_id="session-6",
            notebook_name="nb6",
            notebook_path="path/nb6.ipynb",
            kernel_id="kernel-6"
        )

        tool = ReadCellTool()

        # Mock contents_manager.get - empty notebook
        import nbformat
        mock_notebook_raw = nbformat.v4.new_notebook()
        mock_notebook_raw.cells = []  # No cells

        mock_contents_manager = AsyncMock()
        mock_contents_manager.get = AsyncMock(return_value={
            'content': mock_notebook_raw
        })

        # Execute with out of range index
        result = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            cell_index=10,  # Out of range
            session_id="session-6",
            contents_manager=mock_contents_manager
        )

        # Verify error case
        assert isinstance(result, dict)
        assert "error" in result
        assert result["session_id"] == "session-6"
        assert "metadata" in result

        # Cleanup
        session_store._sessions.clear()


class TestBackwardCompatibility:
    """Backward compatibility tests for all tools"""

    @pytest.mark.asyncio
    async def test_execute_cell_without_session_backward_compat(self, monkeypatch):
        """ExecuteCellTool: session_id 없이 호출해도 작동"""
        mock_config = Mock()
        mock_config.document_id = "compat.ipynb"
        mock_config.runtime_id = "compat-kernel"
        monkeypatch.setattr('jupyter_mcp_server.config.get_config', lambda: mock_config)

        tool = ExecuteCellTool()

        # Mock execution
        monkeypatch.setattr(tool, '_write_outputs_to_cell', AsyncMock())

        async def mock_get_jupyter_ydoc(*args, **kwargs):
            return None

        async def mock_execute_via_execution_stack(*args, **kwargs):
            return ["test output"]

        monkeypatch.setattr('jupyter_mcp_server.tools.execute_cell_tool.get_jupyter_ydoc', mock_get_jupyter_ydoc)
        monkeypatch.setattr('jupyter_mcp_server.tools.execute_cell_tool.execute_via_execution_stack', mock_execute_via_execution_stack)

        # Mock file reading
        import nbformat
        mock_notebook = nbformat.v4.new_notebook()
        mock_notebook.cells = [nbformat.v4.new_code_cell("test")]

        def mock_open(*args, **kwargs):
            from io import StringIO
            content = StringIO(nbformat.writes(mock_notebook))
            return content

        monkeypatch.setattr('builtins.open', mock_open)

        # Mock ServerContext
        mock_context = Mock()
        mock_serverapp = Mock()
        mock_serverapp.root_dir = "/root"
        mock_serverapp.web_app.settings = {"file_id_manager": Mock(get_id=Mock(return_value=None), index=Mock(return_value="file-id-compat"))}
        mock_context.serverapp = mock_serverapp

        def mock_get_server_context():
            return mock_context

        monkeypatch.setattr('jupyter_mcp_server.jupyter_extension.context.get_server_context', mock_get_server_context)

        # Execute WITHOUT session_id
        result = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            cell_index=0,
            kernel_manager=Mock(),
            serverapp=mock_serverapp
        )

        # Verify it still works
        assert isinstance(result, dict)
        assert "outputs" in result
        assert result["notebook_path"] == "compat.ipynb"
        assert result["session_id"] is None
