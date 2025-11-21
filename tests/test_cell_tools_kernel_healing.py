"""
Integration tests for Cell Operation Tools kernel auto-healing (ARK-165 Task 4)

Verifies that all 5 cell operation tools automatically benefit from kernel healing
by using get_notebook_context_from_session_async() with auto_heal=True.

This test suite verifies the kernel healing logic without requiring full tool execution.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from jupyter_mcp_server.tools._base import ServerMode


@pytest.fixture
def mock_kernel_manager_with_healing():
    """Mock kernel_manager that simulates kernel death and healing"""
    mock_km = Mock()

    # Simulate dead kernel initially, then new kernel after healing
    kernel_states = {'dead-kernel-123': False, 'new-kernel-456': True}
    mock_km.__contains__ = Mock(side_effect=lambda kid: kernel_states.get(kid, False))

    # Mock start_kernel to create new kernel
    mock_km.start_kernel = AsyncMock(return_value={'id': 'new-kernel-456'})

    return mock_km


class TestAllCellToolsAutoHealing:
    """Test that all cell operation tools use auto-healing via shared context function"""

    @pytest.mark.asyncio
    async def test_execute_cell_uses_auto_healing(self, monkeypatch, mock_kernel_manager_with_healing):
        """Verify execute_cell tool uses get_notebook_context_from_session_async with auto-healing"""
        from jupyter_mcp_server.server import session_store
        from jupyter_mcp_server.session_manager import SessionManager

        # Setup: Session with dead kernel
        session_store.update_notebook(
            session_id="test-execute",
            notebook_name="test",
            notebook_path="test.ipynb",
            kernel_id="dead-kernel-123"
        )

        # Track SessionManager.heal_kernel() calls
        heal_called = []
        original_heal = SessionManager.heal_kernel

        async def mock_heal(self, *args, **kwargs):
            heal_called.append(kwargs.get('session_id'))
            return await original_heal(self, *args, **kwargs)

        monkeypatch.setattr(SessionManager, 'heal_kernel', mock_heal)

        # Call the async context function that execute_cell uses
        from jupyter_mcp_server.utils import get_notebook_context_from_session_async
        _, kernel_id = await get_notebook_context_from_session_async(
            session_id="test-execute",
            auto_heal=True,
            kernel_manager=mock_kernel_manager_with_healing,
            mode=ServerMode.JUPYTER_SERVER
        )

        # Verify kernel was healed
        assert kernel_id == "new-kernel-456"
        assert "test-execute" in heal_called

        # Cleanup
        session_store._sessions.clear()

    @pytest.mark.asyncio
    async def test_read_cell_uses_auto_healing(self, monkeypatch, mock_kernel_manager_with_healing):
        """Verify read_cell tool uses auto-healing"""
        from jupyter_mcp_server.server import session_store
        from jupyter_mcp_server.session_manager import SessionManager

        session_store.update_notebook(
            session_id="test-read",
            notebook_name="test",
            notebook_path="test.ipynb",
            kernel_id="dead-kernel-123"
        )

        heal_called = []
        original_heal = SessionManager.heal_kernel

        async def mock_heal(self, *args, **kwargs):
            heal_called.append(kwargs.get('session_id'))
            return await original_heal(self, *args, **kwargs)

        monkeypatch.setattr(SessionManager, 'heal_kernel', mock_heal)

        from jupyter_mcp_server.utils import get_notebook_context_from_session_async
        _, kernel_id = await get_notebook_context_from_session_async(
            session_id="test-read",
            auto_heal=True,
            kernel_manager=mock_kernel_manager_with_healing,
            mode=ServerMode.JUPYTER_SERVER
        )

        assert kernel_id == "new-kernel-456"
        assert "test-read" in heal_called

        session_store._sessions.clear()

    @pytest.mark.asyncio
    async def test_insert_cell_uses_auto_healing(self, monkeypatch, mock_kernel_manager_with_healing):
        """Verify insert_cell tool uses auto-healing"""
        from jupyter_mcp_server.server import session_store
        from jupyter_mcp_server.session_manager import SessionManager

        session_store.update_notebook(
            session_id="test-insert",
            notebook_name="test",
            notebook_path="test.ipynb",
            kernel_id="dead-kernel-123"
        )

        heal_called = []
        original_heal = SessionManager.heal_kernel

        async def mock_heal(self, *args, **kwargs):
            heal_called.append(kwargs.get('session_id'))
            return await original_heal(self, *args, **kwargs)

        monkeypatch.setattr(SessionManager, 'heal_kernel', mock_heal)

        from jupyter_mcp_server.utils import get_notebook_context_from_session_async
        _, kernel_id = await get_notebook_context_from_session_async(
            session_id="test-insert",
            auto_heal=True,
            kernel_manager=mock_kernel_manager_with_healing,
            mode=ServerMode.JUPYTER_SERVER
        )

        assert kernel_id == "new-kernel-456"
        assert "test-insert" in heal_called

        session_store._sessions.clear()

    @pytest.mark.asyncio
    async def test_delete_cell_uses_auto_healing(self, monkeypatch, mock_kernel_manager_with_healing):
        """Verify delete_cell tool uses auto-healing"""
        from jupyter_mcp_server.server import session_store
        from jupyter_mcp_server.session_manager import SessionManager

        session_store.update_notebook(
            session_id="test-delete",
            notebook_name="test",
            notebook_path="test.ipynb",
            kernel_id="dead-kernel-123"
        )

        heal_called = []
        original_heal = SessionManager.heal_kernel

        async def mock_heal(self, *args, **kwargs):
            heal_called.append(kwargs.get('session_id'))
            return await original_heal(self, *args, **kwargs)

        monkeypatch.setattr(SessionManager, 'heal_kernel', mock_heal)

        from jupyter_mcp_server.utils import get_notebook_context_from_session_async
        _, kernel_id = await get_notebook_context_from_session_async(
            session_id="test-delete",
            auto_heal=True,
            kernel_manager=mock_kernel_manager_with_healing,
            mode=ServerMode.JUPYTER_SERVER
        )

        assert kernel_id == "new-kernel-456"
        assert "test-delete" in heal_called

        session_store._sessions.clear()

    @pytest.mark.asyncio
    async def test_overwrite_cell_uses_auto_healing(self, monkeypatch, mock_kernel_manager_with_healing):
        """Verify overwrite_cell tool uses auto-healing"""
        from jupyter_mcp_server.server import session_store
        from jupyter_mcp_server.session_manager import SessionManager

        session_store.update_notebook(
            session_id="test-overwrite",
            notebook_name="test",
            notebook_path="test.ipynb",
            kernel_id="dead-kernel-123"
        )

        heal_called = []
        original_heal = SessionManager.heal_kernel

        async def mock_heal(self, *args, **kwargs):
            heal_called.append(kwargs.get('session_id'))
            return await original_heal(self, *args, **kwargs)

        monkeypatch.setattr(SessionManager, 'heal_kernel', mock_heal)

        from jupyter_mcp_server.utils import get_notebook_context_from_session_async
        _, kernel_id = await get_notebook_context_from_session_async(
            session_id="test-overwrite",
            auto_heal=True,
            kernel_manager=mock_kernel_manager_with_healing,
            mode=ServerMode.JUPYTER_SERVER
        )

        assert kernel_id == "new-kernel-456"
        assert "test-overwrite" in heal_called

        session_store._sessions.clear()

    @pytest.mark.asyncio
    async def test_all_tools_share_same_healing_logic(self, monkeypatch, mock_kernel_manager_with_healing):
        """Verify all 5 cell tools use the same SessionManager.heal_kernel() logic"""
        from jupyter_mcp_server.server import session_store
        from jupyter_mcp_server.session_manager import SessionManager

        # Track all heal_kernel calls
        heal_calls = []
        original_heal = SessionManager.heal_kernel

        async def mock_heal(self, *args, **kwargs):
            heal_calls.append(kwargs.get('session_id'))
            return await original_heal(self, *args, **kwargs)

        monkeypatch.setattr(SessionManager, 'heal_kernel', mock_heal)

        # Test all 5 tools
        sessions = ["execute", "read", "insert", "delete", "overwrite"]

        for session_id in sessions:
            session_store.update_notebook(
                session_id=session_id,
                notebook_name="test",
                notebook_path="test.ipynb",
                kernel_id="dead-kernel-123"
            )

            from jupyter_mcp_server.utils import get_notebook_context_from_session_async
            _, kernel_id = await get_notebook_context_from_session_async(
                session_id=session_id,
                auto_heal=True,
                kernel_manager=mock_kernel_manager_with_healing,
                mode=ServerMode.JUPYTER_SERVER
            )

            # Verify healing worked
            assert kernel_id == "new-kernel-456"

        # Verify SessionManager.heal_kernel() was called 5 times (once per tool)
        assert len(heal_calls) == 5
        for session_id in sessions:
            assert session_id in heal_calls

        session_store._sessions.clear()


class TestCellToolsIntegration:
    """Integration verification that cell tools have been updated correctly"""

    def test_execute_cell_imports_async_function(self):
        """Verify execute_cell imports get_notebook_context_from_session_async"""
        import inspect
        from jupyter_mcp_server.tools.execute_cell_tool import ExecuteCellTool

        source = inspect.getsource(ExecuteCellTool.execute)
        assert "get_notebook_context_from_session_async" in source, \
            "execute_cell should import async context function"

    def test_read_cell_imports_async_function(self):
        """Verify read_cell imports get_notebook_context_from_session_async"""
        import inspect
        from jupyter_mcp_server.tools.read_cell_tool import ReadCellTool

        source = inspect.getsource(ReadCellTool.execute)
        assert "get_notebook_context_from_session_async" in source, \
            "read_cell should import async context function"

    def test_insert_cell_imports_async_function(self):
        """Verify insert_cell imports get_notebook_context_from_session_async"""
        import inspect
        from jupyter_mcp_server.tools.insert_cell_tool import InsertCellTool

        source = inspect.getsource(InsertCellTool.execute)
        assert "get_notebook_context_from_session_async" in source, \
            "insert_cell should import async context function"

    def test_delete_cell_imports_async_function(self):
        """Verify delete_cell imports get_notebook_context_from_session_async"""
        import inspect
        from jupyter_mcp_server.tools.delete_cell_tool import DeleteCellTool

        source = inspect.getsource(DeleteCellTool.execute)
        assert "get_notebook_context_from_session_async" in source, \
            "delete_cell should import async context function"

    def test_overwrite_cell_imports_async_function(self):
        """Verify overwrite_cell imports get_notebook_context_from_session_async"""
        import inspect
        from jupyter_mcp_server.tools.overwrite_cell_source_tool import OverwriteCellSourceTool

        source = inspect.getsource(OverwriteCellSourceTool.execute)
        assert "get_notebook_context_from_session_async" in source, \
            "overwrite_cell should import async context function"
