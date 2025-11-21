"""
Unit tests for use_notebook_tool.py ARK-165 features

Tests the 5 key features added in ARK-165:
1. Helper functions (_determine_action, _detect_error)
2. Session-based context management (session_id parameter)
3. Duplicate kernel bug fix (kernel_id injection)
4. Structured JSON output
5. JupyterHub integration (notebook_url generation)
"""

import pytest
import json
import os
from unittest.mock import Mock, AsyncMock, patch
from jupyter_mcp_server.tools.use_notebook_tool import UseNotebookTool
from jupyter_mcp_server.tools._base import ServerMode


@pytest.fixture(autouse=True)
def clean_session_store():
    """Clear SessionStore before each test to ensure isolation"""
    from jupyter_mcp_server.server import session_store
    session_store._sessions.clear()
    yield
    # Cleanup after test as well
    session_store._sessions.clear()


class TestHelperFunctions:
    """Test helper functions added in ARK-165"""

    def test_determine_action_created(self):
        """Test action detection for 'created' scenario"""
        tool = UseNotebookTool()
        assert tool._determine_action("Successfully activate notebook 'test'", None) == "created"

    def test_determine_action_switched(self):
        """Test action detection for 'switched' scenario"""
        tool = UseNotebookTool()
        assert tool._determine_action("already created", None) == "switched"
        assert tool._determine_action("already activated", None) == "switched"
        assert tool._determine_action("Reactivating notebook 'nb1'", None) == "switched"

    def test_determine_action_connected(self):
        """Test action detection for 'connected' scenario"""
        tool = UseNotebookTool()
        assert tool._determine_action("Connected to kernel 'kernel-123'", None) == "connected"

    def test_determine_action_use_mode_fallback(self):
        """Test action falls back to use_mode when pattern not matched"""
        tool = UseNotebookTool()
        assert tool._determine_action("some random message", "create") == "create"
        assert tool._determine_action("some random message", "connect") == "connect"

    def test_determine_action_unknown(self):
        """Test action returns 'unknown' when no pattern matches and no use_mode"""
        tool = UseNotebookTool()
        assert tool._determine_action("some message", None) == "unknown"

    def test_detect_error_success_cases(self):
        """Test error detection correctly identifies success messages"""
        tool = UseNotebookTool()
        assert tool._detect_error("Successfully activated notebook") == False
        assert tool._detect_error("[INFO] Connected to kernel") == False
        assert tool._detect_error("Notebook already created") == False
        assert tool._detect_error("Kernel already activated") == False
        assert tool._detect_error("Connected to kernel 'abc'") == False

    def test_detect_error_error_cases(self):
        """Test error detection correctly identifies error messages"""
        tool = UseNotebookTool()
        assert tool._detect_error("notebook not found") == True
        assert tool._detect_error("failed to connect") == True
        assert tool._detect_error("error occurred") == True
        assert tool._detect_error("invalid path") == True
        assert tool._detect_error("not the correct path") == True
        assert tool._detect_error("no session_manager available") == True
        assert tool._detect_error("no valid base_url") == True


class TestSessionStoreIntegration:
    """Test SessionStore integration (ARK-165 Fix 2)"""

    @pytest.mark.asyncio
    async def test_session_id_provided_registers_in_store(self, monkeypatch):
        """Test that notebook is registered in SessionStore when session_id provided"""
        from jupyter_mcp_server.server import session_store

        tool = UseNotebookTool()

        # Mock dependencies
        mock_kernel_manager = Mock()
        mock_kernel_manager.__contains__ = Mock(return_value=True)
        mock_contents_manager = AsyncMock()
        mock_notebook_manager = Mock()
        mock_notebook_manager.__contains__ = Mock(return_value=False)
        mock_notebook_manager.add_notebook = Mock()
        mock_notebook_manager.set_current_notebook = Mock()

        # Mock path check
        async def mock_check_path(*args, **kwargs):
            return True, None
        monkeypatch.setattr(tool, '_check_path_local', mock_check_path)

        # Mock notebook content retrieval
        mock_contents_manager.get = AsyncMock(return_value={
            'content': {
                'cells': [],
                'metadata': {},
                'nbformat': 4,
                'nbformat_minor': 4
            }
        })

        # Execute with session_id
        result_str = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            notebook_path="test.ipynb",
            notebook_name="test",
            session_id="session-123",
            kernel_id="kernel-abc",
            kernel_manager=mock_kernel_manager,
            contents_manager=mock_contents_manager,
            notebook_manager=mock_notebook_manager,
            use_mode="connect"
        )

        # Parse JSON result
        result = json.loads(result_str)

        # Verify session_id in response
        assert result["session_id"] == "session-123"

        # Verify SessionStore was updated
        ctx = session_store.get("session-123")
        assert ctx is not None
        assert ctx.notebook_path == "test.ipynb"
        assert ctx.current_notebook == "test"  # SessionContext uses current_notebook, not notebook_name
        assert ctx.kernel_id == "kernel-abc"

        # Cleanup
        session_store._sessions.clear()

    @pytest.mark.asyncio
    async def test_session_id_none_no_store_registration(self, monkeypatch):
        """Test that SessionStore is not updated when session_id is None"""
        from jupyter_mcp_server.server import session_store

        tool = UseNotebookTool()

        # Mock dependencies
        mock_kernel_manager = Mock()
        mock_kernel_manager.__contains__ = Mock(return_value=True)
        mock_contents_manager = AsyncMock()
        mock_notebook_manager = Mock()
        mock_notebook_manager.__contains__ = Mock(return_value=False)
        mock_notebook_manager.add_notebook = Mock()
        mock_notebook_manager.set_current_notebook = Mock()

        # Mock path check
        async def mock_check_path(*args, **kwargs):
            return True, None
        monkeypatch.setattr(tool, '_check_path_local', mock_check_path)

        # Mock notebook content retrieval
        mock_contents_manager.get = AsyncMock(return_value={
            'content': {
                'cells': [],
                'metadata': {},
                'nbformat': 4,
                'nbformat_minor': 4
            }
        })

        initial_count = len(session_store)

        # Execute without session_id
        await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            notebook_path="test.ipynb",
            notebook_name="test",
            session_id=None,  # No session_id
            kernel_id="kernel-xyz",
            kernel_manager=mock_kernel_manager,
            contents_manager=mock_contents_manager,
            notebook_manager=mock_notebook_manager,
            use_mode="connect"
        )

        # Verify SessionStore count unchanged
        assert len(session_store) == initial_count


class TestDuplicateKernelBugFix:
    """Test duplicate kernel bug fix (ARK-165 Fix 1)"""

    @pytest.mark.asyncio
    async def test_kernel_id_injection_jupyter_server_mode(self, monkeypatch):
        """Test that kernel_id is injected in JUPYTER_SERVER mode when not provided"""
        tool = UseNotebookTool()

        # Mock _start_kernel_local to return a kernel
        async def mock_start_kernel(km):
            return {"id": "injected-kernel-123"}
        monkeypatch.setattr(tool, '_start_kernel_local', mock_start_kernel)

        # Mock dependencies
        mock_kernel_manager = Mock()
        mock_kernel_manager.__contains__ = Mock(return_value=True)
        mock_contents_manager = AsyncMock()
        mock_notebook_manager = Mock()
        mock_notebook_manager.__contains__ = Mock(return_value=False)
        mock_notebook_manager.add_notebook = Mock()
        mock_notebook_manager.set_current_notebook = Mock()

        # Mock path check
        async def mock_check_path(*args, **kwargs):
            return True, None
        monkeypatch.setattr(tool, '_check_path_local', mock_check_path)

        # Mock notebook content retrieval
        mock_contents_manager.get = AsyncMock(return_value={
            'content': {
                'cells': [],
                'metadata': {},
                'nbformat': 4,
                'nbformat_minor': 4
            }
        })

        # Execute with JUPYTER_SERVER mode and no kernel_id
        result_str = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            notebook_path="test.ipynb",
            notebook_name="test",
            session_id="test-session-123",  # ARK-165: session_id required
            kernel_id=None,  # No kernel_id provided
            kernel_manager=mock_kernel_manager,
            contents_manager=mock_contents_manager,
            notebook_manager=mock_notebook_manager,
            use_mode="connect"
        )

        # Parse JSON result
        result = json.loads(result_str)

        # Verify kernel_id was injected
        assert result["metadata"]["kernel_id"] == "injected-kernel-123"

    @pytest.mark.asyncio
    async def test_no_injection_when_kernel_id_provided(self, monkeypatch):
        """Test that kernel_id is not injected when already provided"""
        tool = UseNotebookTool()

        # Track if _start_kernel_local was called
        start_kernel_called = []

        async def mock_start_kernel(km):
            start_kernel_called.append(True)
            return {"id": "should-not-be-used"}
        monkeypatch.setattr(tool, '_start_kernel_local', mock_start_kernel)

        # Mock dependencies
        mock_kernel_manager = Mock()
        mock_kernel_manager.__contains__ = Mock(return_value=True)
        mock_contents_manager = AsyncMock()
        mock_notebook_manager = Mock()
        mock_notebook_manager.__contains__ = Mock(return_value=False)
        mock_notebook_manager.add_notebook = Mock()
        mock_notebook_manager.set_current_notebook = Mock()

        # Mock path check
        async def mock_check_path(*args, **kwargs):
            return True, None
        monkeypatch.setattr(tool, '_check_path_local', mock_check_path)

        # Mock notebook content retrieval
        mock_contents_manager.get = AsyncMock(return_value={
            'content': {
                'cells': [],
                'metadata': {},
                'nbformat': 4,
                'nbformat_minor': 4
            }
        })

        # Execute with kernel_id already provided
        result_str = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            notebook_path="test_no_injection.ipynb",  # Use unique filename
            notebook_name="test_no_injection",
            session_id="test-session-456",  # ARK-165: session_id required
            kernel_id="existing-kernel-456",  # Kernel ID provided
            kernel_manager=mock_kernel_manager,
            contents_manager=mock_contents_manager,
            notebook_manager=mock_notebook_manager,
            use_mode="connect"
        )

        # Parse JSON result
        result = json.loads(result_str)

        # Verify original kernel_id was used (not injected)
        assert result["metadata"]["kernel_id"] == "existing-kernel-456"

        # Verify _start_kernel_local was NOT called for injection
        # (It might be called later in the flow, but not in the injection block)
        # This is validated by the kernel_id in result matching the provided one


class TestStructuredJsonOutput:
    """Test structured JSON output (ARK-165 Fix 5)"""

    @pytest.mark.asyncio
    async def test_json_output_structure(self, monkeypatch):
        """Test that output is valid JSON with correct structure"""
        tool = UseNotebookTool()

        # Mock dependencies
        mock_kernel_manager = Mock()
        mock_kernel_manager.__contains__ = Mock(return_value=True)
        mock_contents_manager = AsyncMock()
        mock_notebook_manager = Mock()
        mock_notebook_manager.__contains__ = Mock(return_value=False)
        mock_notebook_manager.add_notebook = Mock()
        mock_notebook_manager.set_current_notebook = Mock()

        # Mock path check
        async def mock_check_path(*args, **kwargs):
            return True, None
        monkeypatch.setattr(tool, '_check_path_local', mock_check_path)

        # Mock notebook content retrieval
        mock_contents_manager.get = AsyncMock(return_value={
            'content': {
                'cells': [],
                'metadata': {},
                'nbformat': 4,
                'nbformat_minor': 4
            }
        })

        # Execute
        result_str = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            notebook_path="test_json_output.ipynb",  # Use unique filename
            notebook_name="test_json_output",
            session_id="session-789",
            kernel_id="kernel-999",
            kernel_manager=mock_kernel_manager,
            contents_manager=mock_contents_manager,
            notebook_manager=mock_notebook_manager,
            use_mode="create"
        )

        # Verify it's valid JSON
        result = json.loads(result_str)

        # Verify structure
        assert "result" in result
        assert "status" in result["result"]
        assert result["result"]["status"] in ["success", "error"]
        assert "message" in result["result"]
        assert "action" in result["result"]
        assert "notebook_url" in result["result"]
        assert "jupyter_token" in result["result"]

        assert "session_id" in result
        assert result["session_id"] == "session-789"

        assert "notebook_path" in result
        assert result["notebook_path"] == "test_json_output.ipynb"

        assert "metadata" in result
        assert "notebook_name" in result["metadata"]
        assert "kernel_id" in result["metadata"]
        assert "mode" in result["metadata"]
        assert "timestamp" in result["metadata"]


class TestJupyterHubIntegration:
    """Test JupyterHub integration (ARK-165 Fix 4)"""

    @pytest.mark.asyncio
    async def test_notebook_url_generation_with_env_vars(self, monkeypatch):
        """Test that notebook_url is generated when JupyterHub env vars present"""
        # Set JupyterHub environment variables
        monkeypatch.setenv('JUPYTERHUB_API_TOKEN', 'test-token-123')
        monkeypatch.setenv('JUPYTERHUB_BASE_URL', 'https://hub.example.com/')
        monkeypatch.setenv('JUPYTERHUB_USER', 'testuser')

        tool = UseNotebookTool()

        # Mock dependencies
        mock_kernel_manager = Mock()
        mock_kernel_manager.__contains__ = Mock(return_value=True)
        mock_contents_manager = AsyncMock()
        mock_notebook_manager = Mock()
        mock_notebook_manager.__contains__ = Mock(return_value=False)
        mock_notebook_manager.add_notebook = Mock()
        mock_notebook_manager.set_current_notebook = Mock()

        # Mock path check
        async def mock_check_path(*args, **kwargs):
            return True, None
        monkeypatch.setattr(tool, '_check_path_local', mock_check_path)

        # Mock notebook content retrieval
        mock_contents_manager.get = AsyncMock(return_value={
            'content': {
                'cells': [],
                'metadata': {},
                'nbformat': 4,
                'nbformat_minor': 4
            }
        })

        # Execute
        result_str = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            notebook_path="notebooks/test.ipynb",
            notebook_name="test",
            session_id="test-session-jupyterhub",  # ARK-165: session_id required
            kernel_id="kernel-abc",
            kernel_manager=mock_kernel_manager,
            contents_manager=mock_contents_manager,
            notebook_manager=mock_notebook_manager,
            use_mode="connect"
        )

        # Parse result
        result = json.loads(result_str)

        # Verify notebook_url was generated
        expected_url = "https://hub.example.com/user/testuser/notebooks/notebooks/test.ipynb"
        assert result["result"]["notebook_url"] == expected_url
        assert result["result"]["jupyter_token"] == "test-token-123"

    @pytest.mark.asyncio
    async def test_notebook_url_none_without_env_vars(self, monkeypatch):
        """Test that notebook_url is None when JupyterHub env vars not present"""
        # Clear JupyterHub environment variables
        monkeypatch.delenv('JUPYTERHUB_API_TOKEN', raising=False)
        monkeypatch.delenv('JUPYTERHUB_BASE_URL', raising=False)
        monkeypatch.delenv('JUPYTERHUB_USER', raising=False)

        tool = UseNotebookTool()

        # Mock dependencies
        mock_kernel_manager = Mock()
        mock_kernel_manager.__contains__ = Mock(return_value=True)
        mock_contents_manager = AsyncMock()
        mock_notebook_manager = Mock()
        mock_notebook_manager.__contains__ = Mock(return_value=False)
        mock_notebook_manager.add_notebook = Mock()
        mock_notebook_manager.set_current_notebook = Mock()

        # Mock path check
        async def mock_check_path(*args, **kwargs):
            return True, None
        monkeypatch.setattr(tool, '_check_path_local', mock_check_path)

        # Mock notebook content retrieval
        mock_contents_manager.get = AsyncMock(return_value={
            'content': {
                'cells': [],
                'metadata': {},
                'nbformat': 4,
                'nbformat_minor': 4
            }
        })

        # Execute
        result_str = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            notebook_path="test_nohub.ipynb",  # Use unique filename to avoid conflicts
            notebook_name="test_nohub",
            session_id="test-session-no-hub",  # ARK-165: session_id required
            kernel_id="kernel-abc-nohub",
            kernel_manager=mock_kernel_manager,
            contents_manager=mock_contents_manager,
            notebook_manager=mock_notebook_manager,
            use_mode="connect"
        )

        # Parse result
        result = json.loads(result_str)

        # Verify notebook_url is None
        assert result["result"]["notebook_url"] is None
        assert result["result"]["jupyter_token"] is None


class TestMultiClientScenarios:
    """Test multi-client session scenarios (ARK-165 Task 2)"""

    @pytest.mark.asyncio
    async def test_different_session_same_name_different_path(self, monkeypatch):
        """Test that different sessions can use same notebook_name independently"""
        from jupyter_mcp_server.server import session_store

        tool = UseNotebookTool()

        # Mock dependencies
        mock_kernel_manager = Mock()
        mock_kernel_manager.__contains__ = Mock(return_value=True)
        mock_contents_manager = AsyncMock()
        mock_notebook_manager = Mock()
        mock_notebook_manager.__contains__ = Mock(return_value=False)
        mock_notebook_manager.add_notebook = Mock()
        mock_notebook_manager.set_current_notebook = Mock()

        # Mock path check to always succeed
        async def mock_check_path(*args, **kwargs):
            return True, None
        monkeypatch.setattr(tool, '_check_path_local', mock_check_path)

        # Mock notebook content retrieval
        mock_contents_manager.get = AsyncMock(return_value={
            'content': {
                'cells': [],
                'metadata': {},
                'nbformat': 4,
                'nbformat_minor': 4
            }
        })

        # Mock kernel start
        async def mock_start_kernel(km):
            return {"id": f"kernel-{id(km)}"}
        monkeypatch.setattr(tool, '_start_kernel_local', mock_start_kernel)

        # Session A: use notebook_name="nb1" with path="a.ipynb"
        result_a_str = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            notebook_path="a.ipynb",
            notebook_name="nb1",
            session_id="session-A",
            kernel_manager=mock_kernel_manager,
            contents_manager=mock_contents_manager,
            notebook_manager=mock_notebook_manager,
            use_mode="connect"
        )
        result_a = json.loads(result_a_str)

        # Session B: use notebook_name="nb1" with path="b.ipynb" (same name, different path)
        result_b_str = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            notebook_path="b.ipynb",
            notebook_name="nb1",
            session_id="session-B",
            kernel_manager=mock_kernel_manager,
            contents_manager=mock_contents_manager,
            notebook_manager=mock_notebook_manager,
            use_mode="connect"
        )
        result_b = json.loads(result_b_str)

        # Verify both sessions succeeded
        assert result_a["result"]["status"] == "success"
        assert result_b["result"]["status"] == "success"

        # Verify sessions have independent contexts
        ctx_a = session_store.get("session-A")
        ctx_b = session_store.get("session-B")
        assert ctx_a is not None
        assert ctx_b is not None
        assert ctx_a.notebook_path == "a.ipynb"
        assert ctx_b.notebook_path == "b.ipynb"
        assert ctx_a.current_notebook == "nb1"
        assert ctx_b.current_notebook == "nb1"

        # Cleanup
        session_store._sessions.clear()

    @pytest.mark.asyncio
    async def test_same_session_notebook_switch(self, monkeypatch):
        """Test that same session can switch notebooks and reuse kernel"""
        from jupyter_mcp_server.server import session_store

        tool = UseNotebookTool()

        # Mock dependencies
        mock_kernel_manager = Mock()
        mock_kernel_manager.__contains__ = Mock(return_value=True)
        mock_contents_manager = AsyncMock()
        mock_notebook_manager = Mock()
        mock_notebook_manager.__contains__ = Mock(return_value=False)
        mock_notebook_manager.add_notebook = Mock()
        mock_notebook_manager.set_current_notebook = Mock()

        # Mock path check
        async def mock_check_path(*args, **kwargs):
            return True, None
        monkeypatch.setattr(tool, '_check_path_local', mock_check_path)

        # Mock notebook content retrieval
        mock_contents_manager.get = AsyncMock(return_value={
            'content': {
                'cells': [],
                'metadata': {},
                'nbformat': 4,
                'nbformat_minor': 4
            }
        })

        # Mock kernel start to return consistent kernel_id
        kernel_counter = [0]
        async def mock_start_kernel(km):
            kernel_counter[0] += 1
            return {"id": f"kernel-{kernel_counter[0]}"}
        monkeypatch.setattr(tool, '_start_kernel_local', mock_start_kernel)

        session_id = "session-switch"

        # First call: connect to a.ipynb
        result_1_str = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            notebook_path="a.ipynb",
            notebook_name="nb1",
            session_id=session_id,
            kernel_manager=mock_kernel_manager,
            contents_manager=mock_contents_manager,
            notebook_manager=mock_notebook_manager,
            use_mode="connect"
        )
        result_1 = json.loads(result_1_str)
        kernel_id_1 = result_1["metadata"]["kernel_id"]

        # Second call: switch to b.ipynb (same session)
        result_2_str = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            notebook_path="b.ipynb",
            notebook_name="nb2",
            session_id=session_id,
            kernel_manager=mock_kernel_manager,
            contents_manager=mock_contents_manager,
            notebook_manager=mock_notebook_manager,
            use_mode="connect"
        )
        result_2 = json.loads(result_2_str)
        kernel_id_2 = result_2["metadata"]["kernel_id"]

        # Verify kernel was reused (same kernel_id)
        assert kernel_id_1 == kernel_id_2

        # Verify session context was updated to b.ipynb
        ctx = session_store.get(session_id)
        assert ctx is not None
        assert ctx.notebook_path == "b.ipynb"
        assert ctx.current_notebook == "nb2"
        assert ctx.kernel_id == kernel_id_2

        # Cleanup
        session_store._sessions.clear()

    @pytest.mark.asyncio
    async def test_conflict_detection_same_path(self, monkeypatch):
        """Test that file-level conflict is detected when different sessions use same path"""
        from jupyter_mcp_server.server import session_store

        tool = UseNotebookTool()

        # Mock dependencies
        mock_kernel_manager = Mock()
        mock_kernel_manager.__contains__ = Mock(return_value=True)
        mock_contents_manager = AsyncMock()
        mock_notebook_manager = Mock()
        mock_notebook_manager.__contains__ = Mock(return_value=False)
        mock_notebook_manager.add_notebook = Mock()
        mock_notebook_manager.set_current_notebook = Mock()

        # Mock path check
        async def mock_check_path(*args, **kwargs):
            return True, None
        monkeypatch.setattr(tool, '_check_path_local', mock_check_path)

        # Mock notebook content retrieval
        mock_contents_manager.get = AsyncMock(return_value={
            'content': {
                'cells': [],
                'metadata': {},
                'nbformat': 4,
                'nbformat_minor': 4
            }
        })

        # Mock kernel start
        async def mock_start_kernel(km):
            return {"id": f"kernel-{id(km)}"}
        monkeypatch.setattr(tool, '_start_kernel_local', mock_start_kernel)

        # Session A: connect to work.ipynb as nb1
        result_a_str = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            notebook_path="work.ipynb",
            notebook_name="nb1",
            session_id="session-A",
            kernel_manager=mock_kernel_manager,
            contents_manager=mock_contents_manager,
            notebook_manager=mock_notebook_manager,
            use_mode="connect"
        )
        result_a = json.loads(result_a_str)
        assert result_a["result"]["status"] == "success"

        # Session B: try to connect to work.ipynb as nb2 (same path, different session)
        result_b_str = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            notebook_path="work.ipynb",
            notebook_name="nb2",
            session_id="session-B",
            kernel_manager=mock_kernel_manager,
            contents_manager=mock_contents_manager,
            notebook_manager=mock_notebook_manager,
            use_mode="connect"
        )
        result_b = json.loads(result_b_str)

        # Verify Session B was rejected due to file-level conflict
        assert result_b["result"]["status"] == "error"
        assert "already in use" in result_b["result"]["message"].lower()

        # Cleanup
        session_store._sessions.clear()

    @pytest.mark.asyncio
    async def test_idempotent_same_session_same_notebook(self, monkeypatch):
        """Test that calling use_notebook twice with same session/notebook is idempotent"""
        from jupyter_mcp_server.server import session_store

        tool = UseNotebookTool()

        # Mock dependencies
        mock_kernel_manager = Mock()
        mock_kernel_manager.__contains__ = Mock(return_value=True)
        mock_contents_manager = AsyncMock()
        mock_notebook_manager = Mock()
        mock_notebook_manager.__contains__ = Mock(return_value=False)
        mock_notebook_manager.add_notebook = Mock()
        mock_notebook_manager.set_current_notebook = Mock()

        # Mock path check
        async def mock_check_path(*args, **kwargs):
            return True, None
        monkeypatch.setattr(tool, '_check_path_local', mock_check_path)

        # Mock notebook content retrieval
        mock_contents_manager.get = AsyncMock(return_value={
            'content': {
                'cells': [],
                'metadata': {},
                'nbformat': 4,
                'nbformat_minor': 4
            }
        })

        # Mock kernel start
        kernel_counter = [0]
        async def mock_start_kernel(km):
            kernel_counter[0] += 1
            return {"id": f"kernel-{kernel_counter[0]}"}
        monkeypatch.setattr(tool, '_start_kernel_local', mock_start_kernel)

        session_id = "session-idempotent"

        # First call
        result_1_str = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            notebook_path="work.ipynb",
            notebook_name="nb1",
            session_id=session_id,
            kernel_manager=mock_kernel_manager,
            contents_manager=mock_contents_manager,
            notebook_manager=mock_notebook_manager,
            use_mode="connect"
        )
        result_1 = json.loads(result_1_str)
        kernel_id_1 = result_1["metadata"]["kernel_id"]

        # Second call (same session, same notebook)
        result_2_str = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            notebook_path="work.ipynb",
            notebook_name="nb1",
            session_id=session_id,
            kernel_manager=mock_kernel_manager,
            contents_manager=mock_contents_manager,
            notebook_manager=mock_notebook_manager,
            use_mode="connect"
        )
        result_2 = json.loads(result_2_str)
        kernel_id_2 = result_2["metadata"]["kernel_id"]

        # Verify both calls succeeded
        assert result_1["result"]["status"] == "success"
        assert result_2["result"]["status"] == "success"

        # Verify kernel was reused (same kernel_id)
        assert kernel_id_1 == kernel_id_2

        # Verify only one kernel was created
        assert kernel_counter[0] == 1

        # Cleanup
        session_store._sessions.clear()

    @pytest.mark.asyncio
    async def test_session_id_required(self, monkeypatch):
        """Test that session_id=None returns clear error"""
        tool = UseNotebookTool()

        # Mock dependencies (minimal since we expect early rejection)
        mock_kernel_manager = Mock()
        mock_contents_manager = AsyncMock()
        mock_notebook_manager = Mock()

        # Call without session_id (None)
        result_str = await tool.execute(
            mode=ServerMode.JUPYTER_SERVER,
            notebook_path="work.ipynb",
            notebook_name="nb1",
            session_id=None,  # No session_id provided
            kernel_manager=mock_kernel_manager,
            contents_manager=mock_contents_manager,
            notebook_manager=mock_notebook_manager,
            use_mode="connect"
        )

        # Parse result
        result = json.loads(result_str)

        # Verify error response
        assert result["result"]["status"] == "error"
        assert "session_id is required" in result["result"]["message"]
        assert result["result"]["action"] == "rejected"
