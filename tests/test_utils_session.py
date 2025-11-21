"""
Unit tests for utils.py session functions (ARK-165)

get_notebook_context_from_session() 함수의 모든 기능을 테스트합니다.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from jupyter_mcp_server.utils import get_notebook_context_from_session, get_current_notebook_context


class TestGetNotebookContextFromSession:
    """get_notebook_context_from_session() 함수 테스트"""

    def test_with_session_id_from_session_store(self):
        """session_id 있을 때 Session Store에서 조회"""
        from jupyter_mcp_server.server import session_store

        # Setup session
        session_store.update_notebook(
            session_id="session-1",
            notebook_name="nb1",
            notebook_path="path/nb1.ipynb",
            kernel_id="kernel-1"
        )

        # Test
        path, kernel = get_notebook_context_from_session(session_id="session-1")

        assert path == "path/nb1.ipynb"
        assert kernel == "kernel-1"

        # Cleanup
        session_store._sessions.clear()

    def test_without_session_id_uses_config(self, monkeypatch):
        """session_id 없을 때 config fallback"""
        mock_config = Mock()
        mock_config.document_id = "default.ipynb"
        mock_config.runtime_id = "default-kernel"
        monkeypatch.setattr('jupyter_mcp_server.config.get_config', lambda: mock_config)

        path, kernel = get_notebook_context_from_session(session_id=None)

        assert path == "default.ipynb"
        assert kernel == "default-kernel"

    def test_with_invalid_session_id_uses_config(self, monkeypatch):
        """session_id 있지만 세션이 없을 때 config fallback"""
        mock_config = Mock()
        mock_config.document_id = "default.ipynb"
        mock_config.runtime_id = "default-kernel"
        monkeypatch.setattr('jupyter_mcp_server.config.get_config', lambda: mock_config)

        path, kernel = get_notebook_context_from_session(session_id="non-existent")

        assert path == "default.ipynb"
        assert kernel == "default-kernel"

    def test_warning_log_without_session_id(self, monkeypatch, caplog):
        """session_id 없을 때 warning 로그 출력"""
        mock_config = Mock()
        mock_config.document_id = "default.ipynb"
        mock_config.runtime_id = "default-kernel"
        monkeypatch.setattr('jupyter_mcp_server.config.get_config', lambda: mock_config)

        get_notebook_context_from_session(session_id=None)

        assert "No session_id specified" in caplog.text

    def test_no_warning_log_with_session_id(self, caplog):
        """session_id 있을 때 warning 로그 없음"""
        from jupyter_mcp_server.server import session_store

        # Setup session
        session_store.update_notebook(
            session_id="session-2",
            notebook_name="nb2",
            notebook_path="path/nb2.ipynb",
            kernel_id="kernel-2"
        )

        # Clear any existing logs
        caplog.clear()

        get_notebook_context_from_session(session_id="session-2")

        # Warning 로그가 없어야 함
        assert "No session_id specified" not in caplog.text

        # Cleanup
        session_store._sessions.clear()

    def test_session_incomplete_data_uses_config(self, monkeypatch):
        """세션에 notebook_path나 kernel_id가 없을 때 config fallback"""
        from jupyter_mcp_server.server import session_store

        # 불완전한 세션 생성 (kernel_id만 있음)
        ctx = session_store.get_or_create("incomplete-session")
        ctx.kernel_id = "kernel-123"
        # notebook_path는 None

        mock_config = Mock()
        mock_config.document_id = "default.ipynb"
        mock_config.runtime_id = "default-kernel"
        monkeypatch.setattr('jupyter_mcp_server.config.get_config', lambda: mock_config)

        path, kernel = get_notebook_context_from_session(session_id="incomplete-session")

        # config fallback 사용
        assert path == "default.ipynb"
        assert kernel == "default-kernel"

        # Cleanup
        session_store._sessions.clear()

    def test_returns_tuple(self):
        """반환 타입이 Tuple인지 테스트"""
        from jupyter_mcp_server.server import session_store

        session_store.update_notebook(
            session_id="session-3",
            notebook_name="nb3",
            notebook_path="path/nb3.ipynb",
            kernel_id="kernel-3"
        )

        result = get_notebook_context_from_session(session_id="session-3")

        assert isinstance(result, tuple)
        assert len(result) == 2

        # Cleanup
        session_store._sessions.clear()


class TestGetCurrentNotebookContext:
    """get_current_notebook_context() deprecated 함수 테스트"""

    def test_deprecated_function_warns(self, monkeypatch, caplog):
        """Deprecated 함수 호출 시 warning"""
        mock_config = Mock()
        mock_config.document_id = "default.ipynb"
        mock_config.runtime_id = "default-kernel"
        monkeypatch.setattr('jupyter_mcp_server.config.get_config', lambda: mock_config)

        get_current_notebook_context()

        assert "DEPRECATED" in caplog.text
        assert "get_current_notebook_context" in caplog.text

    def test_deprecated_function_calls_new_function(self, monkeypatch):
        """Deprecated 함수가 새 함수를 호출하는지 테스트"""
        mock_config = Mock()
        mock_config.document_id = "default.ipynb"
        mock_config.runtime_id = "default-kernel"
        monkeypatch.setattr('jupyter_mcp_server.config.get_config', lambda: mock_config)

        # 기존 함수 호출
        old_result = get_current_notebook_context()

        # 새 함수 호출 (session_id=None)
        new_result = get_notebook_context_from_session(session_id=None)

        # 결과가 동일해야 함
        assert old_result == new_result

    def test_backward_compatibility(self, monkeypatch):
        """기존 코드가 수정 없이 작동하는지 테스트"""
        mock_config = Mock()
        mock_config.document_id = "compat.ipynb"
        mock_config.runtime_id = "compat-kernel"
        monkeypatch.setattr('jupyter_mcp_server.config.get_config', lambda: mock_config)

        # 기존 방식: notebook_manager 없이 호출
        path, kernel = get_current_notebook_context()

        # 여전히 작동해야 함
        assert path == "compat.ipynb"
        assert kernel == "compat-kernel"



class TestGetNotebookContextFromSessionAsync:
    """get_notebook_context_from_session_async() 함수 테스트 (ARK-165)
    
    SessionManager 통합 및 커널 자동 힐링 기능을 테스트합니다.
    """

    @pytest.mark.asyncio
    async def test_async_get_context_with_auto_heal_jupyter(self, monkeypatch):
        """JUPYTER_SERVER 모드 - 커널 힐링 성공"""
        from jupyter_mcp_server.utils import get_notebook_context_from_session_async
        from jupyter_mcp_server.server import session_store
        from jupyter_mcp_server.tools._base import ServerMode
        
        # Setup: 죽은 커널을 가진 세션 생성
        session_store.update_notebook(
            session_id="client-A",
            notebook_name="test",
            notebook_path="/work/test.ipynb",
            kernel_id="dead-kernel-123"
        )
        
        # Mock kernel_manager
        mock_km = Mock()
        mock_km.__contains__ = Mock(side_effect=lambda kid: kid == "new-kernel-456")
        mock_km.start_kernel = AsyncMock(return_value={'id': 'new-kernel-456'})
        
        # Test: auto_heal=True (기본값)
        path, kernel = await get_notebook_context_from_session_async(
            session_id="client-A",
            auto_heal=True,
            kernel_manager=mock_km,
            mode=ServerMode.JUPYTER_SERVER
        )
        
        # Assertions
        assert path == "/work/test.ipynb"
        assert kernel == "new-kernel-456"
        
        # 커널이 생성되었는지 확인
        mock_km.start_kernel.assert_called_once()
        
        # SessionStore가 업데이트되었는지 확인
        ctx = session_store.get("client-A")
        assert ctx.kernel_id == "new-kernel-456"
        
        # Cleanup
        session_store._sessions.clear()

    @pytest.mark.asyncio
    async def test_async_get_context_with_auto_heal_mcp(self, monkeypatch):
        """MCP_SERVER 모드 - 커널 힐링 성공"""
        from jupyter_mcp_server.utils import get_notebook_context_from_session_async
        from jupyter_mcp_server.server import session_store
        from jupyter_mcp_server.tools._base import ServerMode
        import sys
        
        # Setup: 죽은 커널을 가진 세션 생성
        session_store.update_notebook(
            session_id="client-B",
            notebook_name="test",
            notebook_path="/work/test.ipynb",
            kernel_id="dead-kernel-789"
        )
        
        # Mock server_client
        mock_kernel = Mock(id="dead-kernel-789")
        mock_new_kernel = Mock(id="new-kernel-abc")
        mock_server_client = Mock()
        mock_server_client.base_url = "http://localhost:8888"
        mock_server_client.token = "test-token"
        mock_server_client.kernels.list_kernels = Mock(return_value=[])
        
        # Mock jupyter_kernel_client
        mock_kernel_client_instance = Mock()
        mock_kernel_client_instance.id = "new-kernel-abc"
        mock_kernel_client_instance.start = Mock()
        
        mock_kernel_client_class = Mock(return_value=mock_kernel_client_instance)
        
        # Create fake module in sys.modules
        fake_module = type(sys)('jupyter_kernel_client')
        fake_module.KernelClient = mock_kernel_client_class
        monkeypatch.setitem(sys.modules, 'jupyter_kernel_client', fake_module)
        
        # Test: auto_heal=True
        path, kernel = await get_notebook_context_from_session_async(
            session_id="client-B",
            auto_heal=True,
            server_client=mock_server_client,
            mode=ServerMode.MCP_SERVER
        )
        
        # Assertions
        assert path == "/work/test.ipynb"
        assert kernel == "new-kernel-abc"
        
        # KernelClient가 생성되고 start 호출되었는지 확인
        mock_kernel_client_class.assert_called_once_with(
            server_url="http://localhost:8888",
            token="test-token"
        )
        mock_kernel_client_instance.start.assert_called_once_with(path="/work/test.ipynb")
        
        # SessionStore가 업데이트되었는지 확인
        ctx = session_store.get("client-B")
        assert ctx.kernel_id == "new-kernel-abc"
        
        # Cleanup
        session_store._sessions.clear()

    @pytest.mark.asyncio
    async def test_async_get_context_without_auto_heal(self):
        """auto_heal=False - 커널 힐링 비활성화"""
        from jupyter_mcp_server.utils import get_notebook_context_from_session_async
        from jupyter_mcp_server.server import session_store
        
        # Setup: 유효한 세션 생성
        session_store.update_notebook(
            session_id="client-C",
            notebook_name="test",
            notebook_path="/work/test.ipynb",
            kernel_id="kernel-999"
        )
        
        # Test: auto_heal=False (커널 체크 안 함)
        path, kernel = await get_notebook_context_from_session_async(
            session_id="client-C",
            auto_heal=False
        )
        
        # Assertions - SessionStore 값 그대로 반환
        assert path == "/work/test.ipynb"
        assert kernel == "kernel-999"
        
        # Cleanup
        session_store._sessions.clear()

    @pytest.mark.asyncio
    async def test_async_get_context_kernel_already_valid(self, monkeypatch):
        """커널이 유효할 때 힐링 스킵"""
        from jupyter_mcp_server.utils import get_notebook_context_from_session_async
        from jupyter_mcp_server.server import session_store
        from jupyter_mcp_server.tools._base import ServerMode
        
        # Setup: 유효한 커널을 가진 세션 생성
        session_store.update_notebook(
            session_id="client-D",
            notebook_name="test",
            notebook_path="/work/test.ipynb",
            kernel_id="valid-kernel-111"
        )
        
        # Mock kernel_manager - 커널이 존재함
        mock_km = Mock()
        mock_km.__contains__ = Mock(return_value=True)
        mock_km.start_kernel = Mock()  # 호출되면 안 됨
        
        # Test: 커널이 유효하므로 힐링 안 함
        path, kernel = await get_notebook_context_from_session_async(
            session_id="client-D",
            auto_heal=True,
            kernel_manager=mock_km,
            mode=ServerMode.JUPYTER_SERVER
        )
        
        # Assertions
        assert path == "/work/test.ipynb"
        assert kernel == "valid-kernel-111"
        
        # start_kernel이 호출되지 않았는지 확인
        mock_km.start_kernel.assert_not_called()
        
        # Cleanup
        session_store._sessions.clear()

    @pytest.mark.asyncio
    async def test_async_get_context_healing_failure(self, monkeypatch):
        """커널 힐링 실패 시 세션 notebook_path 유지, kernel만 config fallback"""
        from jupyter_mcp_server.utils import get_notebook_context_from_session_async
        from jupyter_mcp_server.server import session_store
        from jupyter_mcp_server.tools._base import ServerMode

        # Setup: 죽은 커널을 가진 세션 생성
        session_store.update_notebook(
            session_id="client-E",
            notebook_name="test",
            notebook_path="/work/test.ipynb",
            kernel_id="dead-kernel-222"
        )

        # Mock kernel_manager - 커널 생성 실패
        mock_km = Mock()
        mock_km.__contains__ = Mock(return_value=False)
        mock_km.start_kernel = AsyncMock(side_effect=Exception("Kernel creation failed"))

        # Mock config
        mock_config = Mock()
        mock_config.document_id = "default.ipynb"
        mock_config.runtime_id = "default-kernel"
        monkeypatch.setattr('jupyter_mcp_server.config.get_config', lambda: mock_config)

        # Test: 힐링 실패 시 세션 notebook_path 유지, kernel만 config fallback
        path, kernel = await get_notebook_context_from_session_async(
            session_id="client-E",
            auto_heal=True,
            kernel_manager=mock_km,
            mode=ServerMode.JUPYTER_SERVER
        )

        # Assertions - 세션의 notebook_path 유지, config의 kernel_id만 fallback
        assert path == "/work/test.ipynb"  # 세션 유지!
        assert kernel == "default-kernel"  # config fallback

        # Cleanup
        session_store._sessions.clear()

    @pytest.mark.asyncio
    async def test_async_get_context_config_fallback(self, monkeypatch):
        """session_id 없을 때 config fallback"""
        from jupyter_mcp_server.utils import get_notebook_context_from_session_async
        
        # Mock config
        mock_config = Mock()
        mock_config.document_id = "fallback.ipynb"
        mock_config.runtime_id = "fallback-kernel"
        monkeypatch.setattr('jupyter_mcp_server.config.get_config', lambda: mock_config)
        
        # Test: session_id=None
        path, kernel = await get_notebook_context_from_session_async(
            session_id=None,
            auto_heal=True
        )
        
        # Assertions - config fallback 사용
        assert path == "fallback.ipynb"
        assert kernel == "fallback-kernel"
