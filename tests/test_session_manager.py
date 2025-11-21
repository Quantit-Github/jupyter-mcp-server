"""SessionManager unit tests (ARK-165 커널 힐링 리팩토링).

이 테스트는 SessionManager의 커널 헬스 체크 및 자동 힐링 기능을 검증합니다.

Test Coverage:
    - Kernel health check (JUPYTER_SERVER/MCP_SERVER 모드)
    - Kernel healing (성공/실패 시나리오)
    - SessionStore 부분 업데이트
    - Edge cases (세션 없음, invalid mode 등)

Test Structure:
    - test_jupyter_server_mode_*: JUPYTER_SERVER 모드 테스트
    - test_mcp_server_mode_*: MCP_SERVER 모드 테스트
    - test_heal_kernel_*: 커널 힐링 테스트
    - test_update_notebook_*: SessionStore 부분 업데이트 테스트

Notes:
    - 모든 테스트는 Mock 객체 사용 (외부 의존성 없음)
    - pytest-asyncio 사용 (async 함수 테스트)
    - ARK-165: 커널 힐링 중앙화 핵심 테스트
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from jupyter_mcp_server.session_manager import SessionManager
from jupyter_mcp_server.session_store import SessionStore, SessionContext
from jupyter_mcp_server.tools._base import ServerMode


@pytest.fixture
def session_store():
    """SessionStore 픽스처 (각 테스트마다 새로 생성)."""
    return SessionStore(ttl_hours=24)


@pytest.fixture
def session_manager(session_store):
    """SessionManager 픽스처."""
    return SessionManager(session_store)


class TestKernelHealthCheck:
    """커널 헬스 체크 테스트 (get_session_with_kernel_check)."""

    @pytest.mark.asyncio
    async def test_jupyter_server_mode_kernel_health_check_valid(
        self, session_manager, session_store
    ):
        """JUPYTER_SERVER 모드에서 유효한 커널 확인."""
        # Given: 세션과 유효한 커널 설정
        session_id = "test-session"
        kernel_id = "kernel-123"
        session_store.update_notebook(
            session_id=session_id,
            notebook_name="test",
            notebook_path="/test.ipynb",
            kernel_id=kernel_id
        )

        # Mock kernel_manager: kernel_id가 존재함
        mock_kernel_manager = Mock()
        mock_kernel_manager.__contains__ = Mock(return_value=True)

        # When: 커널 헬스 체크
        ctx, kernel_healthy = await session_manager.get_session_with_kernel_check(
            session_id=session_id,
            kernel_manager=mock_kernel_manager,
            mode=ServerMode.JUPYTER_SERVER
        )

        # Then: 커널이 healthy로 확인됨
        assert ctx is not None
        assert ctx.kernel_id == kernel_id
        assert kernel_healthy is True

    @pytest.mark.asyncio
    async def test_jupyter_server_mode_kernel_health_check_invalid(
        self, session_manager, session_store
    ):
        """JUPYTER_SERVER 모드에서 무효한 커널 감지."""
        # Given: 세션과 죽은 커널 설정
        session_id = "test-session"
        kernel_id = "dead-kernel"
        session_store.update_notebook(
            session_id=session_id,
            notebook_name="test",
            notebook_path="/test.ipynb",
            kernel_id=kernel_id
        )

        # Mock kernel_manager: kernel_id가 존재하지 않음
        mock_kernel_manager = Mock()
        mock_kernel_manager.__contains__ = Mock(return_value=False)

        # When: 커널 헬스 체크
        ctx, kernel_healthy = await session_manager.get_session_with_kernel_check(
            session_id=session_id,
            kernel_manager=mock_kernel_manager,
            mode=ServerMode.JUPYTER_SERVER
        )

        # Then: 커널이 unhealthy로 감지됨
        assert ctx is not None
        assert ctx.kernel_id == kernel_id
        assert kernel_healthy is False

    @pytest.mark.asyncio
    async def test_mcp_server_mode_kernel_health_check_valid(
        self, session_manager, session_store
    ):
        """MCP_SERVER 모드에서 유효한 커널 확인."""
        # Given: 세션과 유효한 커널 설정
        session_id = "test-session"
        kernel_id = "kernel-456"
        session_store.update_notebook(
            session_id=session_id,
            notebook_name="test",
            notebook_path="/test.ipynb",
            kernel_id=kernel_id
        )

        # Mock server_client: kernel_id가 존재함
        mock_kernel = Mock()
        mock_kernel.id = kernel_id
        mock_server_client = Mock()
        mock_server_client.kernels.list_kernels = Mock(return_value=[mock_kernel])

        # When: 커널 헬스 체크
        ctx, kernel_healthy = await session_manager.get_session_with_kernel_check(
            session_id=session_id,
            server_client=mock_server_client,
            mode=ServerMode.MCP_SERVER
        )

        # Then: 커널이 healthy로 확인됨
        assert ctx is not None
        assert ctx.kernel_id == kernel_id
        assert kernel_healthy is True

    @pytest.mark.asyncio
    async def test_mcp_server_mode_kernel_health_check_invalid(
        self, session_manager, session_store
    ):
        """MCP_SERVER 모드에서 무효한 커널 감지."""
        # Given: 세션과 죽은 커널 설정
        session_id = "test-session"
        kernel_id = "dead-kernel"
        session_store.update_notebook(
            session_id=session_id,
            notebook_name="test",
            notebook_path="/test.ipynb",
            kernel_id=kernel_id
        )

        # Mock server_client: kernel_id가 존재하지 않음 (다른 커널만 존재)
        mock_other_kernel = Mock()
        mock_other_kernel.id = "other-kernel"
        mock_server_client = Mock()
        mock_server_client.kernels.list_kernels = Mock(return_value=[mock_other_kernel])

        # When: 커널 헬스 체크
        ctx, kernel_healthy = await session_manager.get_session_with_kernel_check(
            session_id=session_id,
            server_client=mock_server_client,
            mode=ServerMode.MCP_SERVER
        )

        # Then: 커널이 unhealthy로 감지됨
        assert ctx is not None
        assert ctx.kernel_id == kernel_id
        assert kernel_healthy is False


class TestKernelHealing:
    """커널 힐링 테스트 (heal_kernel)."""

    @pytest.mark.asyncio
    async def test_heal_kernel_jupyter_server_mode(
        self, session_manager, session_store
    ):
        """JUPYTER_SERVER 모드 커널 힐링 성공."""
        # Given: 세션 설정
        session_id = "test-session"
        session_store.update_notebook(
            session_id=session_id,
            notebook_name="test",
            notebook_path="/test.ipynb",
            kernel_id="old-kernel"
        )

        # Mock kernel_manager: 새 커널 생성
        new_kernel_id = "new-kernel-123"
        mock_kernel_manager = AsyncMock()
        mock_kernel_manager.start_kernel = AsyncMock(return_value={"id": new_kernel_id})

        # When: 커널 힐링
        result_kernel_id = await session_manager.heal_kernel(
            session_id=session_id,
            kernel_manager=mock_kernel_manager,
            mode=ServerMode.JUPYTER_SERVER
        )

        # Then: 새 커널 ID 반환
        assert result_kernel_id == new_kernel_id

        # Then: SessionStore 업데이트 확인
        ctx = session_store.get(session_id)
        assert ctx.kernel_id == new_kernel_id
        assert ctx.current_notebook == "test"  # 다른 필드는 유지됨
        assert ctx.notebook_path == "/test.ipynb"  # 다른 필드는 유지됨

    @pytest.mark.asyncio
    async def test_heal_kernel_mcp_server_mode(
        self, session_manager, session_store, monkeypatch
    ):
        """MCP_SERVER 모드 커널 힐링 성공."""
        # Given: 세션 설정
        session_id = "test-session"
        notebook_path = "/test.ipynb"
        session_store.update_notebook(
            session_id=session_id,
            notebook_name="test",
            notebook_path=notebook_path,
            kernel_id="old-kernel"
        )

        # Mock KernelClient - sys.modules에 먼저 등록
        import sys
        new_kernel_id = "new-kernel-456"
        mock_kernel = Mock()
        mock_kernel.id = new_kernel_id
        mock_kernel.start = Mock()

        mock_kernel_client_class = Mock(return_value=mock_kernel)

        # Create fake jupyter_kernel_client module
        fake_module = type(sys)('jupyter_kernel_client')
        fake_module.KernelClient = mock_kernel_client_class
        monkeypatch.setitem(sys.modules, 'jupyter_kernel_client', fake_module)

        # Mock server_client
        mock_server_client = Mock()
        mock_server_client.base_url = "http://localhost:8888"
        mock_server_client.token = "test-token"

        # When: 커널 힐링
        result_kernel_id = await session_manager.heal_kernel(
            session_id=session_id,
            server_client=mock_server_client,
            mode=ServerMode.MCP_SERVER,
            notebook_path=notebook_path
        )

        # Then: 새 커널 ID 반환
        assert result_kernel_id == new_kernel_id

        # Then: KernelClient가 올바르게 생성됨
        mock_kernel_client_class.assert_called_once_with(
            server_url="http://localhost:8888",
            token="test-token"
        )
        mock_kernel.start.assert_called_once_with(path=notebook_path)

        # Then: SessionStore 업데이트 확인
        ctx = session_store.get(session_id)
        assert ctx.kernel_id == new_kernel_id

    @pytest.mark.asyncio
    async def test_heal_kernel_updates_session_store(
        self, session_manager, session_store
    ):
        """힐링 후 SessionStore 자동 업데이트 검증."""
        # Given: 세션 설정
        session_id = "test-session"
        old_kernel_id = "old-kernel"
        session_store.update_notebook(
            session_id=session_id,
            notebook_name="test",
            notebook_path="/test.ipynb",
            kernel_id=old_kernel_id
        )

        # Mock kernel_manager
        new_kernel_id = "new-kernel-789"
        mock_kernel_manager = AsyncMock()
        mock_kernel_manager.start_kernel = AsyncMock(return_value={"id": new_kernel_id})

        # When: 커널 힐링
        result_kernel_id = await session_manager.heal_kernel(
            session_id=session_id,
            kernel_manager=mock_kernel_manager,
            mode=ServerMode.JUPYTER_SERVER
        )

        # Then: SessionStore에서 새 kernel_id 확인
        ctx = session_store.get(session_id)
        assert ctx.kernel_id == new_kernel_id
        assert ctx.kernel_id != old_kernel_id

        # Then: 다른 필드들은 유지됨 (부분 업데이트)
        assert ctx.current_notebook == "test"
        assert ctx.notebook_path == "/test.ipynb"

    @pytest.mark.asyncio
    async def test_heal_kernel_creation_failure_jupyter(
        self, session_manager, session_store
    ):
        """JUPYTER_SERVER 모드 커널 생성 실패 시 None 반환."""
        # Given: 세션 설정
        session_id = "test-session"
        session_store.update_notebook(
            session_id=session_id,
            notebook_name="test",
            notebook_path="/test.ipynb",
            kernel_id="old-kernel"
        )

        # Mock kernel_manager: 커널 생성 실패
        mock_kernel_manager = AsyncMock()
        mock_kernel_manager.start_kernel = AsyncMock(
            side_effect=Exception("Kernel creation failed")
        )

        # When: 커널 힐링 시도
        result_kernel_id = await session_manager.heal_kernel(
            session_id=session_id,
            kernel_manager=mock_kernel_manager,
            mode=ServerMode.JUPYTER_SERVER
        )

        # Then: None 반환 (실패)
        assert result_kernel_id is None

        # Then: SessionStore는 변경되지 않음
        ctx = session_store.get(session_id)
        assert ctx.kernel_id == "old-kernel"

    @pytest.mark.asyncio
    async def test_heal_kernel_creation_failure_mcp(
        self, session_manager, session_store, monkeypatch
    ):
        """MCP_SERVER 모드 커널 생성 실패 시 None 반환."""
        # Given: 세션 설정
        session_id = "test-session"
        session_store.update_notebook(
            session_id=session_id,
            notebook_name="test",
            notebook_path="/test.ipynb",
            kernel_id="old-kernel"
        )

        # Mock KernelClient: 생성 실패 - sys.modules에 먼저 등록
        import sys
        mock_kernel_client_class = Mock(side_effect=Exception("Connection failed"))

        # Create fake jupyter_kernel_client module
        fake_module = type(sys)('jupyter_kernel_client')
        fake_module.KernelClient = mock_kernel_client_class
        monkeypatch.setitem(sys.modules, 'jupyter_kernel_client', fake_module)

        # Mock server_client
        mock_server_client = Mock()
        mock_server_client.base_url = "http://localhost:8888"
        mock_server_client.token = "test-token"

        # When: 커널 힐링 시도
        result_kernel_id = await session_manager.heal_kernel(
            session_id=session_id,
            server_client=mock_server_client,
            mode=ServerMode.MCP_SERVER,
            notebook_path="/test.ipynb"
        )

        # Then: None 반환 (실패)
        assert result_kernel_id is None

        # Then: SessionStore는 변경되지 않음
        ctx = session_store.get(session_id)
        assert ctx.kernel_id == "old-kernel"

    @pytest.mark.asyncio
    async def test_heal_kernel_invalid_mode(
        self, session_manager, session_store
    ):
        """무효한 mode일 때 None 반환."""
        # Given: 세션 설정
        session_id = "test-session"
        session_store.update_notebook(
            session_id=session_id,
            notebook_name="test",
            notebook_path="/test.ipynb",
            kernel_id="old-kernel"
        )

        # When: 무효한 mode로 커널 힐링 시도 (mode=None)
        result_kernel_id = await session_manager.heal_kernel(
            session_id=session_id,
            kernel_manager=None,
            server_client=None,
            mode=None
        )

        # Then: None 반환 (실패)
        assert result_kernel_id is None

        # Then: SessionStore는 변경되지 않음
        ctx = session_store.get(session_id)
        assert ctx.kernel_id == "old-kernel"

    @pytest.mark.asyncio
    async def test_session_not_found(
        self, session_manager, session_store
    ):
        """session_id가 없을 때 None 반환."""
        # Given: 존재하지 않는 세션 ID
        session_id = "non-existent-session"

        # Mock kernel_manager
        mock_kernel_manager = AsyncMock()

        # When: 힐링 시도
        result_kernel_id = await session_manager.heal_kernel(
            session_id=session_id,
            kernel_manager=mock_kernel_manager,
            mode=ServerMode.JUPYTER_SERVER
        )

        # Then: None 반환 (세션 없음)
        assert result_kernel_id is None


class TestSessionStorePartialUpdate:
    """SessionStore.update_notebook() 부분 업데이트 테스트."""

    def test_update_notebook_partial_kernel_id_only(self, session_store):
        """kernel_id만 업데이트, 나머지 필드 유지."""
        # Given: 초기 세션 생성
        session_id = "test-session"
        session_store.update_notebook(
            session_id=session_id,
            notebook_name="original",
            notebook_path="/original.ipynb",
            kernel_id="kernel-1"
        )

        # When: kernel_id만 업데이트
        session_store.update_notebook(
            session_id=session_id,
            kernel_id="kernel-2"
        )

        # Then: kernel_id만 변경, 다른 필드는 유지
        ctx = session_store.get(session_id)
        assert ctx.kernel_id == "kernel-2"
        assert ctx.current_notebook == "original"  # 유지됨
        assert ctx.notebook_path == "/original.ipynb"  # 유지됨

    def test_update_notebook_partial_path_only(self, session_store):
        """notebook_path만 업데이트."""
        # Given: 초기 세션 생성
        session_id = "test-session"
        session_store.update_notebook(
            session_id=session_id,
            notebook_name="test",
            notebook_path="/old.ipynb",
            kernel_id="kernel-123"
        )

        # When: notebook_path만 업데이트
        session_store.update_notebook(
            session_id=session_id,
            notebook_path="/new.ipynb"
        )

        # Then: notebook_path만 변경, 다른 필드는 유지
        ctx = session_store.get(session_id)
        assert ctx.notebook_path == "/new.ipynb"
        assert ctx.current_notebook == "test"  # 유지됨
        assert ctx.kernel_id == "kernel-123"  # 유지됨

    def test_update_notebook_full_update(self, session_store):
        """모든 필드 업데이트 (기존 동작)."""
        # Given: 초기 세션 생성
        session_id = "test-session"
        session_store.update_notebook(
            session_id=session_id,
            notebook_name="old",
            notebook_path="/old.ipynb",
            kernel_id="kernel-old"
        )

        # When: 모든 필드 업데이트
        session_store.update_notebook(
            session_id=session_id,
            notebook_name="new",
            notebook_path="/new.ipynb",
            kernel_id="kernel-new"
        )

        # Then: 모든 필드 변경됨
        ctx = session_store.get(session_id)
        assert ctx.current_notebook == "new"
        assert ctx.notebook_path == "/new.ipynb"
        assert ctx.kernel_id == "kernel-new"
