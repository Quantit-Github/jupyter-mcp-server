"""SessionManager: 커널 헬스 체크 및 자동 힐링 관리 (ARK-165).

SessionManager는 SessionStore를 래핑하고 커널 헬스 관리 기능을 제공합니다.
모든 cell operation tool이 자동으로 커널 힐링 혜택을 받을 수 있도록 중앙화된 관리 레이어를 제공합니다.

Architecture:
    Storage Layer (SessionStore) < Orchestration Layer (SessionManager)
    - SessionStore: 경량 저장소 (session_id → SessionContext 매핑)
    - SessionManager: 비즈니스 로직 (커널 헬스 체크, 자동 힐링)

Key Features:
    - 커널 헬스 체크: JUPYTER_SERVER/MCP_SERVER 모드 지원
    - 자동 힐링: 죽은 커널 감지 시 새 커널 자동 생성
    - SessionStore 자동 업데이트: 새 kernel_id로 동기화
    - 에러 핸들링: 힐링 실패 시 명확한 None 반환

Example:
    ```python
    from jupyter_mcp_server.session_manager import SessionManager
    from jupyter_mcp_server.server import session_store

    # SessionManager 초기화
    session_manager = SessionManager(session_store)

    # 커널 헬스 체크
    ctx, kernel_healthy = await session_manager.get_session_with_kernel_check(
        session_id="client-A",
        kernel_manager=kernel_manager,
        mode=ServerMode.JUPYTER_SERVER
    )

    # 커널 힐링 (필요 시)
    if ctx and not kernel_healthy:
        new_kernel_id = await session_manager.heal_kernel(
            session_id="client-A",
            kernel_manager=kernel_manager,
            mode=ServerMode.JUPYTER_SERVER,
            notebook_path=ctx.notebook_path
        )
    ```

Notes:
    - ARK-165: 커널 힐링 중앙화를 위한 핵심 컴포넌트
    - 관심사 분리: SessionStore(storage) vs SessionManager(orchestration)
    - 단방향 의존성: SessionManager → SessionStore (순환 의존성 없음)
"""

import logging
from typing import Optional, Tuple

from jupyter_mcp_server.session_store import SessionStore, SessionContext
from jupyter_mcp_server.tools._base import ServerMode

logger = logging.getLogger(__name__)


class SessionManager:
    """Session management with kernel health checking and auto-healing.

    SessionManager는 SessionStore를 래핑하고 커널 헬스 관리 기능을 추가합니다.
    모든 tool에서 공통으로 사용할 수 있는 중앙화된 커널 힐링 로직을 제공합니다.

    Attributes:
        session_store: SessionStore 인스턴스 (storage layer)

    Methods:
        get_session_with_kernel_check: 세션 조회 + 커널 헬스 체크
        heal_kernel: 죽은 커널 자동 복구

    Example:
        ```python
        # SessionManager 생성
        manager = SessionManager(session_store)

        # 커널 헬스 체크
        ctx, healthy = await manager.get_session_with_kernel_check(
            session_id, kernel_manager, server_client, mode
        )

        # 필요 시 힐링
        if not healthy:
            new_kernel_id = await manager.heal_kernel(
                session_id, kernel_manager, server_client, mode, notebook_path
            )
        ```

    Notes:
        - ARK-165: 커널 힐링 중앙화 핵심 클래스
        - SessionStore에 대한 단방향 의존성만 가짐
        - 모든 async 메서드 (kernel_manager/server_client 호출)
    """

    def __init__(self, session_store: SessionStore):
        """SessionManager 초기화.

        Args:
            session_store: SessionStore 인스턴스
        """
        self.session_store = session_store
        logger.info("SessionManager initialized")

    async def get_session_with_kernel_check(
        self,
        session_id: str,
        kernel_manager=None,
        server_client=None,
        mode: ServerMode = None
    ) -> Tuple[Optional[SessionContext], bool]:
        """세션 조회 + 커널 헬스 체크.

        SessionStore에서 세션을 조회하고, 커널이 실제로 존재하는지 확인합니다.

        Args:
            session_id: 클라이언트 세션 ID
            kernel_manager: JUPYTER_SERVER 모드용 kernel manager (optional)
            server_client: MCP_SERVER 모드용 server client (optional)
            mode: ServerMode (JUPYTER_SERVER or MCP_SERVER)

        Returns:
            (context, kernel_healthy): Tuple
                - context: SessionContext 또는 None (세션 없을 때)
                - kernel_healthy: bool (커널이 유효한지 여부)
                    - True: 커널 존재하고 유효
                    - False: 커널 없거나 죽음

        Example:
            ```python
            # JUPYTER_SERVER 모드
            ctx, healthy = await manager.get_session_with_kernel_check(
                session_id="A",
                kernel_manager=km,
                mode=ServerMode.JUPYTER_SERVER
            )

            # MCP_SERVER 모드
            ctx, healthy = await manager.get_session_with_kernel_check(
                session_id="B",
                server_client=client,
                mode=ServerMode.MCP_SERVER
            )
            ```

        Notes:
            - 세션이 없으면 (None, False) 반환
            - 세션은 있지만 kernel_id가 None이면 (ctx, False) 반환
            - 커널 체크는 가벼운 연산 (O(1) dict lookup 또는 캐싱된 HTTP)
        """
        # 1. SessionStore에서 세션 조회
        ctx = self.session_store.get(session_id)
        if not ctx:
            # 세션 없음 - (None, False) 반환
            logger.debug(f"Session not found: {session_id}")
            return None, False

        # 2. kernel_id가 없으면 커널 없음으로 판단
        if not ctx.kernel_id:
            logger.debug(f"Session {session_id[:8]}... has no kernel_id")
            return ctx, False

        # 3. 실제 커널 존재 여부 확인 (mode별 분기)
        kernel_healthy = False

        if mode == ServerMode.JUPYTER_SERVER and kernel_manager is not None:
            # JUPYTER_SERVER 모드: kernel_manager의 __contains__ 사용
            kernel_healthy = ctx.kernel_id in kernel_manager
            logger.debug(
                f"JUPYTER_SERVER kernel check: {ctx.kernel_id} -> {kernel_healthy}"
            )
        elif mode == ServerMode.MCP_SERVER and server_client is not None:
            # MCP_SERVER 모드: server_client.kernels.list_kernels() 사용
            try:
                kernels = server_client.kernels.list_kernels()
                kernel_healthy = any(k.id == ctx.kernel_id for k in kernels)
                logger.debug(
                    f"MCP_SERVER kernel check: {ctx.kernel_id} -> {kernel_healthy}"
                )
            except Exception as e:
                logger.warning(f"Failed to list kernels in MCP_SERVER mode: {e}")
                kernel_healthy = False
        else:
            # mode 또는 client가 없으면 체크 불가
            logger.warning(
                f"Cannot check kernel health: mode={mode}, "
                f"kernel_manager={kernel_manager is not None}, "
                f"server_client={server_client is not None}"
            )
            kernel_healthy = False

        return ctx, kernel_healthy

    async def heal_kernel(
        self,
        session_id: str,
        kernel_manager=None,
        server_client=None,
        mode: ServerMode = None,
        notebook_path: str = None
    ) -> Optional[str]:
        """죽은 커널 자동 복구 (새 커널 생성 + SessionStore 업데이트).

        커널이 죽었을 때 새 커널을 생성하고 SessionStore의 kernel_id를 자동으로 업데이트합니다.

        Args:
            session_id: 세션 ID (힐링 대상)
            kernel_manager: JUPYTER_SERVER 모드용 kernel manager (optional)
            server_client: MCP_SERVER 모드용 server client (optional)
            mode: ServerMode (JUPYTER_SERVER or MCP_SERVER)
            notebook_path: 노트북 경로 (커널 생성 시 필요, MCP_SERVER 모드)

        Returns:
            Optional[str]: 새로 생성된 kernel_id 또는 None (실패 시)
                - 성공: 새 kernel_id (예: "kernel-abc-123")
                - 실패: None (에러 로그 출력됨)

        Side Effects:
            - 성공 시: SessionStore.update_notebook(session_id, kernel_id=new_id) 호출
            - 실패 시: 에러 로그만 출력, SessionStore 변경 없음

        Example:
            ```python
            # JUPYTER_SERVER 모드 힐링
            new_kernel_id = await manager.heal_kernel(
                session_id="A",
                kernel_manager=km,
                mode=ServerMode.JUPYTER_SERVER
            )
            if new_kernel_id:
                print(f"Kernel healed: {new_kernel_id}")

            # MCP_SERVER 모드 힐링
            new_kernel_id = await manager.heal_kernel(
                session_id="B",
                server_client=client,
                mode=ServerMode.MCP_SERVER,
                notebook_path="/work/notebook.ipynb"
            )
            ```

        Notes:
            - ARK-165: 커널 힐링 로직 중앙화
            - SessionStore 부분 업데이트: kernel_id만 변경
            - 에러 발생 시 None 반환 (예외 전파 안 함)
            - 세션이 없어도 None 반환
        """
        try:
            # 1. 세션 컨텍스트 조회
            ctx = self.session_store.get(session_id)
            if not ctx:
                logger.error(f"Session {session_id} not found for kernel healing")
                return None

            # 2. 커널 생성 시도 (mode별 분기)
            new_kernel_id = None

            if mode == ServerMode.JUPYTER_SERVER and kernel_manager is not None:
                # JUPYTER_SERVER 모드: kernel_manager로 커널 생성
                try:
                    # _start_kernel_local 메서드를 직접 사용하지 않고
                    # kernel_manager의 start_kernel을 직접 호출
                    kernel_info = await kernel_manager.start_kernel()

                    # Handle both dict and string return types
                    if isinstance(kernel_info, dict):
                        new_kernel_id = kernel_info['id']
                    else:
                        # kernel_info is already the kernel_id string
                        new_kernel_id = kernel_info

                    logger.info(
                        f"✓ Created new kernel in JUPYTER_SERVER mode: {new_kernel_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"✗ Failed to create kernel in JUPYTER_SERVER mode: {e}",
                        exc_info=True
                    )
                    return None

            elif mode == ServerMode.MCP_SERVER and server_client is not None:
                # MCP_SERVER 모드: KernelClient로 커널 생성
                try:
                    from jupyter_kernel_client import KernelClient

                    kernel = KernelClient(
                        server_url=server_client.base_url,
                        token=server_client.token
                    )
                    kernel.start(path=notebook_path)
                    new_kernel_id = kernel.id
                    logger.info(
                        f"✓ Created new kernel in MCP_SERVER mode: {new_kernel_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"✗ Failed to create kernel in MCP_SERVER mode: {e}",
                        exc_info=True
                    )
                    return None

            else:
                # Invalid mode 또는 client 없음
                logger.error(
                    f"Invalid mode or missing clients for kernel healing: "
                    f"mode={mode}, kernel_manager={kernel_manager is not None}, "
                    f"server_client={server_client is not None}"
                )
                return None

            # 3. SessionStore 업데이트 (kernel_id만 부분 업데이트)
            if new_kernel_id:
                self.session_store.update_notebook(
                    session_id=session_id,
                    kernel_id=new_kernel_id
                )
                logger.info(
                    f"✓ Kernel healed for session {session_id[:8]}...: {new_kernel_id}"
                )
                return new_kernel_id
            else:
                logger.error(f"✗ Kernel healing failed: new_kernel_id is None")
                return None

        except Exception as e:
            # 예상치 못한 에러 - 로그 후 None 반환
            logger.error(
                f"✗ Unexpected error during kernel healing for session {session_id}: {e}",
                exc_info=True
            )
            return None
