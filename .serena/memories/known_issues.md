# 알려진 이슈 및 해결 방법

## 2025-01-18 - NotebookManager와 SessionStore 상태 충돌 (ARK-165)

### 이슈 설명

**증상**: 서로 다른 세션이 같은 `notebook_name`을 사용하려고 할 때 다음 에러 발생:
```
Notebook 'nb1' is already used. Use different notebook_name to create a new notebook on b.ipynb.
```

**재현 조건**:
1. Session A가 `use_notebook(session_id="A", notebook_name="nb1", path="a.ipynb")` 호출
2. Session B가 `use_notebook(session_id="B", notebook_name="nb1", path="b.ipynb")` 호출
3. Session B가 실패 (NotebookManager에 "nb1"이 이미 존재)

**영향 범위**:
- 멀티클라이언트 환경에서 세션 격리 불가능
- 같은 notebook_name을 다른 세션에서 사용 불가
- ARK-165 멀티클라이언트 지원 기능 동작 불가

### 근본 원인

**이중 상태 관리 시스템**:
```
NotebookManager (notebook_name → kernel/path)
       ↕ 충돌
SessionStore (session_id → SessionContext)
```

**구체적 원인**:
1. `use_notebook_tool.py`의 Line 373-389에서 NotebookManager 기반 검증 수행
2. NotebookManager는 `notebook_name`을 전역 키로 사용 (세션 미인식)
3. SessionStore는 `session_id`를 키로 사용 (세션별 격리)
4. 두 시스템이 서로 다른 키를 사용하여 검증 충돌 발생

**문제 코드** (제거됨):
```python
# Line 373-389 (제거된 코드)
if notebook_name in notebook_manager:
    if use_mode == "create":
        if notebook_manager.get_notebook_path(notebook_name) == notebook_path:
            return f"Notebook '{notebook_name}'(path: {notebook_path}) is already created..."
        else:
            return f"Notebook '{notebook_name}' is already used..."
    else:
        # 추가 검증 로직...
```

### 해결 방법

**Step 1: NotebookManager 검증 로직 제거**
- 파일: `jupyter_mcp_server/tools/use_notebook_tool.py`
- 위치: Line 373-389 (19줄)
- 내용: notebook_manager 기반 if/else 검증 블록 완전 제거

**Step 2: session_id 필수화**
- Line 297-310에 session_id 검증 추가:
```python
if not session_id:
    structured_output = {
        "result": {
            "status": "error",
            "message": "session_id is required for multi-client support (ARK-165)...",
            "action": "rejected"
        },
        ...
    }
    return json.dumps(structured_output, ensure_ascii=False)
```

**Step 3: kernel_id 파라미터 수정**
- 함수 시그니처에 `kernel_id: Optional[str] = None` 명시적 추가
- kernel_id 덮어쓰기 버그 수정:
```python
# Before (버그)
kernel_id = existing_ctx.kernel_id if existing_ctx else None

# After (수정)
if kernel_id is None and existing_ctx:
    kernel_id = existing_ctx.kernel_id
```

**Step 4: notebook_manager.add_notebook() 유지**
- backward compatibility를 위해 `add_notebook()` 호출만 유지
- Line 471, 479에서 호출됨
- 향후 점진적으로 제거 예정

### 검증

**테스트 결과**:
- ✅ 기존 14개 테스트 모두 통과 (session_id 추가)
- ✅ 새로운 5개 멀티클라이언트 시나리오 테스트 통과
- ✅ 총 19/19 테스트 통과 (100%)

**시나리오 검증**:
1. ✅ 서로 다른 세션, 같은 notebook_name, 다른 경로 → 성공
2. ✅ 같은 세션, 노트북 전환 → kernel 재사용
3. ✅ 서로 다른 세션, 같은 경로 → 충돌 감지
4. ✅ 같은 세션/노트북 재호출 → idempotent 동작
5. ✅ session_id=None → 명확한 에러

### 재발 방지

**패턴**:
- ✅ SessionStore만 사용하여 상태 관리
- ✅ notebook_manager 검증 로직 사용 금지
- ✅ session_id를 모든 도구에서 필수로 강제
- ✅ 멀티클라이언트 시나리오 테스트 작성

**가이드라인**:
1. 새로운 도구 개발 시 SessionStore만 사용
2. notebook_manager는 backward compatibility 목적으로만 사용
3. session_id를 항상 필수 파라미터로 설계
4. 멀티클라이언트 격리를 고려한 테스트 작성

### 관련 코드

**핵심 파일**:
- [use_notebook_tool.py:297-310](jupyter_mcp_server/tools/use_notebook_tool.py#L297-L310) - session_id 검증
- [use_notebook_tool.py:372-375](jupyter_mcp_server/tools/use_notebook_tool.py#L372-L375) - kernel_id 재사용 로직
- [session_store.py:297-356](jupyter_mcp_server/session_store.py#L297-L356) - SessionStore 검증 로직

**테스트 파일**:
- [test_use_notebook_tool_ark165.py:485-828](tests/test_use_notebook_tool_ark165.py#L485-L828) - 멀티클라이언트 시나리오 테스트

### 추가 참고사항

**Breaking Change**:
- `session_id`가 이제 필수 파라미터
- 기존 코드에서 `session_id=None`을 사용하던 경우 업데이트 필요
- 명확한 에러 메시지 제공으로 마이그레이션 용이

**향후 작업**:
- 다른 cell operation 도구들도 동일한 패턴으로 업데이트 필요
- NotebookManager 완전 제거 계획 수립
