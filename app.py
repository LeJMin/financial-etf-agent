import os
import time
from pathlib import Path
import streamlit as st

from utils import save_log, save_account_log
from router import route_question
from mock_tools import (
    generate_random_account,
    get_account_balance,
    get_portfolio_status,
    get_recent_orders,
    format_account_for_sidebar
)
from rag_module import (
    create_vectorstore,
    load_vectorstore,
    save_vectorstore,
    make_system_pdf_sources,
    make_uploaded_pdf_sources,
    combine_pdf_sources,
    extract_account_etfs,
    get_retriever,
    answer_question
)

st.set_page_config(page_title="ETF RAG 챗봇", layout="wide")

st.title("🤖 금융 문서 기반 AI 에이전트")
st.markdown(
    "기본 ETF 문서와 업로드한 PDF를 함께 참고하여 질문에 답변합니다. "
    "사용자 계좌의 보유 ETF가 있으면 해당 문서를 우선 검색합니다."
)

# ---------------------------
# 기본 설정
# ---------------------------
TEMP_DIR = "temp_docs"
FAISS_DIR = "faiss_store"

os.makedirs(TEMP_DIR, exist_ok=True)

# ✅ ETF별 '폴더 경로'
ETF_PATH_MAP = {
    "KODEX KRX 300": "./etf/kodex krx300",
    "KODEX 200": "./etf/kodex200",
    "KODEX 미국S&P500": "./etf/kodex미국S&P500",
    "KODEX 코스피대형주": "./etf/kodex코스피대형주",
    "KODEX 반도체": "./etf/kodex반도체",
    "KODEX 철강": "./etf/kodex철강",
    "KODEX 인도Nifty50": "./etf/kodex인도nifty",
}


# ---------------------------
# 유틸
# ---------------------------
def save_uploaded_files(uploaded_files):
    saved_paths = []

    for uploaded_file in uploaded_files:
        temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(temp_path)

    return saved_paths


def build_file_signature(uploaded_files):
    if not uploaded_files:
        return ""
    return "|".join(f"{file.name}_{file.size}" for file in uploaded_files)


def build_system_signature(etf_path_map):
    """
    시스템 문서 변경 여부 확인용 signature
    """
    items = []
    for etf_name, folder_path in sorted(etf_path_map.items()):
        items.append(f"{etf_name}:{folder_path}")
    return "|".join(items)


def format_citations_for_display(citations):
    if not citations:
        return "참고 문서 정보가 없습니다."

    lines = []
    for item in citations:
        source = item.get("source", "알 수 없는 문서")
        page = item.get("page", "?")
        snippet = item.get("snippet", "")
        etf_name = item.get("etf_name")
        source_type = item.get("source_type", "unknown")

        if etf_name:
            lines.append(
                f"- **{source}** / {page}페이지 / ETF: {etf_name} / 유형: {source_type}\n"
                f"  - {snippet}"
            )
        else:
            lines.append(
                f"- **{source}** / {page}페이지 / 유형: {source_type}\n"
                f"  - {snippet}"
            )

    return "\n".join(lines)


def get_valid_system_path_map(etf_path_map):
    """
    실제 존재하는 ETF 폴더(또는 파일)만 남김
    """
    valid_map = {}
    missing = []

    for etf_name, path_value in etf_path_map.items():
        if os.path.exists(path_value):
            valid_map[etf_name] = path_value
        else:
            missing.append((etf_name, path_value))

    return valid_map, missing


def build_base_faiss_signature(valid_system_map, chunk_size, chunk_overlap):
    """
    기본 ETF 문서용 저장 인덱스 식별자
    """
    items = []
    for etf_name, folder_path in sorted(valid_system_map.items()):
        items.append(f"{etf_name}:{folder_path}")
    return f"{'|'.join(items)}__{chunk_size}__{chunk_overlap}"


def get_base_faiss_dir(valid_system_map, chunk_size, chunk_overlap):
    """
    chunk 옵션이 바뀌면 다른 저장 폴더를 쓰도록 구성
    """
    safe_sig = build_base_faiss_signature(valid_system_map, chunk_size, chunk_overlap)
    safe_sig = safe_sig.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")
    return os.path.join(FAISS_DIR, safe_sig)


def initialize_vectorstore(uploaded_files, chunk_size, chunk_overlap):
    """
    시스템 ETF 폴더 + 업로드 PDF를 합쳐 vectorstore 생성
    - 업로드 문서가 없으면 저장된 기본 인덱스 재사용
    - 업로드 문서가 있으면 매번 새로 생성
    """
    valid_system_map, missing_paths = get_valid_system_path_map(ETF_PATH_MAP)
    system_sources = make_system_pdf_sources(valid_system_map)

    uploaded_paths = save_uploaded_files(uploaded_files) if uploaded_files else []
    uploaded_sources = make_uploaded_pdf_sources(uploaded_paths)

    if not system_sources and not uploaded_sources:
        raise ValueError("분석할 문서가 없습니다. ETF 폴더 경로나 업로드 파일을 확인하세요.")

    base_faiss_dir = get_base_faiss_dir(valid_system_map, chunk_size, chunk_overlap)

    # 1) 업로드 문서가 없는 경우: 저장 인덱스 재사용
    if not uploaded_paths:
        if os.path.exists(base_faiss_dir):
            vectorstore = load_vectorstore(base_faiss_dir)
        else:
            vectorstore = create_vectorstore(
                pdf_sources=system_sources,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            save_vectorstore(vectorstore, base_faiss_dir)

        return vectorstore, valid_system_map, missing_paths, uploaded_paths

    # 2) 업로드 문서가 있는 경우: 시스템 + 업로드 합쳐서 새로 생성
    all_sources = combine_pdf_sources(
        system_pdf_sources=system_sources,
        uploaded_pdf_sources=uploaded_sources
    )

    vectorstore = create_vectorstore(
        pdf_sources=all_sources,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return vectorstore, valid_system_map, missing_paths, uploaded_paths


def build_vectorstore_signature(uploaded_files, chunk_size, chunk_overlap, top_k):
    """
    세션 내 재생성 여부 판단용 signature
    """
    upload_sig = build_file_signature(uploaded_files)
    system_sig = build_system_signature(ETF_PATH_MAP)
    return f"{system_sig}__{upload_sig}__{chunk_size}__{chunk_overlap}__{top_k}"


def make_document_result(prompt, current_account, top_k):
    """
    document 질문 처리:
    - 보유 ETF 기준 우선 검색
    - 업로드 문서 포함
    - 검색 실패 시 전체 문서 fallback
    """
    if st.session_state.vectorstore is None:
        return {
            "answer": "아직 분석된 문서가 없습니다. 기본 ETF 폴더 경로 또는 업로드 문서를 확인해주세요.",
            "citations": []
        }

    user_etfs = extract_account_etfs(current_account) if current_account else []

    filtered_retriever = get_retriever(
        vectorstore=st.session_state.vectorstore,
        top_k=top_k,
        score_threshold=0.5,
        allowed_etfs=user_etfs,
        include_user_uploads=True,
        only_user_uploads=False
    )

    fallback_retriever = get_retriever(
        vectorstore=st.session_state.vectorstore,
        top_k=top_k,
        score_threshold=0.5,
        allowed_etfs=None,
        include_user_uploads=True,
        only_user_uploads=False
    )

    return answer_question(
        question=prompt,
        retriever=filtered_retriever,
        fallback_retriever=fallback_retriever
    )


# ---------------------------
# 사이드바
# ---------------------------
with st.sidebar:
    st.header("설정")

    user_name = st.text_input("사용자 이름", placeholder="예: 정민")

    uploaded_files = st.file_uploader(
        "PDF 파일을 업로드하세요",
        type=["pdf"],
        accept_multiple_files=True,
        help="월간 리서치, 보고서 등 추가 참고 문서를 넣을 수 있습니다."
    )

    chunk_size = st.slider("Chunk Size", 300, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk Overlap", 50, 300, 100, 10)
    top_k = st.slider("Top-K", 1, 10, 4, 1)

    st.caption("운영자용 튜닝 옵션입니다.")


# ---------------------------
# 사용자 계좌 관리
# ---------------------------
if "accounts" not in st.session_state:
    st.session_state.accounts = {}

current_account = None

if user_name:
    if user_name not in st.session_state.accounts:
        account = generate_random_account(user_name)
        st.session_state.accounts[user_name] = account
        save_account_log(user_name, account)

    current_account = st.session_state.accounts[user_name]

    sidebar_info = format_account_for_sidebar(current_account)

    st.sidebar.markdown(f"### 👤 {user_name}님의 계좌")
    st.sidebar.markdown(sidebar_info["summary"])

    with st.sidebar.expander("보유 종목"):
        st.markdown(sidebar_info["portfolio_text"])

    with st.sidebar.expander("최근 주문내역"):
        st.markdown(sidebar_info["orders_text"])


# ---------------------------
# 초기 세션 상태
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "current_signature" not in st.session_state:
    st.session_state.current_signature = ""

if "loaded_system_map" not in st.session_state:
    st.session_state.loaded_system_map = {}

if "missing_system_paths" not in st.session_state:
    st.session_state.missing_system_paths = []

if "uploaded_paths" not in st.session_state:
    st.session_state.uploaded_paths = []


# ---------------------------
# 문서 분석
# ---------------------------
current_signature = build_vectorstore_signature(
    uploaded_files=uploaded_files,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    top_k=top_k
)

if st.session_state.current_signature != current_signature:
    try:
        with st.spinner("문서를 분석 중입니다..."):
            vectorstore, valid_system_map, missing_paths, uploaded_paths = initialize_vectorstore(
                uploaded_files=uploaded_files,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.loaded_system_map = valid_system_map
            st.session_state.missing_system_paths = missing_paths
            st.session_state.uploaded_paths = uploaded_paths
            st.session_state.current_signature = current_signature
            st.session_state.messages = []

        st.success("문서 분석이 완료되었습니다.")

    except Exception as e:
        st.error(f"문서 분석 중 오류가 발생했습니다: {e}")
        st.stop()


# ---------------------------
# 현재 분석 대상 문서 표시
# ---------------------------
with st.expander("현재 분석 대상 문서 보기"):
    st.markdown("**기본 ETF 폴더**")
    if st.session_state.loaded_system_map:
        for etf_name, folder_path in st.session_state.loaded_system_map.items():
            st.write(f"- {etf_name} ({folder_path})")
    else:
        st.write("- 불러온 기본 ETF 문서가 없습니다.")

    st.markdown("**업로드 문서**")
    if uploaded_files:
        for file in uploaded_files:
            st.write(f"- {file.name}")
    else:
        st.write("- 업로드된 문서 없음")

    if st.session_state.missing_system_paths:
        st.markdown("**경로를 찾지 못한 ETF 폴더/파일**")
        for etf_name, path_value in st.session_state.missing_system_paths:
            st.write(f"- {etf_name}: {path_value}")


# ---------------------------
# 보유 ETF 안내
# ---------------------------
if current_account:
    user_etfs = extract_account_etfs(current_account)
    st.caption(
        f"현재 사용자의 보유 ETF 기준 우선 검색 대상: "
        f"{', '.join(user_etfs) if user_etfs else '없음'}"
    )
else:
    st.caption("사용자 이름을 입력하면 mock 계좌가 생성되고, 보유 ETF 기준 우선 검색이 적용됩니다.")


# ---------------------------
# 기존 메시지 출력
# ---------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "citations" in message:
            with st.expander("참고 문서"):
                st.markdown(format_citations_for_display(message["citations"]))


# ---------------------------
# 사용자 입력
# ---------------------------
prompt = st.chat_input("질문을 입력하세요")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("답변 생성 중..."):
                start_time = time.time()
                route = route_question(prompt)

                if route == "document":
                    result = make_document_result(
                        prompt=prompt,
                        current_account=current_account,
                        top_k=top_k
                    )

                elif route == "account":
                    if not current_account:
                        result = {
                            "answer": "사용자 이름을 먼저 입력해주세요.",
                            "citations": []
                        }
                    else:
                        if "주문" in prompt or "거래" in prompt:
                            result = get_recent_orders(current_account)
                        elif "보유" in prompt or "수익률" in prompt:
                            result = get_portfolio_status(current_account)
                        else:
                            result = get_account_balance(current_account)

                elif route == "guard":
                    result = {
                        "answer": "이 서비스는 금융 문서 및 계좌 관련 질문에만 답변합니다.",
                        "citations": []
                    }

                elif route == "clarify":
                    result = {
                        "answer": "질문이 모호합니다. 어떤 내용을 알고 싶은지 구체적으로 입력해주세요.",
                        "citations": []
                    }

                else:
                    result = {
                        "answer": "질문 처리 중 문제가 발생했습니다.",
                        "citations": []
                    }

                answer = result["answer"]
                citations = result.get("citations", [])
                response_time = round(time.time() - start_time, 2)

                save_log(
                    question=prompt,
                    route=route,
                    response_time=response_time,
                    success=True,
                    user_name=user_name if user_name else "unknown"
                )

                st.markdown(answer)

                with st.expander("참고 문서"):
                    st.markdown(format_citations_for_display(citations))

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations
            })

        except Exception as e:
            save_log(
                question=prompt,
                route="error",
                response_time=0,
                success=False,
                user_name=user_name if user_name else "unknown"
            )

            st.error(f"오류 발생: {e}")