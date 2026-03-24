import os
import time
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# =========================================================
# 기본 설정
# =========================================================
DEFAULT_FAISS_DIR = "faiss_store"
EMBEDDING_BATCH_SIZE = 5
MAX_EMBED_RETRIES = 6
INITIAL_RETRY_DELAY = 2


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")


def normalize_etf_name(name: Optional[str]) -> str:
    """
    ETF 이름 비교용 정규화
    - 공백 제거
    - 대문자 통일
    """
    if not name:
        return ""
    return str(name).strip().replace(" ", "").upper()


# =========================================================
# 파일 / 폴더 로딩 유틸
# =========================================================
def collect_pdf_files(path_str: str) -> List[Path]:
    """
    path_str가 파일이면 해당 PDF 1개 반환,
    폴더이면 폴더 내부의 PDF 파일 목록 반환
    """
    path = Path(path_str)

    if not path.exists():
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {path_str}")

    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"PDF 파일이 아닙니다: {path_str}")
        return [path]

    pdf_files = sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    )

    if not pdf_files:
        raise ValueError(f"폴더 안에 PDF 파일이 없습니다: {path_str}")

    return pdf_files


def load_pdf_docs(pdf_file: Path):
    """
    PDF 1개를 로드해서 docs 반환
    """
    loader = PyMuPDFLoader(str(pdf_file))
    return loader.load()


# =========================================================
# 문서 로딩 / 분할
# =========================================================
def load_and_split_documents(
    pdf_sources: List[Dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100
):
    """
    여러 PDF 또는 PDF 폴더를 불러와 chunk로 분할하고 metadata를 부여합니다.

    pdf_sources 예시:
    [
        {
            "path": "./etf/kodex200",
            "source_type": "system",
            "etf_name": "KODEX 200"
        },
        {
            "path": "./temp_docs/monthly_report.pdf",
            "source_type": "user_upload",
            "etf_name": None
        }
    ]
    """
    all_docs = []

    for item in pdf_sources:
        source_path = item["path"]
        source_type = item.get("source_type", "system")
        etf_name = item.get("etf_name")
        etf_name_norm = normalize_etf_name(etf_name) if etf_name else None

        pdf_files = collect_pdf_files(source_path)

        for pdf_file in pdf_files:
            docs = load_pdf_docs(pdf_file)
            file_name = pdf_file.name

            for doc in docs:
                page_num = doc.metadata.get("page", 0) + 1

                doc.metadata["source"] = file_name
                doc.metadata["page"] = page_num
                doc.metadata["source_type"] = source_type
                doc.metadata["file_path"] = str(pdf_file)
                doc.metadata["parent_path"] = str(source_path)

                if etf_name:
                    doc.metadata["etf_name"] = etf_name
                    doc.metadata["etf_name_norm"] = etf_name_norm
                else:
                    doc.metadata["etf_name"] = None
                    doc.metadata["etf_name_norm"] = None

            all_docs.extend(docs)

    if not all_docs:
        raise ValueError("불러온 문서가 없습니다.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    split_documents = text_splitter.split_documents(all_docs)

    if not split_documents:
        raise ValueError("문서 분할 결과가 비어 있습니다.")

    return split_documents


# =========================================================
# 임베딩 / 벡터스토어 생성
# =========================================================
def get_embeddings():
    """
    임베딩 객체 생성
    - batch size를 낮춰 rate limit 및 max token 문제 완화
    """
    return OpenAIEmbeddings(chunk_size=EMBEDDING_BATCH_SIZE)


def create_vectorstore(
    pdf_sources: List[Dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100
):
    """
    여러 PDF 문서 또는 PDF 폴더를 하나의 벡터스토어로 변환합니다.
    재시도(backoff) 포함.
    """
    check_api_key()

    split_documents = load_and_split_documents(
        pdf_sources=pdf_sources,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    embeddings = get_embeddings()

    delay = INITIAL_RETRY_DELAY
    last_error = None

    for attempt in range(MAX_EMBED_RETRIES):
        try:
            vectorstore = FAISS.from_documents(
                documents=split_documents,
                embedding=embeddings
            )
            return vectorstore

        except Exception as e:
            last_error = e
            error_text = str(e).lower()

            # rate limit / 일시적 토큰 한도 / transient error 재시도
            if (
                "rate_limit" in error_text
                or "429" in error_text
                or "max_tokens_per_request" in error_text
                or "tokens per minute" in error_text
                or "try again" in error_text
            ):
                if attempt < MAX_EMBED_RETRIES - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue

            raise

    raise RuntimeError(f"Vectorstore 생성 실패: {last_error}")


def save_vectorstore(vectorstore, folder_path: str = DEFAULT_FAISS_DIR):
    """
    로컬에 FAISS 저장
    """
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(folder_path)


def load_vectorstore(folder_path: str = DEFAULT_FAISS_DIR):
    """
    저장된 FAISS 로드
    """
    check_api_key()

    if not Path(folder_path).exists():
        raise FileNotFoundError(f"저장된 벡터스토어가 없습니다: {folder_path}")

    embeddings = get_embeddings()

    return FAISS.load_local(
        folder_path,
        embeddings,
        allow_dangerous_deserialization=True
    )


# =========================================================
# 검색용 retriever
# =========================================================
def build_filter(
    allowed_etfs: Optional[List[str]] = None,
    include_user_uploads: bool = True,
    only_user_uploads: bool = False
):
    """
    metadata filter 생성
    """
    normalized_etfs = []
    if allowed_etfs:
        normalized_etfs = [
            normalize_etf_name(x) for x in allowed_etfs if x
        ]

    if only_user_uploads:
        return {"source_type": "user_upload"}

    if normalized_etfs:
        etf_filter = {"etf_name_norm": {"$in": normalized_etfs}}

        if include_user_uploads:
            return {
                "$or": [
                    etf_filter,
                    {"source_type": "user_upload"}
                ]
            }
        return etf_filter

    if include_user_uploads:
        return None

    return {"source_type": "system"}


def get_retriever(
    vectorstore,
    top_k: int = 4,
    score_threshold: float = 0.5,
    allowed_etfs: Optional[List[str]] = None,
    include_user_uploads: bool = True,
    only_user_uploads: bool = False
):
    """
    relevance filtering + ETF metadata filtering 포함 retriever 생성
    """
    search_kwargs = {
        "k": top_k,
        "score_threshold": score_threshold
    }

    metadata_filter = build_filter(
        allowed_etfs=allowed_etfs,
        include_user_uploads=include_user_uploads,
        only_user_uploads=only_user_uploads
    )

    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs=search_kwargs
    )


# =========================================================
# 계좌에서 보유 ETF 추출
# =========================================================
def extract_account_etfs(account: dict) -> List[str]:
    """
    mock account에서 보유 ETF ticker 목록 추출
    """
    if not account:
        return []

    portfolio = account.get("portfolio", [])
    if not isinstance(portfolio, list):
        return []

    tickers = []
    for item in portfolio:
        ticker = item.get("ticker")
        if ticker:
            tickers.append(ticker)

    return tickers


# =========================================================
# LLM 입력용 문서 포맷
# =========================================================
def format_docs(docs, max_length: int = 2200):
    """
    검색된 문서를 LLM 입력용 문자열로 정리
    - ETF명 / 문서명 / 페이지 / 문서유형 포함
    - context 길이 제한
    """
    formatted = []

    for doc in docs:
        source = doc.metadata.get("source", "알 수 없는 문서")
        page = doc.metadata.get("page", "?")
        etf_name = doc.metadata.get("etf_name")
        source_type = doc.metadata.get("source_type", "unknown")
        content = doc.page_content.strip()

        if etf_name:
            header = f"[ETF: {etf_name} | 문서: {source} | 페이지: {page} | 유형: {source_type}]"
        else:
            header = f"[문서: {source} | 페이지: {page} | 유형: {source_type}]"

        formatted.append(f"{header}\n{content}")

    full_text = "\n\n".join(formatted)
    return full_text[:max_length]


def format_citations(docs):
    """
    검색된 문서에서 citation용 정보 정리
    - source/page 기준 중복 제거
    """
    citations = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "알 수 없는 문서")
        page = doc.metadata.get("page", "?")
        etf_name = doc.metadata.get("etf_name")
        source_type = doc.metadata.get("source_type", "unknown")
        snippet = doc.page_content[:100].replace("\n", " ").strip()

        key = (source, page, etf_name, source_type)
        if key not in seen:
            seen.add(key)
            citations.append({
                "source": source,
                "page": page,
                "etf_name": etf_name,
                "source_type": source_type,
                "snippet": snippet
            })

    return citations


# =========================================================
# 질문 응답
# =========================================================
def answer_question(question, retriever, fallback_retriever=None):
    """
    질문을 받아 답변 + citation 정보를 함께 반환합니다.
    """
    check_api_key()

    docs = retriever.invoke(question)

    if not docs and fallback_retriever is not None:
        docs = fallback_retriever.invoke(question)

    context = format_docs(docs)

    template = """
당신은 ETF 투자설명서 및 금융 문서를 기반으로 답변하는 전문 AI입니다.

다음 규칙을 반드시 따르세요:
1. 반드시 Context에 있는 정보만 사용하세요.
2. 질문과 직접 관련된 핵심 정보만 추출하여 요약하세요.
3. 특히 다음 항목이 있으면 우선적으로 반영하세요:
   - 총보수 / 수수료 구조
   - 투자 목적 / 전략
   - 위험 요인
   - 비교지수 / 추적지수
4. 표, 항목, 정의 형태라도 의미를 파악해 자연스럽게 설명하세요.
5. Context에 근거가 없으면 반드시 "문서에서 확인되지 않습니다."라고 답하세요.
6. 절대 추측하거나 외부 지식을 추가하지 마세요.
7. 답변은 한국어로 2~4문장, 핵심만 간결하게 작성하세요.
8. 여러 문서가 함께 검색되더라도, 질문과 가장 직접적으로 관련된 문서 기준으로 답하세요.

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        max_tokens=350
    )

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    answer_text = response.content if hasattr(response, "content") else str(response)
    citations = format_citations(docs)

    return {
        "answer": answer_text,
        "citations": citations,
        "retrieved_docs": docs
    }


# =========================================================
# 편의 함수
# =========================================================
def make_system_pdf_sources(etf_path_map: dict):
    """
    ETF 문서 맵(파일 또는 폴더 경로)을 pdf_sources 형태로 변환
    """
    pdf_sources = []

    for etf_name, path_value in etf_path_map.items():
        pdf_sources.append({
            "path": path_value,
            "source_type": "system",
            "etf_name": etf_name
        })

    return pdf_sources


def make_uploaded_pdf_sources(uploaded_pdf_paths: list):
    """
    사용자 업로드 PDF 경로 목록을 pdf_sources 형태로 변환
    """
    pdf_sources = []

    for pdf_path in uploaded_pdf_paths:
        pdf_sources.append({
            "path": pdf_path,
            "source_type": "user_upload",
            "etf_name": None
        })

    return pdf_sources


def combine_pdf_sources(system_pdf_sources=None, uploaded_pdf_sources=None):
    """
    시스템 문서 + 사용자 업로드 문서를 합쳐서 반환
    """
    combined = []

    if system_pdf_sources:
        combined.extend(system_pdf_sources)

    if uploaded_pdf_sources:
        combined.extend(uploaded_pdf_sources)

    return combined