def route_question(question: str) -> str:
    q = question.strip().lower()

    account_keywords = [
        "계좌", "잔고", "예수금", "보유", "보유종목", "주문",
        "거래", "체결", "내 계좌", "내 자산",  "산 종목", "샀", "매수", "내가 산", "내가 샀"
    ]

    document_keywords = [
        "약관", "규정", "설명서", "투자", "위험", "수수료",
        "환매", "etf", "pdf", "지수", "구성", "운용"
    ]

    guard_keywords = [
        "날씨", "영화", "연예인", "맛집", "축구", "게임"
    ]

    ambiguous_keywords = [
        "수수료", "보수", "비용", "위험", "환매"
    ]

    # 1. 너무 짧거나 의미 부족
    if len(q) <= 3:
        return "clarify"

    # 2. 계좌 질문 (우선순위 높음)
    if any(keyword in q for keyword in account_keywords):
        return "account"

    # 3. 금융 외 질문
    if any(keyword in q for keyword in guard_keywords):
        return "guard"

    # 4. 애매한 질문 (핵심 개선)
    if any(word in q for word in ambiguous_keywords):
        if len(q) < 10:
            return "clarify"

    # 5. 문서 질문
    if any(keyword in q for keyword in document_keywords):
        return "document"

    # 6. 기본값
    return "document"