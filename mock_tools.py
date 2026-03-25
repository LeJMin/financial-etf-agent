import random

ETF_CANDIDATES = [
    {
        "ticker": "KODEX KRX 300",
        "path": "./etf/kodex krx300",
        "price_range": (10000, 18000),
    },
    {
        "ticker": "KODEX 200",
        "path": "./etf/kodex200",
        "price_range": (25000, 40000),
    },
    {
        "ticker": "KODEX 미국S&P500",
        "path": "./etf/kodex미국S&P500",
        "price_range": (50000, 90000),
    },
    {
        "ticker": "KODEX 코스피대형주",
        "path": "./etf/kodex코스피대형주",
        "price_range": (18000, 35000),
    },
    {
        "ticker": "KODEX 반도체",
        "path": "./etf/kodex반도체",
        "price_range": (12000, 30000),
    },
    {
        "ticker": "KODEX 철강",
        "path": "./etf/kodex철강",
        "price_range": (9000, 20000),
    },
    {
        "ticker": "KODEX 인도Nifty50",
        "path": "./etf/kodex인도nifty",
        "price_range": (70000, 120000),
    },
]


def generate_random_account(user_name: str):
    """
    사용자별 랜덤 계좌 생성
    """
    # 이름 기반으로 어느 정도 재현 가능하게 seed 부여
    random.seed(user_name)

    cash = random.randint(500_000, 5_000_000)

    selected_etfs = random.sample(ETF_CANDIDATES, k=random.randint(2, 4))

    portfolio = []
    for etf in selected_etfs:
        quantity = random.randint(1, 30)
        current_price = random.randint(*etf["price_range"])

        # 평균 매입단가를 현재가 기준 ±15% 범위에서 생성
        avg_buy_price = int(current_price * random.uniform(0.85, 1.15))

        portfolio.append({
            "ticker": etf["ticker"],
            "quantity": quantity,
            "current_price": current_price,
            "avg_buy_price": avg_buy_price
        })

    orders = []
    order_count = random.randint(2, 5)

    for _ in range(order_count):
        etf = random.choice(ETF_CANDIDATES)
        quantity = random.randint(1, 10)
        order_type = random.choice(["매수", "매도"])

        orders.append({
            "ticker": etf["ticker"],
            "quantity": quantity,
            "type": order_type
        })

    return {
        "user_name": user_name,
        "cash": cash,
        "portfolio": portfolio,
        "orders": orders
    }


def calculate_total_value(account: dict) -> int:
    """
    보유 종목 총 평가금액 계산
    """
    return sum(
        item["quantity"] * item["current_price"]
        for item in account["portfolio"]
    )


def calculate_total_cost(account: dict) -> int:
    """
    총 매입금액 계산
    """
    return sum(
        item["quantity"] * item["avg_buy_price"]
        for item in account["portfolio"]
    )


def calculate_total_return_rate(account: dict) -> float:
    """
    전체 포트폴리오 수익률 계산
    """
    total_cost = calculate_total_cost(account)
    total_value = calculate_total_value(account)

    if total_cost == 0:
        return 0.0

    return ((total_value - total_cost) / total_cost) * 100


def get_account_balance(account: dict):
    """
    예수금 + 총 평가금액 응답
    """
    cash = account["cash"]
    total_value = calculate_total_value(account)

    return {
        "answer": (
            f"{account['user_name']}님의 현재 계좌 예수금은 {cash:,}원이며, "
            f"보유 종목 총 평가금액은 {total_value:,}원입니다."
        ),
        "citations": []
    }


def get_portfolio_status(account: dict):
    """
    보유 ETF + 전체 수익률 응답
    """
    if not account["portfolio"]:
        return {
            "answer": f"{account['user_name']}님은 현재 보유 중인 ETF가 없습니다.",
            "citations": []
        }

    holdings = []
    for item in account["portfolio"]:
        holdings.append(
            f"{item['ticker']} {item['quantity']}주"
        )

    return_rate = calculate_total_return_rate(account)

    return {
        "answer": (
            f"{account['user_name']}님의 현재 보유 ETF는 {', '.join(holdings)}입니다. "
            f"전체 기준 추정 수익률은 {return_rate:+.2f}%입니다."
        ),
        "citations": []
    }


def get_recent_orders(account: dict):
    """
    최근 주문내역 응답
    """
    if not account["orders"]:
        return {
            "answer": f"{account['user_name']}님의 최근 주문내역이 없습니다.",
            "citations": []
        }

    summaries = []
    for order in account["orders"]:
        summaries.append(
            f"{order['ticker']} {order['quantity']}주 {order['type']}"
        )

    return {
        "answer": (
            f"{account['user_name']}님의 최근 주문내역은 총 {len(account['orders'])}건입니다. "
            f"{', '.join(summaries)}"
        ),
        "citations": []
    }


def format_account_for_sidebar(account: dict):
    """
    Streamlit 사이드바 표시용 문자열 구성
    """
    total_value = calculate_total_value(account)
    return_rate = calculate_total_return_rate(account)

    portfolio_lines = []
    for item in account["portfolio"]:
        portfolio_lines.append(
            f"- {item['ticker']} | {item['quantity']}주 | 현재가 {item['current_price']:,}원"
        )

    order_lines = []
    for order in account["orders"]:
        order_lines.append(
            f"- {order['ticker']} | {order['quantity']}주 | {order['type']}"
        )

    return {
        "summary": (
            f"예수금: {account['cash']:,}원\n\n"
            f"총 평가금액: {total_value:,}원\n\n"
            f"전체 수익률: {return_rate:+.2f}%"
        ),
        "portfolio_text": "\n".join(portfolio_lines) if portfolio_lines else "- 없음",
        "orders_text": "\n".join(order_lines) if order_lines else "- 없음"
    }
