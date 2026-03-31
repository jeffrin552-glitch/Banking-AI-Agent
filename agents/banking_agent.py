"""
agents/banking_agent.py
LangGraph-powered agentic AI for banking operations.
ReAct pattern: agent reasons -> calls tools -> reasons again -> responds.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Annotated, TypedDict, Literal
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from database import db_manager

# ══════════════════════════════════════════════
# BANKING TOOLS (LangChain @tool decorators)
# ══════════════════════════════════════════════

@tool
def check_balance(customer_name: str) -> str:
    """
    Check the bank account balance of a customer by their name.
    Use whenever someone asks about balance or how much money someone has.

    Args:
        customer_name: First name or full name (e.g. 'Ramu', 'Vishnu Prasad')
    """
    result = db_manager.get_balance(customer_name)
    if result["success"]:
        return (
            f"BALANCE_OK|name={result['name']}|id={result['account_id']}"
            f"|type={result['account_type']}|balance={result['balance']:.2f}"
        )
    return f"BALANCE_FAIL|error={result['error']}"


@tool
def deposit_money(customer_name: str, amount: float) -> str:
    """
    Deposit money into a customer's bank account.
    Use when someone says 'deposit X to NAME' or 'add X to NAME account'.

    Args:
        customer_name: Name of the account holder
        amount: Amount in rupees (must be > 0)
    """
    result = db_manager.deposit(customer_name, amount)
    if result["success"]:
        return (
            f"DEPOSIT_OK|txn={result['transaction_id']}|name={result['name']}"
            f"|amount={result['amount']:.2f}|before={result['old_balance']:.2f}"
            f"|after={result['new_balance']:.2f}"
        )
    return f"DEPOSIT_FAIL|error={result['error']}"


@tool
def withdraw_money(customer_name: str, amount: float) -> str:
    """
    Withdraw money from a customer's bank account.
    Use when someone says 'withdraw X from NAME' or 'deduct X from NAME'.

    Args:
        customer_name: Name of the account holder
        amount: Amount in rupees (must be > 0)
    """
    result = db_manager.withdraw(customer_name, amount)
    if result["success"]:
        return (
            f"WITHDRAW_OK|txn={result['transaction_id']}|name={result['name']}"
            f"|amount={result['amount']:.2f}|before={result['old_balance']:.2f}"
            f"|after={result['new_balance']:.2f}"
        )
    return f"WITHDRAW_FAIL|error={result['error']}"


@tool
def transfer_money(from_customer: str, to_customer: str, amount: float) -> str:
    """
    Transfer money between two customer accounts.
    Use when someone says 'transfer X from A to B' or 'send X to NAME'.

    Args:
        from_customer: Sender's name
        to_customer: Recipient's name
        amount: Amount in rupees
    """
    result = db_manager.transfer(from_customer, to_customer, amount)
    if result["success"]:
        return (
            f"TRANSFER_OK|txn={result['transaction_id']}"
            f"|from={result['from_name']}|from_after={result['from_new_balance']:.2f}"
            f"|to={result['to_name']}|to_after={result['to_new_balance']:.2f}"
            f"|amount={result['amount']:.2f}"
        )
    return f"TRANSFER_FAIL|error={result['error']}"


@tool
def get_transaction_history(customer_name: str, limit: int = 5) -> str:
    """
    Get recent transaction history for a customer.
    Use when someone asks about past transactions or recent activity.

    Args:
        customer_name: Name of the customer
        limit: Number of transactions to show (default 5)
    """
    result = db_manager.get_transaction_history(customer_name, limit)
    if not result["success"]:
        return f"HISTORY_FAIL|error={result['error']}"
    txns = result["transactions"]
    if not txns:
        return f"HISTORY_EMPTY|name={result['name']}"
    lines = [f"Transaction history for {result['name']} ({result['account_id']}):"]
    for i, t in enumerate(txns, 1):
        dt = t["timestamp"][:16].replace("T", " ")
        lines.append(f"{i}. [{dt}] {t['type']} ₹{t['amount']:,.2f} | Balance: ₹{t['balance_after']:,.2f}")
    return "\n".join(lines)


@tool
def list_all_customers() -> str:
    """
    List all customers in the bank. Use when someone asks who are the customers or to show all accounts.
    """
    accounts = db_manager.get_all_accounts()
    lines = ["All Bank Customers:"]
    for acc in accounts:
        lines.append(f"• {acc['full_name']} | {acc['account_id']} | {acc['account_type']} | ₹{acc['balance']:,.2f}")
    return "\n".join(lines)


# ══════════════════════════════════════════════
# LANGGRAPH STATE & GRAPH
# ══════════════════════════════════════════════

class BankState(TypedDict):
    messages: Annotated[list, add_messages]


BANKING_TOOLS = [
    check_balance, deposit_money, withdraw_money,
    transfer_money, get_transaction_history, list_all_customers,
]

SYSTEM_PROMPT = """You are BankAssist AI — a professional, secure, intelligent banking assistant for an Indian bank.

You have access to these banking tools:
- check_balance: Check a customer's balance
- deposit_money: Deposit money into an account
- withdraw_money: Withdraw money from an account
- transfer_money: Transfer money between accounts
- get_transaction_history: Show recent transactions
- list_all_customers: List all bank customers

RULES:
1. Only answer banking and ATM related questions. For anything else say: "I only handle banking queries."
2. ALWAYS use the correct tool for banking operations — never guess or fabricate numbers.
3. Be concise, professional, and clear.
4. Never ask for or reveal PINs, passwords, or OTPs.
5. If asked who created you: "I am BankAssist AI, created by Anusha Kovi."
6. After tool results, summarize clearly in natural language with ₹ symbol.
7. For transfers like "deposit 1000 to Ramu" — use deposit_money tool.
8. For "send 500 from Vishnu to Ramu" — use transfer_money tool.

Today's date: {date}"""


def build_banking_agent(groq_api_key: str):
    """Build and compile the LangGraph ReAct agent."""
    # FIX: Use a valid Groq model. "openai/gpt-oss-120b" is not a real Groq model.
    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama3-70b-8192",   # ✅ Valid Groq model
        max_tokens=600,
        temperature=0,
    ).bind_tools(BANKING_TOOLS)

    tool_node = ToolNode(BANKING_TOOLS)

    def agent_node(state: BankState) -> BankState:
        sys_msg = SystemMessage(
            content=SYSTEM_PROMPT.format(date=datetime.now().strftime("%d %b %Y"))
        )
        response = llm.invoke([sys_msg] + state["messages"])
        return {"messages": [response]}

    def should_continue(state: BankState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(BankState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# ══════════════════════════════════════════════
# CONVERSATION MANAGER
# ══════════════════════════════════════════════

class BankingConversation:
    """Stateful multi-turn conversation manager."""

    def __init__(self, groq_api_key: str):
        self.agent = build_banking_agent(groq_api_key)
        self.history: list = []

    def chat(self, user_message: str) -> str:
        """Process a user message and return the AI response."""
        self.history.append(HumanMessage(content=user_message))
        result = self.agent.invoke({"messages": self.history})

        # Find final AI text response (not a tool-call message)
        final = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                has_tool_calls = hasattr(msg, "tool_calls") and bool(msg.tool_calls)
                if not has_tool_calls:
                    final = msg
                    break

        if not final:
            final = result["messages"][-1]

        # Update history to include tool messages too (for context)
        self.history = list(result["messages"])
        content = final.content if hasattr(final, "content") else str(final)
        return content

    def reset(self):
        """Clear conversation history."""
        self.history = []
