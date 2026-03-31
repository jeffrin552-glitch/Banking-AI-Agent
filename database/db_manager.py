"""
database/db_manager.py
Handles all JSON database read/write operations.
Thread-safe with file locking for production reliability.

FIX: On Render (and similar platforms), the source directory is read-only.
We copy accounts.json to /tmp on first run so writes succeed.
"""

import json
import os
import shutil
import threading
from datetime import datetime
from typing import Optional

# Source (bundled with repo) — may be read-only on Render
_SRC_PATH = os.path.join(os.path.dirname(__file__), "accounts.json")

# Writable copy in /tmp — guaranteed writable on every platform
_DB_PATH = "/tmp/bankassist_accounts.json"

_lock = threading.Lock()


def _ensure_db():
    """Copy the seed DB to /tmp if it doesn't exist there yet."""
    if not os.path.exists(_DB_PATH):
        shutil.copy2(_SRC_PATH, _DB_PATH)


def _read_db() -> dict:
    """Read the entire database."""
    _ensure_db()
    with open(_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_db(data: dict):
    """Write the entire database atomically."""
    _ensure_db()
    tmp_path = _DB_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, _DB_PATH)


def get_account_by_name(name: str) -> Optional[dict]:
    """Find account by first name (case-insensitive)."""
    with _lock:
        db = _read_db()
        name_lower = name.strip().lower()
        for acc in db["accounts"]:
            if acc["name"].lower() == name_lower or acc["full_name"].lower().startswith(name_lower):
                return acc
    return None


def get_account_by_id(account_id: str) -> Optional[dict]:
    """Find account by account ID."""
    with _lock:
        db = _read_db()
        for acc in db["accounts"]:
            if acc["account_id"] == account_id:
                return acc
    return None


def get_all_accounts() -> list:
    """Return all accounts."""
    with _lock:
        db = _read_db()
        return db["accounts"]


def get_balance(name: str) -> dict:
    """
    Get balance for a user by name.
    Returns: {success, account_id, name, balance, account_type} or {success, error}
    """
    acc = get_account_by_name(name)
    if not acc:
        return {"success": False, "error": f"No account found for '{name}'."}
    return {
        "success": True,
        "account_id": acc["account_id"],
        "name": acc["full_name"],
        "balance": acc["balance"],
        "account_type": acc["account_type"],
    }


def deposit(name: str, amount: float) -> dict:
    """
    Deposit amount into account.
    Returns: {success, account_id, name, old_balance, new_balance, transaction_id}
    """
    if amount <= 0:
        return {"success": False, "error": "Deposit amount must be greater than ₹0."}
    if amount > 10_00_000:
        return {"success": False, "error": "Single deposit limit is ₹10,00,000."}

    with _lock:
        db = _read_db()
        for acc in db["accounts"]:
            if acc["name"].lower() == name.strip().lower() or \
               acc["full_name"].lower().startswith(name.strip().lower()):
                old_balance = acc["balance"]
                acc["balance"] = round(old_balance + amount, 2)
                txn_id = _record_transaction(db, acc["account_id"], "DEPOSIT", amount, old_balance, acc["balance"])
                _write_db(db)
                return {
                    "success": True,
                    "transaction_id": txn_id,
                    "account_id": acc["account_id"],
                    "name": acc["full_name"],
                    "old_balance": old_balance,
                    "new_balance": acc["balance"],
                    "amount": amount,
                }
    return {"success": False, "error": f"No account found for '{name}'."}


def withdraw(name: str, amount: float) -> dict:
    """
    Withdraw amount from account.
    Returns: {success, account_id, name, old_balance, new_balance, transaction_id}
    """
    if amount <= 0:
        return {"success": False, "error": "Withdrawal amount must be greater than ₹0."}

    with _lock:
        db = _read_db()
        for acc in db["accounts"]:
            if acc["name"].lower() == name.strip().lower() or \
               acc["full_name"].lower().startswith(name.strip().lower()):
                if acc["balance"] < amount:
                    return {
                        "success": False,
                        "error": f"Insufficient balance. {acc['full_name']} has ₹{acc['balance']:,.2f}, but you tried to withdraw ₹{amount:,.2f}."
                    }
                old_balance = acc["balance"]
                acc["balance"] = round(old_balance - amount, 2)
                txn_id = _record_transaction(db, acc["account_id"], "WITHDRAWAL", amount, old_balance, acc["balance"])
                _write_db(db)
                return {
                    "success": True,
                    "transaction_id": txn_id,
                    "account_id": acc["account_id"],
                    "name": acc["full_name"],
                    "old_balance": old_balance,
                    "new_balance": acc["balance"],
                    "amount": amount,
                }
    return {"success": False, "error": f"No account found for '{name}'."}


def transfer(from_name: str, to_name: str, amount: float) -> dict:
    """
    Transfer amount between two accounts.
    Returns: {success, from_account, to_account, amount, transaction_id}
    """
    if amount <= 0:
        return {"success": False, "error": "Transfer amount must be greater than ₹0."}
    if from_name.strip().lower() == to_name.strip().lower():
        return {"success": False, "error": "Cannot transfer to the same account."}

    with _lock:
        db = _read_db()
        from_acc = None
        to_acc = None
        for acc in db["accounts"]:
            n = acc["name"].lower()
            fn = acc["full_name"].lower()
            if n == from_name.strip().lower() or fn.startswith(from_name.strip().lower()):
                from_acc = acc
            if n == to_name.strip().lower() or fn.startswith(to_name.strip().lower()):
                to_acc = acc

        if not from_acc:
            return {"success": False, "error": f"Sender account '{from_name}' not found."}
        if not to_acc:
            return {"success": False, "error": f"Recipient account '{to_name}' not found."}
        if from_acc["balance"] < amount:
            return {
                "success": False,
                "error": f"Insufficient balance. {from_acc['full_name']} has ₹{from_acc['balance']:,.2f}, but tried to transfer ₹{amount:,.2f}."
            }

        from_old = from_acc["balance"]
        to_old = to_acc["balance"]
        from_acc["balance"] = round(from_old - amount, 2)
        to_acc["balance"] = round(to_old + amount, 2)

        txn_id = _record_transaction(
            db, from_acc["account_id"], "TRANSFER_OUT", amount, from_old, from_acc["balance"],
            reference=to_acc["account_id"]
        )
        _record_transaction(
            db, to_acc["account_id"], "TRANSFER_IN", amount, to_old, to_acc["balance"],
            reference=from_acc["account_id"]
        )
        _write_db(db)
        return {
            "success": True,
            "transaction_id": txn_id,
            "from_name": from_acc["full_name"],
            "from_account_id": from_acc["account_id"],
            "from_old_balance": from_old,
            "from_new_balance": from_acc["balance"],
            "to_name": to_acc["full_name"],
            "to_account_id": to_acc["account_id"],
            "to_old_balance": to_old,
            "to_new_balance": to_acc["balance"],
            "amount": amount,
        }


def get_transaction_history(name: str, limit: int = 5) -> dict:
    """Get last N transactions for a user."""
    acc = get_account_by_name(name)
    if not acc:
        return {"success": False, "error": f"No account found for '{name}'."}

    with _lock:
        db = _read_db()
        txns = [t for t in db["transactions"] if t["account_id"] == acc["account_id"]]
        txns_sorted = sorted(txns, key=lambda x: x["timestamp"], reverse=True)[:limit]
        return {
            "success": True,
            "name": acc["full_name"],
            "account_id": acc["account_id"],
            "transactions": txns_sorted,
        }


def _record_transaction(db: dict, account_id: str, txn_type: str,
                        amount: float, old_balance: float, new_balance: float,
                        reference: str = None) -> str:
    """Internal: append a transaction record."""
    txn_id = f"TXN{datetime.now().strftime('%Y%m%d%H%M%S%f')[:18]}"
    record = {
        "transaction_id": txn_id,
        "account_id": account_id,
        "type": txn_type,
        "amount": amount,
        "balance_before": old_balance,
        "balance_after": new_balance,
        "timestamp": datetime.now().isoformat(),
        "reference": reference,
    }
    db["transactions"].append(record)
    return txn_id
