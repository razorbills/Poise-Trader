from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="E:/Poise Trader/Files/.env")

WALLET_PUBLIC_KEY = os.getenv("WALLET_PUBLIC_KEY")
WALLET_PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY")

if not WALLET_PUBLIC_KEY or not WALLET_PRIVATE_KEY:
    raise ValueError("❌ Wallet keys not found. Check your .env file.")
