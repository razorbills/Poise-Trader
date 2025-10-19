#!/usr/bin/env python3
"""
🗄️ PRODUCTION DATA LAYER
Unified async interface for scalable storage and streaming:
- PostgreSQL/TimescaleDB (time-series storage)
- Redis (real-time caching)
- Kafka (streaming)
- ClickHouse (analytics)

All integrations are optional with safe fallbacks so the system runs even if
packages or services are unavailable.
"""
import os
import json
import asyncio
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import asyncpg  # PostgreSQL / TimescaleDB
    ASYNCPG_AVAILABLE = True
except Exception:
    ASYNCPG_AVAILABLE = False

try:
    import redis.asyncio as aioredis  # Redis
    AIOREDIS_AVAILABLE = True
except Exception:
    AIOREDIS_AVAILABLE = False

try:
    from aiokafka import AIOKafkaProducer  # Kafka
    AIOKAFKA_AVAILABLE = True
except Exception:
    AIOKAFKA_AVAILABLE = False

try:
    import clickhouse_connect  # ClickHouse (sync client)
    CLICKHOUSE_AVAILABLE = True
except Exception:
    CLICKHOUSE_AVAILABLE = False

class ProductionDataLayer:
    """Scalable production data layer with optional backends"""

    def __init__(self, config_path: str = "config/data_layer.json"):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.pg_pool = None
        self.redis: Optional[aioredis.Redis] = None if AIOREDIS_AVAILABLE else None
        self.kafka_producer: Optional[AIOKafkaProducer] = None
        self.clickhouse_client = None
        self.is_initialized = False

        # Load or create default config
        self._load_config()

    def _load_config(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                self._create_default_config()
        except Exception as e:
            logger.error(f"Failed to load data layer config: {e}")
            self._create_default_config()

    def _create_default_config(self):
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        self.config = {
            "postgres": {
                "enabled": False,
                "dsn": os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/poise"),
                "timescaledb": True
            },
            "redis": {
                "enabled": False,
                "url": os.getenv("REDIS_URL", "redis://localhost:6379/0")
            },
            "kafka": {
                "enabled": False,
                "bootstrap_servers": os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
                "client_id": "poise_trader"
            },
            "clickhouse": {
                "enabled": False,
                "host": os.getenv("CLICKHOUSE_HOST", "localhost"),
                "port": int(os.getenv("CLICKHOUSE_PORT", "8123")),
                "username": os.getenv("CLICKHOUSE_USER", "default"),
                "password": os.getenv("CLICKHOUSE_PASSWORD", "")
            }
        }
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"📝 Created default data layer config at {self.config_path}")

    async def initialize(self):
        """Initialize all configured backends"""
        if self.is_initialized:
            return True

        # PostgreSQL / TimescaleDB
        if self.config.get("postgres", {}).get("enabled") and ASYNCPG_AVAILABLE:
            try:
                dsn = self.config["postgres"]["dsn"]
                self.pg_pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5)
                await self._init_pg_schema()
                logger.info("✅ PostgreSQL/TimescaleDB ready")
            except Exception as e:
                logger.warning(f"⚠️ PostgreSQL init failed: {e}")
                self.pg_pool = None
        elif self.config.get("postgres", {}).get("enabled"):
            logger.warning("⚠️ asyncpg not installed; skipping PostgreSQL")

        # Redis
        if self.config.get("redis", {}).get("enabled") and AIOREDIS_AVAILABLE:
            try:
                self.redis = aioredis.from_url(self.config["redis"]["url"], decode_responses=True)
                await self.redis.ping()
                logger.info("✅ Redis connected")
            except Exception as e:
                logger.warning(f"⚠️ Redis init failed: {e}")
                self.redis = None
        elif self.config.get("redis", {}).get("enabled"):
            logger.warning("⚠️ redis-py (async) not installed; skipping Redis")

        # Kafka
        if self.config.get("kafka", {}).get("enabled") and AIOKAFKA_AVAILABLE:
            try:
                self.kafka_producer = AIOKafkaProducer(
                    bootstrap_servers=self.config["kafka"]["bootstrap_servers"],
                    client_id=self.config["kafka"].get("client_id", "poise_trader")
                )
                await self.kafka_producer.start()
                logger.info("✅ Kafka producer started")
            except Exception as e:
                logger.warning(f"⚠️ Kafka init failed: {e}")
                self.kafka_producer = None
        elif self.config.get("kafka", {}).get("enabled"):
            logger.warning("⚠️ aiokafka not installed; skipping Kafka")

        # ClickHouse
        if self.config.get("clickhouse", {}).get("enabled") and CLICKHOUSE_AVAILABLE:
            try:
                self.clickhouse_client = clickhouse_connect.get_client(
                    host=self.config["clickhouse"]["host"],
                    port=self.config["clickhouse"]["port"],
                    username=self.config["clickhouse"]["username"],
                    password=self.config["clickhouse"]["password"],
                )
                await self._init_clickhouse_schema()
                logger.info("✅ ClickHouse client ready")
            except Exception as e:
                logger.warning(f"⚠️ ClickHouse init failed: {e}")
                self.clickhouse_client = None
        elif self.config.get("clickhouse", {}).get("enabled"):
            logger.warning("⚠️ clickhouse-connect not installed; skipping ClickHouse")

        self.is_initialized = True
        return True

    async def _init_pg_schema(self):
        if not self.pg_pool:
            return
        async with self.pg_pool.acquire() as conn:
            # Attempt to enable TimescaleDB (ignore errors if not superuser)
            if self.config.get("postgres", {}).get("timescaledb", True):
                try:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                except Exception:
                    pass
            # Create tables
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS social_sentiment (
                    ts TIMESTAMPTZ NOT NULL,
                    platform TEXT,
                    symbol TEXT,
                    sentiment_score DOUBLE PRECISION,
                    mention_count INT,
                    engagement_score DOUBLE PRECISION,
                    trending_score DOUBLE PRECISION,
                    key_topics JSONB,
                    confidence DOUBLE PRECISION
                );
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS onchain_data (
                    ts TIMESTAMPTZ NOT NULL,
                    symbol TEXT,
                    active_addresses BIGINT,
                    transaction_count BIGINT,
                    large_transactions BIGINT,
                    exchange_inflows DOUBLE PRECISION,
                    exchange_outflows DOUBLE PRECISION,
                    whale_activity_score DOUBLE PRECISION,
                    hodl_waves JSONB,
                    fear_greed_index DOUBLE PRECISION
                );
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS economic_data (
                    ts TIMESTAMPTZ NOT NULL,
                    event_name TEXT,
                    impact TEXT,
                    actual DOUBLE PRECISION,
                    forecast DOUBLE PRECISION,
                    previous DOUBLE PRECISION,
                    currency TEXT,
                    market_impact_score DOUBLE PRECISION
                );
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS options_data (
                    ts TIMESTAMPTZ NOT NULL,
                    symbol TEXT,
                    total_volume DOUBLE PRECISION,
                    put_call_ratio DOUBLE PRECISION,
                    max_pain DOUBLE PRECISION,
                    gamma_exposure DOUBLE PRECISION,
                    implied_volatility DOUBLE PRECISION,
                    skew DOUBLE PRECISION,
                    term_structure JSONB,
                    unusual_activity JSONB
                );
            """)
            # Create hypertables if possible
            if self.config.get("postgres", {}).get("timescaledb", True):
                for table in ["social_sentiment", "onchain_data", "economic_data", "options_data"]:
                    try:
                        await conn.execute(
                            f"SELECT create_hypertable('{table}', 'ts', if_not_exists => TRUE);"
                        )
                    except Exception:
                        pass

    async def _init_clickhouse_schema(self):
        if not self.clickhouse_client:
            return
        def _init():
            client = self.clickhouse_client
            client.command(
                """
                CREATE TABLE IF NOT EXISTS social_sentiment (
                    ts DateTime,
                    platform String,
                    symbol String,
                    sentiment_score Float64,
                    mention_count Int64,
                    engagement_score Float64,
                    trending_score Float64,
                    key_topics Array(String),
                    confidence Float64
                ) ENGINE = MergeTree ORDER BY (symbol, ts)
                """
            )
            client.command(
                """
                CREATE TABLE IF NOT EXISTS onchain_data (
                    ts DateTime,
                    symbol String,
                    active_addresses Int64,
                    transaction_count Int64,
                    large_transactions Int64,
                    exchange_inflows Float64,
                    exchange_outflows Float64,
                    whale_activity_score Float64,
                    hodl_waves String,
                    fear_greed_index Float64
                ) ENGINE = MergeTree ORDER BY (symbol, ts)
                """
            )
            client.command(
                """
                CREATE TABLE IF NOT EXISTS economic_data (
                    ts DateTime,
                    event_name String,
                    impact String,
                    actual Nullable(Float64),
                    forecast Nullable(Float64),
                    previous Nullable(Float64),
                    currency String,
                    market_impact_score Float64
                ) ENGINE = MergeTree ORDER BY (event_name, ts)
                """
            )
            client.command(
                """
                CREATE TABLE IF NOT EXISTS options_data (
                    ts DateTime,
                    symbol String,
                    total_volume Float64,
                    put_call_ratio Float64,
                    max_pain Float64,
                    gamma_exposure Float64,
                    implied_volatility Float64,
                    skew Float64,
                    term_structure String,
                    unusual_activity String
                ) ENGINE = MergeTree ORDER BY (symbol, ts)
                """
            )
        await asyncio.to_thread(_init)

    def _to_dict(self, obj: Any) -> Dict[str, Any]:
        if is_dataclass(obj):
            d = asdict(obj)
        elif isinstance(obj, dict):
            d = obj.copy()
        else:
            # Fallback best-effort mapping
            d = {k: getattr(obj, k) for k in dir(obj) if not k.startswith('_') and not callable(getattr(obj, k))}
        # Normalize timestamp field name
        if 'timestamp' in d:
            d['ts'] = d.pop('timestamp')
        return d

    async def _pg_insert(self, table: str, payload: Dict[str, Any]):
        if not self.pg_pool:
            return
        cols = list(payload.keys())
        vals = [payload[c] for c in cols]
        placeholders = ','.join(f'${i+1}' for i in range(len(cols)))
        sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
        async with self.pg_pool.acquire() as conn:
            await conn.execute(sql, *vals)

    async def _kafka_send(self, topic: str, payload: Dict[str, Any]):
        if not self.kafka_producer:
            return
        try:
            await self.kafka_producer.send_and_wait(topic, json.dumps(payload, default=str).encode('utf-8'))
        except Exception as e:
            logger.warning(f"Kafka send failed for {topic}: {e}")

    async def _ch_insert(self, table: str, payload: Dict[str, Any]):
        if not self.clickhouse_client:
            return
        cols = list(payload.keys())
        data = [payload]
        def _insert():
            self.clickhouse_client.insert(table, data, column_names=cols)
        await asyncio.to_thread(_insert)

    # Public write APIs -------------------------------------------------------
    async def write_social_sentiment(self, data: Any):
        payload = self._to_dict(data)
        try:
            await asyncio.gather(
                self._pg_insert('social_sentiment', payload) if self.pg_pool else asyncio.sleep(0),
                self._kafka_send('social_sentiment', payload) if self.kafka_producer else asyncio.sleep(0),
                self._ch_insert('social_sentiment', payload) if self.clickhouse_client else asyncio.sleep(0),
            )
            # Lightweight Redis cache (last value per symbol)
            if self.redis and payload.get('symbol'):
                await self.redis.hset(f"sentiment:{payload['symbol']}", mapping={
                    'sentiment_score': payload.get('sentiment_score', 0),
                    'ts': str(payload.get('ts'))
                })
        except Exception as e:
            logger.warning(f"write_social_sentiment failed: {e}")

    async def write_onchain_data(self, data: Any):
        payload = self._to_dict(data)
        try:
            await asyncio.gather(
                self._pg_insert('onchain_data', payload) if self.pg_pool else asyncio.sleep(0),
                self._kafka_send('onchain', payload) if self.kafka_producer else asyncio.sleep(0),
                self._ch_insert('onchain_data', payload) if self.clickhouse_client else asyncio.sleep(0),
            )
        except Exception as e:
            logger.warning(f"write_onchain_data failed: {e}")

    async def write_economic_event(self, data: Any):
        payload = self._to_dict(data)
        try:
            await asyncio.gather(
                self._pg_insert('economic_data', payload) if self.pg_pool else asyncio.sleep(0),
                self._kafka_send('economic_events', payload) if self.kafka_producer else asyncio.sleep(0),
                self._ch_insert('economic_data', payload) if self.clickhouse_client else asyncio.sleep(0),
            )
        except Exception as e:
            logger.warning(f"write_economic_event failed: {e}")

    async def write_options_data(self, data: Any):
        payload = self._to_dict(data)
        try:
            await asyncio.gather(
                self._pg_insert('options_data', payload) if self.pg_pool else asyncio.sleep(0),
                self._kafka_send('options_market', payload) if self.kafka_producer else asyncio.sleep(0),
                self._ch_insert('options_data', payload) if self.clickhouse_client else asyncio.sleep(0),
            )
        except Exception as e:
            logger.warning(f"write_options_data failed: {e}")

    async def close(self):
        try:
            if self.kafka_producer:
                await self.kafka_producer.stop()
        except Exception:
            pass
        try:
            if self.redis:
                await self.redis.close()
        except Exception:
            pass
        try:
            if self.pg_pool:
                await self.pg_pool.close()
        except Exception:
            pass
        self.is_initialized = False

# Global instance
data_layer = ProductionDataLayer()
