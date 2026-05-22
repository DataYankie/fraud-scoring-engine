from datetime import datetime
from uuid import uuid4
from sqlalchemy import String, Numeric, DateTime, ForeignKey, Integer, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class UserRiskProfile(Base):
    __tablename__ = "user_risk_profiles"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Pre-aggregated behavioral features for fast ML inference
    transaction_count_1h: Mapped[int] = mapped_column(Integer, default=0)
    total_spend_24h: Mapped[float] = mapped_column(Numeric(10, 2), default=0.00)
    distinct_ip_count_24h: Mapped[int] = mapped_column(Integer, default=1)
    
    # Metadata for the agentic layer (e.g., trusted device tokens, geographical baselines)
    risk_metadata: Mapped[dict] = mapped_column(JSON, nullable=True)

    transactions: Mapped[list["Transaction"]] = relationship(back_populates="user_profile")


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("user_risk_profiles.user_id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True) # Indexed for fast time-window queries
    amount: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    
    # Device & Network telemetry data
    ip_address: Mapped[str] = mapped_column(String(45), nullable=False) # Supports IPv4/IPv6
    device_id: Mapped[str] = mapped_column(String, nullable=False)
    
    user_profile: Mapped["UserRiskProfile"] = relationship(back_populates="transactions")
    alert: Mapped["FraudAlert"] = relationship(back_populates="transaction")


class FraudAlert(Base):
    __tablename__ = "fraud_alerts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    transaction_id: Mapped[str] = mapped_column(String, ForeignKey("transactions.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Engine Pipeline Outputs
    raw_ml_score: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False) # e.g., 0.942
    decision: Mapped[str] = mapped_column(String(20), nullable=False) # ALLOW, REVIEW, BLOCK
    
    # The Agentic Moat: Natural language justification
    ai_analyst_reason: Mapped[str] = mapped_column(String, nullable=False)

    transaction: Mapped["Transaction"] = relationship(back_populates="alert")