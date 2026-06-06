from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    __tablename__ = "transactions"

    # In the dataset, TransactionID is the explicit primary key linking both files
    transaction_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    is_fraud: Mapped[int | None] = mapped_column(Integer, nullable=True)  # The target variable from the dataset

    # Core Tabular Features from IEEE-CIS
    transaction_amt: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    product_cd: Mapped[str | None] = mapped_column(String(10), nullable=True)
    transaction_dt: Mapped[int] = mapped_column(Integer, nullable=False, index=True)  # Raw seconds
    transaction_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)  # Derived datetime

    # Card details (card1 - card6)
    card1: Mapped[float | None] = mapped_column(Float, nullable=True)
    card2: Mapped[float | None] = mapped_column(Float, nullable=True)
    card3: Mapped[float | None] = mapped_column(Float, nullable=True)
    card4: Mapped[str | None] = mapped_column(String(50), nullable=True)  # e.g., "visa", "mastercard"
    card5: Mapped[float | None] = mapped_column(Float, nullable=True)
    card6: Mapped[str | None] = mapped_column(String(50), nullable=True)  # e.g., "debit", "credit"

    # Email domains
    p_emaildomain: Mapped[str | None] = mapped_column(String(100), nullable=True)
    r_emaildomain: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Address / distance signals
    addr1: Mapped[float | None] = mapped_column(Float, nullable=True)
    addr2: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist1: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist2: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    identity: Mapped["TransactionIdentity"] = relationship(back_populates="transaction", uselist=False)
    alert: Mapped["FraudAlert"] = relationship(back_populates="transaction")


class TransactionIdentity(Base):
    __tablename__ = "transaction_identities"

    transaction_id: Mapped[int] = mapped_column(Integer, ForeignKey("transactions.transaction_id"), primary_key=True)

    # Telemetry data from train_identity.csv
    id_30: Mapped[str | None] = mapped_column(String(100), nullable=True)  # e.g., OS Version "Windows 10"
    id_31: Mapped[str | None] = mapped_column(String(100), nullable=True)  # e.g., Browser "chrome 63.0"
    device_type: Mapped[str | None] = mapped_column(String(50), nullable=True)  # e.g., "desktop", "mobile"
    device_info: Mapped[str | None] = mapped_column(String(100), nullable=True)  # e.g., "Windows Id:Windows"

    transaction: Mapped["Transaction"] = relationship(back_populates="identity")


class FraudAlert(Base):
    __tablename__ = "fraud_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[int] = mapped_column(Integer, ForeignKey("transactions.transaction_id"), nullable=False)

    # Engine Pipeline Outputs
    predicted_ml_prob: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False)
    decision: Mapped[str] = mapped_column(String(20), nullable=False)  # ALLOW, REVIEW, BLOCK

    # The Agentic Output Moat
    ai_analyst_reason: Mapped[str] = mapped_column(String, nullable=False)

    transaction: Mapped["Transaction"] = relationship(back_populates="alert")
