"""Initial schema setup

Revision ID: 7c47273e8371
Revises:
Create Date: 2026-05-22 20:57:44.071903

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7c47273e8371'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.create_table(
        'transactions',
        sa.Column('transaction_id', sa.Integer(), nullable=False),
        sa.Column('is_fraud', sa.Integer(), nullable=True),
        sa.Column('transaction_amt', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('product_cd', sa.String(length=10), nullable=True),
        sa.Column('transaction_dt', sa.Integer(), nullable=False),
        sa.Column('transaction_at', sa.DateTime(), nullable=False),
        sa.Column('card1', sa.Float(), nullable=True),
        sa.Column('card2', sa.Float(), nullable=True),
        sa.Column('card3', sa.Float(), nullable=True),
        sa.Column('card4', sa.String(length=50), nullable=True),
        sa.Column('card5', sa.Float(), nullable=True),
        sa.Column('card6', sa.String(length=50), nullable=True),
        sa.Column('p_emaildomain', sa.String(length=100), nullable=True),
        sa.Column('r_emaildomain', sa.String(length=100), nullable=True),
        sa.Column('addr1', sa.Float(), nullable=True),
        sa.Column('addr2', sa.Float(), nullable=True),
        sa.Column('dist1', sa.Float(), nullable=True),
        sa.Column('dist2', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('transaction_id'),
    )
    op.create_index(op.f('ix_transactions_transaction_dt'), 'transactions', ['transaction_dt'], unique=False)
    op.create_index(op.f('ix_transactions_transaction_at'), 'transactions', ['transaction_at'], unique=False)

    op.create_table(
        'transaction_identities',
        sa.Column('transaction_id', sa.Integer(), nullable=False),
        sa.Column('id_30', sa.String(length=100), nullable=True),
        sa.Column('id_31', sa.String(length=100), nullable=True),
        sa.Column('device_type', sa.String(length=50), nullable=True),
        sa.Column('device_info', sa.String(length=100), nullable=True),
        sa.ForeignKeyConstraint(['transaction_id'], ['transactions.transaction_id']),
        sa.PrimaryKeyConstraint('transaction_id'),
    )

    op.create_table(
        'fraud_alerts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('transaction_id', sa.Integer(), nullable=False),
        sa.Column('predicted_ml_prob', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('decision', sa.String(length=20), nullable=False),
        sa.Column('ai_analyst_reason', sa.String(), nullable=False),
        sa.ForeignKeyConstraint(['transaction_id'], ['transactions.transaction_id']),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_table('fraud_alerts')
    op.drop_table('transaction_identities')
    op.drop_index(op.f('ix_transactions_transaction_at'), table_name='transactions')
    op.drop_index(op.f('ix_transactions_transaction_dt'), table_name='transactions')
    op.drop_table('transactions')
