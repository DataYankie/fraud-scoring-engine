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

    op.create_table('user_risk_profiles',
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.Column('transaction_count_1h', sa.Integer(), nullable=False),
    sa.Column('total_spend_24h', sa.Numeric(precision=10, scale=2), nullable=False),
    sa.Column('distinct_ip_count_24h', sa.Integer(), nullable=False),
    sa.Column('risk_metadata', sa.JSON(), nullable=True),
    sa.PrimaryKeyConstraint('user_id')
    )

    op.create_table('transactions',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=False),
    sa.Column('amount', sa.Numeric(precision=10, scale=2), nullable=False),
    sa.Column('ip_address', sa.String(length=45), nullable=False),
    sa.Column('device_id', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['user_risk_profiles.user_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_transactions_timestamp'), 'transactions', ['timestamp'], unique=False)

    op.create_table('fraud_alerts',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('transaction_id', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('raw_ml_score', sa.Numeric(precision=4, scale=3), nullable=False),
    sa.Column('decision', sa.String(length=20), nullable=False),
    sa.Column('ai_analyst_reason', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['transaction_id'], ['transactions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_table('fraud_alerts')
    op.drop_index(op.f('ix_transactions_timestamp'), table_name='transactions')
    op.drop_table('transactions')
    op.drop_table('user_risk_profiles')

