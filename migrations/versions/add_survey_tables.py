"""Add survey tables

Revision ID: add_survey_tables
Revises: 8f4b1c7c86fc
Create Date: 2025-12-11
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_survey_tables'
down_revision = '8f4b1c7c86fc'
branch_labels = None
depends_on = None


def upgrade():
    # Create survey_images table (if not exists)
    from sqlalchemy import inspect
    from alembic import context
    bind = context.get_bind()
    inspector = inspect(bind)
    existing_tables = inspector.get_table_names()

    if 'survey_images' not in existing_tables:
        op.create_table('survey_images',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('before_image', sa.String(length=255), nullable=False),
        sa.Column('after_image', sa.String(length=255), nullable=False),
        sa.Column('is_normalized', sa.Boolean(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )

    # Create survey_responses table (if not exists)
    if 'survey_responses' not in existing_tables:
        op.create_table('survey_responses',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('image_pair_id', sa.Integer(), nullable=False),
        sa.Column('response', sa.String(length=10), nullable=False),
        sa.Column('session_id', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('user_agent', sa.String(length=500), nullable=True),
            sa.ForeignKeyConstraint(['image_pair_id'], ['survey_images.id'], ),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index('ix_survey_responses_image_pair_id', 'survey_responses', ['image_pair_id'], unique=False)
        op.create_index('ix_survey_responses_session_id', 'survey_responses', ['session_id'], unique=False)


def downgrade():
    op.drop_index('ix_survey_responses_session_id', table_name='survey_responses')
    op.drop_index('ix_survey_responses_image_pair_id', table_name='survey_responses')
    op.drop_table('survey_responses')
    op.drop_table('survey_images')
