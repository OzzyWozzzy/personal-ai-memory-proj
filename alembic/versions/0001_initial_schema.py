"""Local temporal-memory schema.

Revision ID: 0001
Revises:
"""
from __future__ import annotations

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

DDL = r"""
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='app_user') THEN
    CREATE ROLE app_user NOLOGIN NOSUPERUSER NOBYPASSRLS NOINHERIT;
  END IF;
  ALTER ROLE app_user WITH NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE
    NOREPLICATION NOINHERIT;
END
$$;

REVOKE CREATE ON SCHEMA public FROM PUBLIC, app_user;
CREATE SCHEMA IF NOT EXISTS private;
REVOKE ALL ON SCHEMA private FROM PUBLIC;

CREATE TABLE tenant_settings (
  tenant_id UUID PRIMARY KEY,
  timezone TEXT NOT NULL DEFAULT 'UTC',
  locale TEXT NOT NULL DEFAULT 'en',
  active BOOLEAN NOT NULL DEFAULT true,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE episodes (
  episode_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID NOT NULL,
  subject_scope UUID,
  source TEXT NOT NULL,
  source_ref TEXT,
  occurred_at TIMESTAMPTZ NOT NULL,
  recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  content TEXT,
  content_json JSONB,
  content_hash TEXT NOT NULL,
  meta JSONB NOT NULL DEFAULT '{}',
  legal_hold BOOLEAN NOT NULL DEFAULT false,
  CONSTRAINT episodes_content_present CHECK (
    (content IS NOT NULL) <> (content_json IS NOT NULL)
  ),
  CONSTRAINT episodes_tenant_hash_unique UNIQUE (tenant_id, content_hash)
);

CREATE TABLE entities (
  entity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID NOT NULL,
  entity_type TEXT NOT NULL,
  canonical_name TEXT NOT NULL,
  identifiers JSONB NOT NULL DEFAULT '{}',
  summary TEXT,
  summary_stale BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE entity_aliases (
  alias_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID NOT NULL,
  from_entity UUID NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
  to_entity UUID NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
  merged_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  merged_by TEXT NOT NULL,
  evidence JSONB NOT NULL DEFAULT '{}',
  retracted_at TIMESTAMPTZ
);

CREATE TABLE audit_log (
  audit_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  tenant_id UUID NOT NULL,
  at TIMESTAMPTZ NOT NULL DEFAULT now(),
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  detail JSONB NOT NULL
);

CREATE TABLE api_keys (
  key_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  key_hash BYTEA NOT NULL UNIQUE,
  key_prefix TEXT NOT NULL,
  tenant_id UUID NOT NULL,
  name TEXT NOT NULL,
  label TEXT NOT NULL DEFAULT '',
  scopes TEXT[] NOT NULL DEFAULT '{}',
  allowed_subject_scopes UUID[],
  active BOOLEAN NOT NULL DEFAULT true,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at TIMESTAMPTZ,
  last_used_at TIMESTAMPTZ
);

CREATE TABLE tenant_policies (
  tenant_id UUID PRIMARY KEY,
  retention JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX episodes_tenant_time_idx
  ON episodes (tenant_id, occurred_at DESC, episode_id);
CREATE INDEX entities_tenant_name_idx
  ON entities (tenant_id, lower(canonical_name));
CREATE INDEX audit_tenant_action_idx
  ON audit_log (tenant_id, action, audit_id);
CREATE INDEX api_keys_tenant_idx ON api_keys (tenant_id, created_at);

ALTER TABLE tenant_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenant_settings FORCE ROW LEVEL SECURITY;
ALTER TABLE episodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE episodes FORCE ROW LEVEL SECURITY;
ALTER TABLE entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE entities FORCE ROW LEVEL SECURITY;
ALTER TABLE entity_aliases ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_aliases FORCE ROW LEVEL SECURITY;
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log FORCE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys FORCE ROW LEVEL SECURITY;
ALTER TABLE tenant_policies ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenant_policies FORCE ROW LEVEL SECURITY;

CREATE POLICY tenant_settings_isolation ON tenant_settings
  USING (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
  WITH CHECK (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);
CREATE POLICY episodes_isolation ON episodes
  USING (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
  WITH CHECK (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);
CREATE POLICY entities_isolation ON entities
  USING (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
  WITH CHECK (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);
CREATE POLICY entity_aliases_isolation ON entity_aliases
  USING (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
  WITH CHECK (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);
CREATE POLICY audit_log_isolation ON audit_log
  USING (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
  WITH CHECK (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);
CREATE POLICY api_keys_isolation ON api_keys
  USING (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
  WITH CHECK (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);
CREATE POLICY tenant_policies_isolation ON tenant_policies
  USING (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
  WITH CHECK (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);

GRANT USAGE ON SCHEMA public, private TO app_user;
GRANT SELECT ON tenant_settings TO app_user;
GRANT SELECT, INSERT, DELETE ON episodes TO app_user;
GRANT UPDATE (legal_hold) ON episodes TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON entities, entity_aliases TO app_user;
GRANT SELECT, INSERT ON audit_log TO app_user;
GRANT SELECT, INSERT, UPDATE ON tenant_policies TO app_user;
GRANT SELECT (key_id, key_prefix, tenant_id, name, label, scopes,
  allowed_subject_scopes, active, created_at, expires_at, last_used_at)
  ON api_keys TO app_user;
GRANT INSERT (key_hash, key_prefix, tenant_id, name, label, scopes,
  allowed_subject_scopes, active, expires_at) ON api_keys TO app_user;
GRANT UPDATE (active) ON api_keys TO app_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;

CREATE OR REPLACE FUNCTION private.resolve_api_key_hash(p_key_hash BYTEA)
RETURNS TABLE (
  tenant_id UUID,
  key_id UUID,
  key_kind TEXT,
  key_label TEXT,
  scopes TEXT[],
  allowed_subject_scopes UUID[],
  timezone TEXT,
  locale TEXT,
  expires_at TIMESTAMPTZ
)
LANGUAGE sql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
  WITH matched AS (
    UPDATE api_keys AS key
       SET last_used_at = CASE
         WHEN key.last_used_at IS NULL OR key.last_used_at < now() - interval '5 minutes'
         THEN now() ELSE key.last_used_at END
     WHERE key.key_hash = p_key_hash
       AND key.active
       AND (key.expires_at IS NULL OR key.expires_at > now())
     RETURNING key.*
  )
  SELECT matched.tenant_id, matched.key_id, matched.name, matched.label,
         matched.scopes, matched.allowed_subject_scopes,
         settings.timezone, settings.locale, matched.expires_at
    FROM matched
    JOIN tenant_settings AS settings USING (tenant_id)
   WHERE settings.active
   LIMIT 1
$$;
REVOKE ALL ON FUNCTION private.resolve_api_key_hash(BYTEA) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION private.resolve_api_key_hash(BYTEA) TO app_user;
"""


def upgrade() -> None:
    op.execute(DDL)


def downgrade() -> None:
    op.execute(
        """
        DROP FUNCTION IF EXISTS private.resolve_api_key_hash(BYTEA);
        DROP TABLE IF EXISTS tenant_policies, api_keys, audit_log,
          entity_aliases, entities, episodes, tenant_settings CASCADE;
        """
    )
