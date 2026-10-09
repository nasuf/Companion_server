-- Confirmed incident repair, intentionally bounded to reviewed row fingerprints.
-- No user memories, profile seeds, embeddings, or source chats are deleted.
-- Full original rows are stored atomically in memory_changelogs.old_value.
BEGIN;
SET LOCAL lock_timeout = '3s';
SET LOCAL statement_timeout = '15s';
WITH approved(id, content_md5, user_id, workspace_id, agent_id, city_md5) AS (VALUES
    ('f8814c46-f773-4f0a-be8f-c0eed64601cc','25a5b27f53211af7bfc01ce568593b8d','94d52301-2cae-4f47-880d-b9eefe366f2a','55838ef0854f4d0fa5d8eee426234b89','3ce97b0d-10e0-4c81-9022-bcccac70f50d','72740004c0c6832b8bf5d9d1402da0cf'),
    ('cebaecbb-4542-4e27-a9d7-1d641f1d59d2','a607f03feee43be9bb32eaf0bc5e7941','94d52301-2cae-4f47-880d-b9eefe366f2a','55838ef0854f4d0fa5d8eee426234b89','3ce97b0d-10e0-4c81-9022-bcccac70f50d','72740004c0c6832b8bf5d9d1402da0cf'),
    ('18b738ab-7a6e-401f-90e0-078740c4e015','6cbe25a6f6abbde570ff4ac449448f6a','94d52301-2cae-4f47-880d-b9eefe366f2a','55838ef0854f4d0fa5d8eee426234b89','3ce97b0d-10e0-4c81-9022-bcccac70f50d','72740004c0c6832b8bf5d9d1402da0cf'),
    ('3db259e7-71ce-4026-b961-dc873a051c06','0713b1fe06e178b5a939e2e80b899070','94d52301-2cae-4f47-880d-b9eefe366f2a','55838ef0854f4d0fa5d8eee426234b89','3ce97b0d-10e0-4c81-9022-bcccac70f50d','72740004c0c6832b8bf5d9d1402da0cf'),
    ('fe50a0ed-9080-4e25-9f98-1b2a29a5825d','edbf9cde0d886df59a031a0bafa4e12b','94d52301-2cae-4f47-880d-b9eefe366f2a','55838ef0854f4d0fa5d8eee426234b89','3ce97b0d-10e0-4c81-9022-bcccac70f50d','72740004c0c6832b8bf5d9d1402da0cf'),
    ('99da22b3-576e-4a83-8259-75e5e7d7a56f','098fb51379e428d8cbf58466a2cfb30c','94d52301-2cae-4f47-880d-b9eefe366f2a','55838ef0854f4d0fa5d8eee426234b89','3ce97b0d-10e0-4c81-9022-bcccac70f50d','72740004c0c6832b8bf5d9d1402da0cf'),
    ('f08c2416-7b11-494a-a0f5-3622a3059452','445605df936aa6fa02daa51b16b766c8','94d52301-2cae-4f47-880d-b9eefe366f2a','55838ef0854f4d0fa5d8eee426234b89','3ce97b0d-10e0-4c81-9022-bcccac70f50d','72740004c0c6832b8bf5d9d1402da0cf'),
    ('610a524f-ccf3-4cad-a54d-dcd86e6f3e82','7e782c215d87fca672fd2ba4d056cc37','94d52301-2cae-4f47-880d-b9eefe366f2a','55838ef0854f4d0fa5d8eee426234b89','3ce97b0d-10e0-4c81-9022-bcccac70f50d','72740004c0c6832b8bf5d9d1402da0cf'),
    ('8084959d-e673-4940-8704-cf1912845eb7','d3b21dbddb7c47d74c05c89ec61fa5d7','e511c3d1-293e-403e-8eac-db4ea0d0edd2','d9955afcbd5841348275ea0c7df85fd7','d54de45b-14a2-4351-9636-0586b6552b0f','88579f714b1946048c82579f6e09abfa'),
    ('f31e3126-bf38-4498-ac8c-83613e73e81c','f8fba6da17134083ddf8005908980f2a','078de6ff-bd7c-49cb-b536-47e22796480f','155645efd5034350ba9dcd48bff76ea8','fd1c43c2-ace8-4cc4-81b2-77b941895c7f','37d538f88214d8df9ee35324feda1f87')
), locked AS (
    SELECT m.* FROM memories_ai m
    JOIN approved p ON p.id=m.id AND p.content_md5=md5(m.content)
      AND p.user_id=m.user_id AND p.workspace_id=m.workspace_id
    JOIN chat_workspaces w ON w.id=m.workspace_id AND w.user_id=m.user_id
    JOIN ai_agents a ON a.id=w.agent_id AND a.user_id=m.user_id
    WHERE a.id=p.agent_id AND md5(a.city)=p.city_md5
      AND m.provenance IN ('ai_authored','daily_summary')
      AND NOT m.is_archived
    FOR UPDATE OF m
), audit AS (
    INSERT INTO memory_changelogs(id,user_id,workspace_id,memory_id,operation,old_value,new_value,created_at)
    SELECT 'persona-quarantine-20261009-' || id,user_id,workspace_id,id,
           'persona_quarantine',row_to_json(locked)::text,
           '{"is_archived":true,"reason":"confirmed_persona_location_conflict","batch":"MEMLOC-20261009"}',CURRENT_TIMESTAMP
    FROM locked RETURNING memory_id
)
UPDATE memories_ai m SET is_archived=TRUE,updated_at=CURRENT_TIMESTAMP
FROM audit WHERE m.id=audit.memory_id;
COMMIT;
