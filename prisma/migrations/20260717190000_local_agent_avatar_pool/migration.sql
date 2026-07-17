WITH numbered AS (
  SELECT
    id,
    gender,
    (
      (('x' || substr(md5(id), 1, 8))::bit(32)::bigint %
        CASE
          WHEN gender = 'male' THEN 27
          WHEN gender = 'female' THEN 22
          ELSE 49
        END
      ) + 1
    )::int AS slot
  FROM "ai_agents"
),
mapped AS (
  SELECT
    id,
    CASE
      WHEN gender = 'male' THEN
        'companion-male-' || lpad(slot::text, 2, '0')
      WHEN gender = 'female' THEN
        'companion-female-' || lpad(slot::text, 2, '0')
      WHEN slot <= 27 THEN
        'companion-male-' || lpad(slot::text, 2, '0')
      ELSE
        'companion-female-' || lpad((slot - 27)::text, 2, '0')
    END AS avatar_key
  FROM numbered
)
UPDATE "ai_agents" AS agent
SET
  "avatar_key" = mapped.avatar_key,
  "avatar_url" = '/agents/avatar/' || mapped.avatar_key || '.png'
FROM mapped
WHERE agent.id = mapped.id;

DROP TABLE IF EXISTS "agent_avatar_cache";
