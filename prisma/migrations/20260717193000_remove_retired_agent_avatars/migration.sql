WITH mapped AS (
  SELECT
    id,
    CASE
      WHEN "avatar_key" IN (
        'companion-male-14',
        'companion-male-17',
        'companion-male-24'
      ) THEN (
        ARRAY[
          'companion-male-01', 'companion-male-02', 'companion-male-03',
          'companion-male-04', 'companion-male-05', 'companion-male-06',
          'companion-male-07', 'companion-male-08', 'companion-male-09',
          'companion-male-10', 'companion-male-11', 'companion-male-12',
          'companion-male-13', 'companion-male-15', 'companion-male-16',
          'companion-male-18', 'companion-male-19', 'companion-male-20',
          'companion-male-21', 'companion-male-22', 'companion-male-23',
          'companion-male-25', 'companion-male-26', 'companion-male-27'
        ]
      )[
        ((('x' || substr(md5(id), 1, 8))::bit(32)::bigint % 24) + 1)::int
      ]
      ELSE (
        ARRAY[
          'companion-female-01', 'companion-female-02', 'companion-female-03',
          'companion-female-04', 'companion-female-05', 'companion-female-06',
          'companion-female-07', 'companion-female-08', 'companion-female-09',
          'companion-female-10', 'companion-female-11', 'companion-female-12',
          'companion-female-13', 'companion-female-14', 'companion-female-15',
          'companion-female-16', 'companion-female-17', 'companion-female-18',
          'companion-female-19', 'companion-female-20', 'companion-female-22'
        ]
      )[
        ((('x' || substr(md5(id), 1, 8))::bit(32)::bigint % 21) + 1)::int
      ]
    END AS avatar_key
  FROM "ai_agents"
  WHERE "avatar_key" IN (
    'companion-male-14',
    'companion-male-17',
    'companion-male-24',
    'companion-female-21'
  )
)
UPDATE "ai_agents" AS agent
SET
  "avatar_key" = mapped.avatar_key,
  "avatar_url" = '/agents/avatar/' || mapped.avatar_key || '.png'
FROM mapped
WHERE agent.id = mapped.id;
