ALTER TABLE "ai_agents" ADD COLUMN "avatar_key" TEXT;
ALTER TABLE "ai_agents" ADD COLUMN "avatar_url" TEXT;

WITH avatar_choices AS (
  SELECT
    id,
    CASE
      WHEN gender = 'male' THEN ARRAY[
        'bansheng-male-01',
        'bansheng-male-02',
        'bansheng-male-03',
        'bansheng-male-04',
        'bansheng-male-05',
        'bansheng-male-06'
      ]
      WHEN gender = 'female' THEN ARRAY[
        'bansheng-female-01',
        'bansheng-female-02',
        'bansheng-female-03',
        'bansheng-female-04',
        'bansheng-female-05',
        'bansheng-female-06'
      ]
      ELSE ARRAY[
        'bansheng-female-01',
        'bansheng-male-01',
        'bansheng-female-02',
        'bansheng-male-02',
        'bansheng-female-03',
        'bansheng-male-03'
      ]
    END AS keys
  FROM "ai_agents"
  WHERE "avatar_key" IS NULL
),
picked AS (
  SELECT
    id,
    keys[
      ((('x' || substr(md5(id), 1, 8))::bit(32)::bigint % array_length(keys, 1)) + 1)::int
    ] AS avatar_key
  FROM avatar_choices
)
UPDATE "ai_agents" AS a
SET
  "avatar_key" = p.avatar_key,
  "avatar_url" = 'https://api.dicebear.com/9.x/open-peeps/png?seed='
    || p.avatar_key
    || '&radius=50&size=128&backgroundType=gradientLinear&backgroundColor=b6e3f4,c0aede,d1d4f9,ffd5dc,ffdfbf&accessoriesProbability=20'
    || CASE
      WHEN p.avatar_key LIKE '%-male-%' THEN '&head=short1,short2,short3,short4,short5,flatTop,pomp,mohawk&facialHairProbability=12'
      ELSE '&head=long,longBangs,longCurly,bangs,bangs2,bun,bun2,buns,mediumStraight&facialHairProbability=0'
    END
FROM picked AS p
WHERE a.id = p.id;
