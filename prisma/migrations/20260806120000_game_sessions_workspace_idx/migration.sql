-- 主动交流每分钟要判断"用户是不是还在"，而一起玩游戏也算在。那条判断按
-- workspace 查 game_sessions 的最后活动时间，此前没有可用索引，走的是全表扫描
-- (EXPLAIN: Seq Scan, 770 行 0.8ms)。两个用户 14 天就产生了 770 局，公开发布后
-- 这张表会很快长到让每分钟的顺序扫描变成真问题。
--
-- 只索引 workspace_id + started_at：查询里的时间条件是
-- COALESCE(ended_at, updated_at, started_at)，表达式索引匹配不上，但先按
-- workspace 收敛到几十行之后再过滤时间已经足够快。
CREATE INDEX IF NOT EXISTS "game_sessions_workspace_started_idx"
    ON "game_sessions" ("workspace_id", "started_at" DESC);
