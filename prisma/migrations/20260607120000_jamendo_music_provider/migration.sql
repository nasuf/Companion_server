ALTER TABLE "music_favorites"
    ALTER COLUMN "artist" SET DEFAULT 'Jamendo',
    ALTER COLUMN "album" SET DEFAULT 'Jamendo Library',
    ALTER COLUMN "library" SET DEFAULT 'focus',
    ALTER COLUMN "source" SET DEFAULT 'jamendo';

ALTER TABLE "music_playbacks"
    ALTER COLUMN "artist" SET DEFAULT 'Jamendo',
    ALTER COLUMN "album" SET DEFAULT 'Jamendo Library',
    ALTER COLUMN "library" SET DEFAULT 'focus',
    ALTER COLUMN "source" SET DEFAULT 'jamendo';

ALTER TABLE "music_co_listening_sessions"
    ALTER COLUMN "artist" SET DEFAULT 'Jamendo',
    ALTER COLUMN "album" SET DEFAULT 'Jamendo Library',
    ALTER COLUMN "library" SET DEFAULT 'focus',
    ALTER COLUMN "source" SET DEFAULT 'jamendo';
