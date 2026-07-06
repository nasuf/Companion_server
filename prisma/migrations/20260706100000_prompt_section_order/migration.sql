-- Section order override for composite prompts (currently only chat.system_base).
-- No row = use the code-default order from section_order.CHAT_SECTION_SLOTS.
CREATE TABLE IF NOT EXISTS prompt_section_orders (
    prompt_key TEXT PRIMARY KEY,
    order_json JSONB NOT NULL,
    updated_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
