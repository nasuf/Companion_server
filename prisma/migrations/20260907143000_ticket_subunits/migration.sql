-- Store ticket balances in 0.1-ticket subunits (×10) so chat overage can debit
-- 0.5 / 0.3 per message precisely instead of batching whole tickets.
UPDATE user_wallets
SET ticket_balance = ticket_balance * 10,
    gift_ticket_balance = gift_ticket_balance * 10,
    overage_accrued = 0;

UPDATE wallet_ledger
SET delta = delta * 10,
    balance_after = balance_after * 10
WHERE currency IN ('ticket', 'gift_ticket');
