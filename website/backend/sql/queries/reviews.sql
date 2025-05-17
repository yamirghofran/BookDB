-- name: GetReviewsByBookID :many
SELECT
    r.id,
    r.book_id,
    r.user_id,
    u.name AS user_name,
    -- Removed u.avatar_url as it does not exist in the Users table
    r.rating,
    r.text AS review_text, -- Corrected column name to r.text
    r.created_at,
    r.updated_at,
    COUNT(*) OVER() AS total_reviews -- Add total count for pagination
FROM Reviews r
JOIN Users u ON r.user_id = u.id
WHERE r.book_id = $1
ORDER BY r.created_at DESC
LIMIT $2
OFFSET $3;