-- name: CreateReview :one
INSERT INTO Reviews (text, rating, book_id, user_id)
VALUES ($1, $2, $3, $4)
RETURNING *;

-- name: GetReviewByID :one
SELECT *
FROM Reviews
WHERE id = $1 LIMIT 1;

-- name: GetReviewsForBook :many
SELECT r.*, u.name AS user_name -- Include user name for context
FROM Reviews r
JOIN Users u ON r.user_id = u.id
WHERE r.book_id = $1
ORDER BY r.created_at DESC;

-- name: GetReviewsByUser :many
SELECT r.*, b.title AS book_title -- Include book title for context
FROM Reviews r
JOIN Books b ON r.book_id = b.id
WHERE r.user_id = $1
ORDER BY r.created_at DESC;

-- name: UpdateReview :one
UPDATE Reviews
SET
  text = COALESCE(sqlc.arg(text), text),
  rating = sqlc.arg(rating), -- Allow setting rating to NULL
  updated_at = CURRENT_TIMESTAMP
WHERE id = sqlc.arg(id) AND user_id = sqlc.arg(user_id) -- Ensure user owns the review
RETURNING *;

-- name: DeleteReview :exec
DELETE FROM Reviews
WHERE id = $1 AND user_id = $2; -- Ensure user owns the review