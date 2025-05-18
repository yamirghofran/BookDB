-- name: CreateUser :one
INSERT INTO Users (name, email)
VALUES ($1, $2)
RETURNING id, name, email, created_at, updated_at;

-- name: GetUserByID :one
SELECT id, name, email, created_at, updated_at
FROM Users
WHERE id = $1 LIMIT 1;

-- name: GetUserByEmail :one
SELECT id, name, email, created_at, updated_at
FROM Users
WHERE email = $1 LIMIT 1;

-- name: UpdateUser :one
UPDATE Users
SET
  name = COALESCE(sqlc.arg(name), name),
  email = COALESCE(sqlc.arg(email), email),
  updated_at = CURRENT_TIMESTAMP
WHERE id = sqlc.arg(id)
RETURNING id, name, email, created_at, updated_at;

-- name: DeleteUser :exec
DELETE FROM Users
WHERE id = $1;

-- name: ListUsers :many
SELECT id, name, email, created_at, updated_at
FROM Users
ORDER BY created_at
LIMIT $1 OFFSET $2;

-- name: GetUserLibraryDetails :many
SELECT
    b.id,
    b.goodreads_id, -- Added goodreads_id
    b.title,
    b.cover_image_url,
    COALESCE(ARRAY_AGG(DISTINCT a.name ORDER BY a.name) FILTER (WHERE a.name IS NOT NULL), '{}') AS authors
FROM UserLibrary ul
JOIN Books b ON ul.book_id = b.id
LEFT JOIN BookAuthors ba ON b.id = ba.book_id
LEFT JOIN Authors a ON ba.author_id = a.id
WHERE ul.user_id = $1
GROUP BY b.id, ul.added_at -- Assuming b.id is PK, order by when user added to library
ORDER BY ul.added_at DESC;

-- name: GetUserReviewsWithBookInfo :many
SELECT
    r.id AS review_id,
    r.book_id,
    b.title AS book_title,
    b.cover_image_url AS book_cover_image_url, -- Added book cover
    r.user_id,
    r.rating,
    r.text AS review_text, -- Corrected column name from review_text to text
    r.created_at AS review_created_at,
    r.updated_at AS review_updated_at
FROM Reviews r
JOIN Books b ON r.book_id = b.id
WHERE r.user_id = $1
ORDER BY r.updated_at DESC;