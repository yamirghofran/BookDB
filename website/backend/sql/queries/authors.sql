-- name: CreateAuthor :one
INSERT INTO Authors (name, average_rating, ratings_count)
VALUES ($1, $2, $3)
RETURNING id, name, average_rating, ratings_count, created_at, updated_at;

-- name: GetAuthorByID :one
SELECT id, name, average_rating, ratings_count, created_at, updated_at
FROM Authors
WHERE id = $1 LIMIT 1;

-- name: GetAuthorByName :many
SELECT id, name, average_rating, ratings_count, created_at, updated_at
FROM Authors
WHERE name ILIKE '%' || $1 || '%'
ORDER BY name;

-- name: UpdateAuthor :one
UPDATE Authors
SET
  name = COALESCE(sqlc.arg(name), name),
  average_rating = COALESCE(sqlc.arg(average_rating), average_rating),
  ratings_count = COALESCE(sqlc.arg(ratings_count), ratings_count),
  updated_at = CURRENT_TIMESTAMP
WHERE id = sqlc.arg(id)
RETURNING id, name, average_rating, ratings_count, created_at, updated_at;

-- name: DeleteAuthor :exec
DELETE FROM Authors
WHERE id = $1;

-- name: ListAuthors :many
SELECT id, name, average_rating, ratings_count, created_at, updated_at
FROM Authors
ORDER BY name
LIMIT $1 OFFSET $2;