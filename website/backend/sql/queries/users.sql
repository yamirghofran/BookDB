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