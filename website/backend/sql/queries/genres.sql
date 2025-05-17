-- name: CreateGenre :one
INSERT INTO Genres (name)
VALUES ($1)
RETURNING id, name;

-- name: GetGenreByID :one
SELECT id, name
FROM Genres
WHERE id = $1 LIMIT 1;

-- name: GetGenreByName :one
SELECT id, name
FROM Genres
WHERE name = $1 LIMIT 1;

-- name: ListGenres :many
SELECT id, name
FROM Genres
ORDER BY name;