-- name: CreateBook :one
INSERT INTO Books (
    goodreads_id, goodreads_url, title, description, publication_year,
    cover_image_url, average_rating, ratings_count
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8
)
RETURNING *; -- Return all columns after creation

-- name: GetBookByID :one
SELECT id, goodreads_id, goodreads_url, title, description, publication_year, cover_image_url, average_rating, ratings_count, search_vector::text AS search_vector, created_at, updated_at
FROM Books
WHERE id = $1 LIMIT 1;

-- name: GetBookByGoodreadsID :one
SELECT id, goodreads_id, goodreads_url, title, description, publication_year, cover_image_url, average_rating, ratings_count, search_vector::text AS search_vector, created_at, updated_at
FROM Books
WHERE goodreads_id = $1 LIMIT 1;

-- name: ListBooks :many
SELECT id, goodreads_id, goodreads_url, title, description, publication_year, cover_image_url, average_rating, ratings_count, search_vector::text AS search_vector, created_at, updated_at
FROM Books
ORDER BY title
LIMIT $1 OFFSET $2;

-- name: SearchBooks :many
SELECT id, goodreads_id, goodreads_url, title, description, publication_year, cover_image_url, average_rating, ratings_count, search_vector::text AS search_vector, created_at, updated_at
FROM Books
WHERE search_vector @@ websearch_to_tsquery('english', $1)
ORDER BY ts_rank(search_vector, websearch_to_tsquery('english', $1)) DESC
LIMIT $2 OFFSET $3;

-- name: UpdateBook :one
UPDATE Books
SET
  goodreads_id = COALESCE(sqlc.arg(goodreads_id), goodreads_id),
  goodreads_url = COALESCE(sqlc.arg(goodreads_url), goodreads_url),
  title = COALESCE(sqlc.arg(title), title),
  description = COALESCE(sqlc.arg(description), description),
  publication_year = COALESCE(sqlc.arg(publication_year), publication_year),
  cover_image_url = COALESCE(sqlc.arg(cover_image_url), cover_image_url),
  average_rating = COALESCE(sqlc.arg(average_rating), average_rating),
  ratings_count = COALESCE(sqlc.arg(ratings_count), ratings_count),
  updated_at = CURRENT_TIMESTAMP -- Trigger handles search_vector update
WHERE id = sqlc.arg(id)
RETURNING *;

-- name: DeleteBook :exec
DELETE FROM Books
WHERE id = $1;

-- name: GetBooksByAuthor :many
SELECT b.id, b.goodreads_id, b.goodreads_url, b.title, b.description, b.publication_year, b.cover_image_url, b.average_rating, b.ratings_count, b.search_vector::text AS search_vector, b.created_at, b.updated_at
FROM Books b
JOIN BookAuthors ba ON b.id = ba.book_id
WHERE ba.author_id = $1
ORDER BY b.title;

-- name: GetBooksByGenre :many
SELECT b.id, b.goodreads_id, b.goodreads_url, b.title, b.description, b.publication_year, b.cover_image_url, b.average_rating, b.ratings_count, b.search_vector::text AS search_vector, b.created_at, b.updated_at
FROM Books b
JOIN BookGenres bg ON b.id = bg.book_id
WHERE bg.genre_id = $1
ORDER BY b.title;