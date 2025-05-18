-- name: CreateBook :one
INSERT INTO Books (
    goodreads_id, goodreads_url, title, description, publication_year,
    cover_image_url, average_rating, ratings_count
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8
)
RETURNING *; -- Return all columns after creation

-- name: GetBookByID :one
SELECT
    b.id, b.goodreads_id, b.goodreads_url, b.title, b.description, b.publication_year,
    b.cover_image_url, b.average_rating, b.ratings_count,
    b.search_vector::text AS search_vector, b.created_at, b.updated_at,
    COALESCE(ARRAY_AGG(DISTINCT a.name ORDER BY a.name) FILTER (WHERE a.name IS NOT NULL), '{}') AS authors,
    COALESCE(ARRAY_AGG(DISTINCT g.name ORDER BY g.name) FILTER (WHERE g.name IS NOT NULL), '{}') AS genres
FROM Books b
LEFT JOIN BookAuthors ba ON b.id = ba.book_id
LEFT JOIN Authors a ON ba.author_id = a.id
LEFT JOIN BookGenres bg ON b.id = bg.book_id
LEFT JOIN Genres g ON bg.genre_id = g.id
WHERE b.id = $1
GROUP BY b.id -- Assuming b.id is PK
LIMIT 1;

-- name: GetBookByGoodreadsID :one
SELECT id, goodreads_id, goodreads_url, title, description, publication_year, cover_image_url, average_rating, ratings_count, search_vector::text AS search_vector, created_at, updated_at
FROM Books
WHERE goodreads_id = $1 LIMIT 1;

-- name: ListBooks :many
WITH RankedBooks AS (
    SELECT
        b.id,
        b.goodreads_id,
        b.goodreads_url,
        b.title,
        b.description,
        b.publication_year,
        b.cover_image_url,
        b.average_rating,
        b.ratings_count,
        b.search_vector::text AS search_vector,
        b.created_at,
        b.updated_at,
        ROW_NUMBER() OVER(PARTITION BY b.title ORDER BY b.average_rating DESC NULLS LAST, b.ratings_count DESC NULLS LAST, b.id DESC) as rn
    FROM Books b
),
UniqueRankedBooks AS (
    SELECT * FROM RankedBooks WHERE rn = 1
)
SELECT
    urb.id,
    urb.goodreads_id,
    urb.goodreads_url,
    urb.title,
    urb.description,
    urb.publication_year,
    urb.cover_image_url,
    urb.average_rating,
    urb.ratings_count,
    urb.search_vector,
    urb.created_at,
    urb.updated_at,
    COALESCE(ARRAY_AGG(DISTINCT aut.name ORDER BY aut.name) FILTER (WHERE aut.name IS NOT NULL), '{}') AS authors
FROM UniqueRankedBooks urb
LEFT JOIN BookAuthors ba ON urb.id = ba.book_id
LEFT JOIN Authors aut ON ba.author_id = aut.id
GROUP BY
    urb.id, urb.goodreads_id, urb.goodreads_url, urb.title, urb.description, urb.publication_year,
    urb.cover_image_url, urb.average_rating, urb.ratings_count, urb.search_vector,
    urb.created_at, urb.updated_at -- Ensure all selected non-aggregated columns from urb are in GROUP BY
ORDER BY urb.average_rating DESC NULLS LAST, urb.ratings_count DESC NULLS LAST, urb.title
LIMIT $1 OFFSET $2;

-- name: SearchBooks :many
SELECT
    b.id, b.goodreads_id, b.goodreads_url, b.title, b.description, b.publication_year,
    b.cover_image_url, b.average_rating, b.ratings_count,
    b.search_vector::text AS search_vector, b.created_at, b.updated_at,
    COALESCE(ARRAY_AGG(DISTINCT a.name ORDER BY a.name) FILTER (WHERE a.name IS NOT NULL), '{}') AS authors
FROM Books b
LEFT JOIN BookAuthors ba ON b.id = ba.book_id
LEFT JOIN Authors a ON ba.author_id = a.id
WHERE b.search_vector @@ websearch_to_tsquery('english', $1)
GROUP BY b.id -- Assuming b.id is PK
ORDER BY ts_rank(b.search_vector, websearch_to_tsquery('english', $1)) DESC
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

-- name: GetBooksByGoodreadsIDs :many
SELECT
    b.id, b.goodreads_id, b.goodreads_url, b.title, b.description, b.publication_year,
    b.cover_image_url, b.average_rating, b.ratings_count,
    b.search_vector::text AS search_vector, b.created_at, b.updated_at,
    COALESCE(ARRAY_AGG(DISTINCT a.name ORDER BY a.name) FILTER (WHERE a.name IS NOT NULL), '{}') AS authors,
    COALESCE(ARRAY_AGG(DISTINCT g.name ORDER BY g.name) FILTER (WHERE g.name IS NOT NULL), '{}') AS genres
FROM Books b
LEFT JOIN BookAuthors ba ON b.id = ba.book_id
LEFT JOIN Authors a ON ba.author_id = a.id
LEFT JOIN BookGenres bg ON b.id = bg.book_id
LEFT JOIN Genres g ON bg.genre_id = g.id
WHERE b.goodreads_id = ANY(sqlc.arg(goodreads_ids)::bigint[])
GROUP BY b.id
ORDER BY b.goodreads_id;