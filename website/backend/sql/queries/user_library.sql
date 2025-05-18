-- name: AddBookToLibrary :exec
INSERT INTO UserLibrary (user_id, book_id)
VALUES ($1, $2);

-- name: RemoveBookFromLibrary :exec
DELETE FROM UserLibrary
WHERE user_id = $1 AND book_id = $2;

-- name: IsBookInLibrary :one
SELECT EXISTS (
    SELECT 1
    FROM UserLibrary
    WHERE user_id = $1 AND book_id = $2
);

-- name: GetUserLibrary :many
SELECT b.id, b.goodreads_id, b.goodreads_url, b.title, b.description, b.publication_year, b.cover_image_url, b.average_rating, b.ratings_count, b.search_vector::text AS search_vector, b.created_at, b.updated_at -- Select book details
FROM Books b
JOIN UserLibrary ul ON b.id = ul.book_id
WHERE ul.user_id = $1
ORDER BY ul.added_at DESC;

-- name: GetUsersByBookInLibrary :many
SELECT
    u.id,
    u.name,
    -- Add u.avatar_url here if you add it to the Users table later
    COUNT(*) OVER() as total_users
FROM Users u
JOIN UserLibrary ul ON u.id = ul.user_id
WHERE ul.book_id = $1
ORDER BY u.name -- Or by ul.added_at if preferred
LIMIT $2
OFFSET $3;