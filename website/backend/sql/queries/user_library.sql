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
SELECT b.* -- Select book details
FROM Books b
JOIN UserLibrary ul ON b.id = ul.book_id
WHERE ul.user_id = $1
ORDER BY ul.added_at DESC;