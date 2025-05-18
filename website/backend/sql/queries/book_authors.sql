-- name: AddBookAuthor :exec
INSERT INTO BookAuthors (book_id, author_id)
VALUES ($1, $2);

-- name: RemoveBookAuthor :exec
DELETE FROM BookAuthors
WHERE book_id = $1 AND author_id = $2;

-- name: GetAuthorsForBook :many
SELECT a.id, a.name
FROM Authors a
JOIN BookAuthors ba ON a.id = ba.author_id
WHERE ba.book_id = $1
ORDER BY a.name;

-- name: GetBooksForAuthor :many
SELECT b.id, b.title
FROM Books b
JOIN BookAuthors ba ON b.id = ba.book_id
WHERE ba.author_id = $1
ORDER BY b.title;