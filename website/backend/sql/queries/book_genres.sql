-- name: AddBookGenre :exec
INSERT INTO BookGenres (book_id, genre_id)
VALUES ($1, $2);

-- name: RemoveBookGenre :exec
DELETE FROM BookGenres
WHERE book_id = $1 AND genre_id = $2;

-- name: GetGenresForBook :many
SELECT g.id, g.name
FROM Genres g
JOIN BookGenres bg ON g.id = bg.genre_id
WHERE bg.book_id = $1
ORDER BY g.name;

-- name: GetBooksForGenre :many
SELECT b.id, b.title
FROM Books b
JOIN BookGenres bg ON b.id = bg.book_id
WHERE bg.genre_id = $1
ORDER BY b.title;