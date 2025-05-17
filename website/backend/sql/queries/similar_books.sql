-- name: AddSimilarBook :exec
INSERT INTO SimilarBooks (book_id_1, book_id_2)
VALUES (
    CASE WHEN $1 < $2 THEN $1 ELSE $2 END,
    CASE WHEN $1 < $2 THEN $2 ELSE $1 END
); -- Ensure book_id_1 < book_id_2

-- name: RemoveSimilarBook :exec
DELETE FROM SimilarBooks
WHERE (book_id_1 = $1 AND book_id_2 = $2) OR (book_id_1 = $2 AND book_id_2 = $1);

-- name: GetSimilarBooks :many
-- This query just returns the IDs, keeping the simpler logic if you need just IDs
SELECT
    CASE WHEN book_id_1 = $1 THEN book_id_2 ELSE book_id_1 END AS similar_book_id
FROM SimilarBooks
WHERE book_id_1 = $1 OR book_id_2 = $1;

-- name: GetSimilarBookDetails :many
-- This query joins with Books to get full details
SELECT b.*
FROM Books b
JOIN (
    SELECT book_id_2 AS similar_id FROM SimilarBooks WHERE SimilarBooks.book_id_1 = $1
    UNION
    SELECT book_id_1 AS similar_id FROM SimilarBooks WHERE SimilarBooks.book_id_2 = $1
) AS similar_ids ON b.id = similar_ids.similar_id;