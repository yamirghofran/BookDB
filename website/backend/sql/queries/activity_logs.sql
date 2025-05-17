-- name: LogActivity :one
INSERT INTO ActivityLogs (user_id, activity_type, target_id, details)
VALUES ($1, $2, $3, $4)
RETURNING *;

-- name: GetActivityLogsForUser :many
SELECT *
FROM ActivityLogs
WHERE user_id = $1
ORDER BY created_at DESC
LIMIT $2 OFFSET $3;

-- name: GetActivityLogsByType :many
SELECT *
FROM ActivityLogs
WHERE activity_type = $1
ORDER BY created_at DESC
LIMIT $2 OFFSET $3;